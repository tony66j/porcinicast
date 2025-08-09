from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import math, os, asyncio
import httpx

APP = FastAPI(title="TrovaPorcini v3.0 API")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)
app = APP  # <- uvicorn main:app

HEAD = {"User-Agent": "TrovaPorcini-3.0 (+https://example.org)"}
OWM_KEY = os.getenv("OPENWEATHER_API_KEY")

# ----------------------- util -----------------------
def clamp(x, a, b): return max(a, min(b, x))

def aspect_octant(deg: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    i = int((deg % 360)/45.0 + 0.5) % 8
    return dirs[i]

def days_since_last_rain(past_mm: List[float]) -> Optional[int]:
    # past_mm[0]= ieri, 1 = 2 giorni fa, ... (solo passati)
    for d, mm in enumerate(past_mm):
        if (mm or 0.0) > 0.2:
            return d+1
    return None

def next_rain_from_daily(fut_mm: List[float]) -> Optional[Tuple[int, float]]:
    # fut_mm[0]= oggi, 1=domani...
    for d, mm in enumerate(fut_mm):
        if (mm or 0.0) >= 1.0:
            return d, float(mm or 0.0)
    return None

def hargreaves_et0(tmin, tmax, tmean, lat_deg: float) -> float:
    tr = max(0.0, (tmax - tmin))
    phi = math.radians(lat_deg)
    r_a = 25 + 10*abs(math.sin(phi))  # grezza ma abbastanza per pesi relativi
    et0 = 0.0023 * (tmean + 17.8) * math.sqrt(tr) * (r_a / 2.45)
    return clamp(et0, 0.0, 12.0)

def forest_label_from_osm_kind(kind: Optional[str], alt_m: float) -> str:
    if kind == "coniferous": return "Pinus/Abies/Picea"
    if kind == "broadleaved":
        if alt_m > 900: return "Fagus sylvatica"
        if 500 < alt_m <= 900: return "Castanea sativa"
        return "Quercus spp."
    if alt_m > 1400: return "Pinus/Abies/Picea"
    if alt_m > 900:  return "Fagus sylvatica"
    if alt_m > 500:  return "Castanea sativa"
    return "Quercus spp."

def porcini_species(lat: float, alt_m: float, forest: str, tmean: float) -> List[str]:
    S = []
    if "Castanea" in forest or "Quercus" in forest:
        if lat < 43.5 and 12 <= tmean <= 22 and 300 <= alt_m <= 1100:
            S.append("Boletus aereus")
        S.append("Boletus reticulatus")
    if "Fagus" in forest or alt_m >= 900:
        S.append("Boletus edulis")
    if "Pinus" in forest or "Abies" in forest or "Picea" in forest:
        S.append("Boletus pinophilus")
    # unici
    out, seen = [], set()
    for x in S:
        if x not in seen:
            out.append(x); seen.add(x)
    return out or ["Boletus edulis"]

def seasonal_alt_window(lat: float) -> Tuple[int,int]:
    if lat > 44.5: base = 1100
    elif lat > 41.5: base = 900
    else: base = 700
    return base-450, base+350

def slope_aspect_from_grid(z: List[List[float]], lat: float, cell_m: float=30.0) -> Tuple[float,float]:
    # Horn (1981) su griglia 3x3
    z00,z01,z02 = z[0]
    z10,z11,z12 = z[1]
    z20,z21,z22 = z[2]
    dx = ((z02 + 2*z12 + z22) - (z00 + 2*z10 + z20)) / (8*cell_m)
    dy = ((z20 + 2*z21 + z22) - (z00 + 2*z01 + z02)) / (8*cell_m)
    slope = math.degrees(math.atan(math.sqrt(dx*dx + dy*dy)))
    aspect = math.degrees(math.atan2(dy, -dx))
    if aspect < 0: aspect += 360.0
    return slope, aspect

# ----------------------- providers -----------------------
async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format":"json", "limit":1, "addressdetails":1}
    async with httpx.AsyncClient(timeout=20, headers=HEAD) as cli:
        r = await cli.get(url, params=params); r.raise_for_status()
        j = r.json()
    if not j: raise httpx.HTTPError("Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0].get("display_name","")}

async def open_meteo_daily(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "daily": ",".join([
            "precipitation_sum","temperature_2m_mean","temperature_2m_max","temperature_2m_min"
        ]),
        "past_days": 14, "forecast_days": 10
    }
    async with httpx.AsyncClient(timeout=25, headers=HEAD) as cli:
        r = await cli.get(url, params=params); r.raise_for_status()
        return r.json()

async def openweather_hourly(lat: float, lon: float) -> Optional[List[float]]:
    if not OWM_KEY: return None
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {"lat": lat, "lon": lon, "appid": OWM_KEY, "units":"metric", "exclude":"minutely,alerts,daily"}
    async with httpx.AsyncClient(timeout=25, headers=HEAD) as cli:
        r = await cli.get(url, params=params); r.raise_for_status()
        j = r.json()
    if "hourly" not in j: return None
    return [float(h.get("rain",{}).get("1h",0.0)) for h in j["hourly"][:48]]

async def overpass_forest(lat: float, lon: float) -> Optional[str]:
    query = f"""
    [out:json][timeout:25];
    (
      way(around:800,{lat},{lon})[natural=wood];
      relation(around:800,{lat},{lon})[natural=wood];
    );out tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    async with httpx.AsyncClient(timeout=30, headers=HEAD) as cli:
        r = await cli.post(url, data={"data": query}); r.raise_for_status()
        j = r.json()
    labels=[]
    for el in j.get("elements",[]):
        tags = el.get("tags",{})
        if "leaf_type" in tags:
            lt = tags["leaf_type"].lower()
            if "broad" in lt: labels.append("broadleaved")
            elif "conifer" in lt: labels.append("coniferous")
        elif "wood" in tags:
            w = tags["wood"]
            if w in ("deciduous","broadleaved"): labels.append("broadleaved")
            if w in ("coniferous","needleleaved"): labels.append("coniferous")
    if labels:
        b = labels.count("broadleaved"); c = labels.count("coniferous")
        return "broadleaved" if b>=c else "coniferous"
    return None

async def elev_grid_3x3(lat: float, lon: float, step_m: float=30.0) -> List[List[float]]:
    # converte metri in gradi locali
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)))
    dlat = step_m * deg_per_m_lat
    dlon = step_m * deg_per_m_lon
    coords=[]
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            coords.append({"latitude": lat + dr*dlat, "longitude": lon + dc*dlon})
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=25, headers=HEAD) as cli:
        r = await cli.post(url, json={"locations": coords}); r.raise_for_status()
        j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]

# ----------------------- scoring -----------------------
def seasonal_altfit(lat: float, alt_m: float) -> float:
    lo, hi = seasonal_alt_window(lat)
    if alt_m<=lo-300 or alt_m>=hi+300: return 0.0
    if lo<=alt_m<=hi: return 1.0
    if alt_m<lo: return clamp((alt_m-(lo-300))/300, 0, 1)
    return clamp(((hi+300)-alt_m)/300, 0, 1)

def thermo_fit(t: float) -> float:
    if t<=6 or t>=26: return 0.0
    if 14<=t<=18: return 1.0
    if t<14: return (t-6)/(14-6)
    return (26-t)/(26-18)

def composite_score(
    p14_mm: float, tmean7: float, et0_7: float, alt_m: float,
    aspect_deg: float, forest: str, lat: float, rain_next10: List[float]
) -> Tuple[float, Dict[str,float]]:
    water = clamp( 0.6*clamp(p14_mm/40.0,0,1) + 0.4*clamp((p14_mm-0.8*et0_7)/40.0, -0.5,1), 0,1 )
    thermo = clamp(thermo_fit(tmean7), 0, 1)
    altfit = seasonal_altfit(lat, alt_m)
    # aspetto: preferenza nord nei periodi caldi
    def north_pref(a):
        d = min(abs((a-0+180)%360-180), abs((a-360+180)%360-180))
        return clamp(1 - d/180.0, 0, 1)
    aspect = north_pref(aspect_deg)
    compat = 0.7 if ("Castanea" in forest or "Quercus" in forest) else \
             0.8 if ("Fagus" in forest) else \
             0.6 if ("Pinus" in forest or "Abies" in forest or "Picea" in forest) else 0.5
    fut3 = sum(rain_next10[:3])
    future = clamp(fut3/25.0, 0, 1)

    s = (0.34*water + 0.26*thermo + 0.18*altfit + 0.10*aspect + 0.12*compat) * 100.0
    return clamp(s,0,100), {
        "water": round(water,3), "thermo": round(thermo,3),
        "alt": round(altfit,3), "aspect": round(aspect,3),
        "compat": round(compat,3), "future": round(future,3)
    }

def caps_estimate(score: float, compat: float, uncert: float) -> Tuple[int, Tuple[int,int]]:
    if score<20: base=(0,1)
    elif score<40: base=(1,3)
    elif score<60: base=(3,8)
    elif score<80: base=(8,18)
    else: base=(18,35)
    m = 0.6 + 0.6*compat - 0.3*uncert
    lo = max(0, int(round(base[0]*m)))
    hi = max(lo+1, int(round(base[1]*m)))
    mean = int(round(0.6*lo+0.4*hi))
    return mean, (lo,hi)

# ----------------------- endpoints -----------------------
@APP.get("/api/geocode")
async def api_geocode(q: str): return await geocode(q)

@APP.get("/api/score")
async def api_score(
    lat: float = Query(...), lon: float = Query(...), day: int = Query(0, ge=0, le=9)
):
    # 1) meteo daily (storico+previsioni)
    md = await open_meteo_daily(lat, lon)
    D = md.get("daily", {})
    precip = [float(x or 0.0) for x in D.get("precipitation_sum",[])]
    tmean  = [float(x) for x in D.get("temperature_2m_mean",[])]
    tmin   = [float(x) for x in D.get("temperature_2m_min",[])]
    tmax   = [float(x) for x in D.get("temperature_2m_max",[])]
    if len(precip) < 20: raise httpx.HTTPError("Serie meteo insufficiente")

    # split: 14 passati + 10 futuri
    past14 = precip[:14][::-1]   # ieri..14 giorni fa (per days_since)
    p14_mm = sum(precip[:14])    # ultimi 14 gg totali
    fut10  = precip[14:14+10]    # oggi..D+9

    # tmean7 centrata sul giorno (ultimi 7 fino al giorno selezionato)
    # costruiamo lista unendo passati e futuri
    t_all = tmean[:14] + tmean[14:14+10]
    i0 = 14 + day  # index assoluto (oggi=14)
    t7 = sum(t_all[max(0, i0-6):i0+1]) / max(1, (i0 - max(0, i0-6) + 1))

    # ET0 7gg stimata con Hargreaves
    et0_7 = 0.0
    for k in range(max(0, i0-6), i0+1):
        et0_7 += hargreaves_et0(tmin[k], tmax[k], t_all[k], lat)

    # 2) nowcast prossime ore (se disponibile)
    hourly_rain = await openweather_hourly(lat, lon)
    next_rain_hour = None
    if hourly_rain:
        for h, mm in enumerate(hourly_rain):
            if mm >= 0.2:
                next_rain_hour = h
                break

    # 3) bosco + quota + aspetto reali
    try:
        kind = await overpass_forest(lat, lon)
    except Exception:
        kind = None
    # quota + griglia per slope/aspect
    try:
        grid = await elev_grid_3x3(lat, lon, 30.0)
        elev_m = float(grid[1][1])
        slope_deg, aspect_deg = slope_aspect_from_grid(grid, lat, 30.0)
    except Exception:
        elev_m = 800.0
        slope_deg, aspect_deg = 10.0, 0.0
    forest = forest_label_from_osm_kind(kind, elev_m)

    # 4) score
    score, br = composite_score(p14_mm, t7, et0_7, elev_m, aspect_deg, forest, lat, fut10[day:])
    uncert = 0.12
    if p14_mm < 6: uncert += 0.10
    if kind is None: uncert += 0.05
    uncert = clamp(uncert, 0.05, 0.35)

    mean_caps, rng = caps_estimate(score, br["compat"], uncert)
    species = porcini_species(lat, elev_m, forest, t7)

    # 5) diagnostica piogge
    last_rain_d = days_since_last_rain(past14)
    nr = next_rain_from_daily(fut10[1:])
    next_rain_text = None
    if next_rain_hour is not None:
        next_rain_text = f"Pioggia nelle prossime ore (~{next_rain_hour}h)."
    elif nr:
        d, mm = nr
        next_rain_text = f"Prossima pioggia in D+{d+1} (~{mm:.0f} mm)."

    # 6) consigli dinamici
    tips = []
    if br["water"]<0.45:
        tips.append("cerca suoli profondi/ombrosi, impluvi e sottobosco con lettiera spessa")
    if br["thermo"]>0.6:
        tips.append("preferisci esposizioni N–NE–NW e ore più fresche")
    if br["alt"]<0.6:
        tips.append("spostati di quota verso fasce più " + ("alte" if elev_m < seasonal_alt_window(lat)[0] else "basse"))
    if "Castanea" in forest:
        tips.append("castagneti maturi ottimi dopo piogge >20–30 mm; controlla margini e vecchie ceppaie")
    if "Fagus" in forest:
        tips.append("in faggeta cerca zone umide e margini con radure")
    if "Pinus" in forest or "Abies" in forest or "Picea" in forest:
        tips.append("in conifere preferisci radure, margini e zone con muschi umidi")
    if br["future"]>=0.5:
        tips.append("piogge imminenti: probabile finestra 7–12 giorni dopo")
    if slope_deg>=5 and slope_deg<=20:
        tips.append("pendenze 5–20° spesso produttive; esplora traversi e dossi ombreggiati")

    # 7) perché (bullet sintetici)
    why = []
    if br["water"]<0.35:  why.append("suolo secco (P14 basso vs ET₀)")
    if br["thermo"]<0.35: why.append("T media poco favorevole al micelio")
    if br["alt"]<0.4:    why.append("quota lontana dall’optimum stagionale/latitudinale")
    if br["aspect"]<0.4: why.append("esposizione soleggiata (S–SE–SW) durante stagioni calde")
    if not why: why.append("condizioni coerenti per sviluppo di carpofori")

    return {
        "lat": lat, "lon": lon, "alt_m": round(elev_m,1),
        "forest": forest, "slope_deg": round(slope_deg,1),
        "aspect_deg": round(aspect_deg,1), "aspect_octant": aspect_octant(aspect_deg),
        "score": int(round(score)), "breakdown": br,
        "caps_3h": {"mean": mean_caps, "range": rng, "uncertainty": round(uncert,2)},
        "species": species,
        "p14_mm": round(p14_mm,1), "tmean7_c": round(t7,1), "et0_7mm": round(et0_7,1),
        "rain_next10_mm": fut10, "last_rain_days": last_rain_d,
        "next_rain": next_rain_text,
        "advice": tips, "why": why
    }

@APP.get("/api/heatmap")
async def api_heatmap(day: int = Query(0, ge=0, le=9)):
    bbox = (36.5, 6.5, 47.2, 18.8)  # Italia
    step = 0.7
    lats = [bbox[0] + i*step for i in range(int((bbox[2]-bbox[0])/step)+1)]
    lons = [bbox[1] + j*step for j in range(int((bbox[3]-bbox[1])/step)+1)]
    pts=[]
    async def one(lat, lon):
        try:
            md = await open_meteo_daily(lat, lon)
            D = md.get("daily",{})
            pr = [float(x or 0.0) for x in D.get("precipitation_sum",[])]
            tm = [float(x) for x in D.get("temperature_2m_mean",[])]
            tn = [float(x) for x in D.get("temperature_2m_min",[])]
            tx = [float(x) for x in D.get("temperature_2m_max",[])]
            if len(pr)<20: return
            p14 = sum(pr[:14]); fut = pr[14:14+10]
            i0 = 14+day
            t7 = sum((tm[:14]+tm[14:14+10])[max(0,i0-6):i0+1]) / max(1,(i0-max(0,i0-6)+1))
            et = hargreaves_et0(tn[i0], tx[i0], t7, lat)
            elev=800.0; aspect=0.0
            s, _ = composite_score(p14, t7, et, elev, aspect, forest_label_from_osm_kind(None,elev), lat, fut[day:])
            pts.append({"lat": lat, "lon": lon, "s": int(round(s))})
        except Exception:
            pass
    await asyncio.gather(*[one(a,b) for a in lats for b in lons])
    return {"day": day, "points": pts}




