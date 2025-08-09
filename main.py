# main.py  (v0.96)
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
import math, os, asyncio, httpx

APP = FastAPI(title="TrovaPorcini API v0.96")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Utilità geografiche & DEM ----------
def _octant_from_deg(deg: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    idx = int((deg%360)/45.0 + 0.5) % 8
    return dirs[idx]

def _fit_plane_aspect_slope(z: List[List[float]], cell_m: float=30.0) -> Tuple[float,float]:
    """
    Fit z = ax + by + c su griglia 5x5 centrata -> slope (°), aspect (° da N, CW).
    """
    n = len(z); m = len(z[0])
    cx, cy = (m-1)/2.0, (n-1)/2.0
    Sxx=Syy=Sxy=Sxz=Syz=0.0
    for r in range(n):
        for c in range(m):
            x = (c-cx)*cell_m
            y = (cy-r)*cell_m   # y cresce verso nord
            zz= z[r][c]
            Sxx += x*x
            Syy += y*y
            Sxy += x*y
            Sxz += x*zz
            Syz += y*zz
    det = (Sxx*Syy - Sxy*Sxy) or 1e-9
    a = (Syy*Sxz - Sxy*Syz)/det  # dz/dx
    b = (Sxx*Syz - Sxy*Sxz)/det  # dz/dy
    slope_rad = math.atan(math.hypot(a,b))
    slope_deg = slope_rad*180/math.pi
    # direzione di massima discesa (aspect): verso -gradiente
    aspect_rad = math.atan2(-a, -b)  # y positivo = nord
    aspect_deg = (aspect_rad*180/math.pi) % 360
    return slope_deg, aspect_deg

async def _elev_grid(lat: float, lon: float, step_m: float=30.0) -> List[List[float]]:
    # 5x5 grid around point using Open-Elevation API (batched)
    coords=[]
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    d = [-2,-1,0,1,2]
    for dy in d:
        for dx in d:
            coords.append({
                "latitude": lat + dy*step_m*deg_per_m_lat,
                "longitude": lon + dx*step_m*deg_per_m_lon
            })
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json={"locations": coords})
        r.raise_for_status()
        js = r.json()["results"]
    vals = [p["elevation"] for p in js]
    grid = [vals[i*5:(i+1)*5] for i in range(5)]
    return grid

# ---------- Meteo: Open-Meteo + OpenWeather ----------
async def _openmeteo(lat: float, lon: float) -> Dict[str,Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join([
            "precipitation_sum","temperature_2m_mean",
            "et0_fao_evapotranspiration"
        ]),
        "past_days": 14, "forecast_days": 10, "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def _openweather(lat: float, lon: float) -> Optional[Dict[str,Any]]:
    key = os.getenv("OPENWEATHER_API_KEY","")
    if not key: return None
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {"lat":lat, "lon":lon, "appid":key, "units":"metric", "exclude":"minutely,hourly,alerts"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

def _mm_from_openweather_daily(ow: Dict[str,Any]) -> List[float]:
    out=[]
    for d in ow.get("daily",[]):
        mm = float(d.get("rain",0.0))
        out.append(mm)
    return out

# ---------- Indice, finestra, specie ----------
def _species_prob(lat: float, alt_m: float, forest: str, month: int) -> List[str]:
    out=[]
    f = forest.lower()
    # molto semplificato ma coerente con letteratura italiana
    if "castanea" in f or "fagus" in f or "broad" in f:
        if month>=5 and month<=11:
            out.append("Boletus aereus" if (alt_m<1000 and month>=6) else "Boletus reticulatus")
            out.append("Boletus edulis")
    if "pinus" in f or "abies" in f or "picea" in f or "conifer" in f:
        out.append("Boletus pinophilus")
        if "abies" in f or "picea" in f:
            out.append("Boletus edulis")
    if not out:
        out.append("Boletus edulis")
    # rimuovi duplicati preservando ordine
    seen=set(); res=[]
    for s in out:
        if s not in seen: res.append(s); seen.add(s)
    return res[:3]

def _score_day(p14: float, et0_7: float, tmean7: float,
               alt_m: float, aspect_deg: float,
               forest: str, lat: float, month: int) -> Tuple[int, Dict[str,float]]:
    # idoneità idrica
    water = max(0.0, min(1.0, (p14 - 0.4*et0_7) / 20.0))  # 0..1
    # fascia termica ottimale (media 7 gg)
    t_opt_low, t_opt_high = (12.0, 20.0)
    if month<=5 or month>=10: t_opt_low, t_opt_high = (8.0, 18.0)
    if "aereus" in " ".join(_species_prob(lat,alt_m,forest,month)).lower():
        t_opt_high = 23.0
    if tmean7<=t_opt_low: therm = max(0.0, (tmean7 - (t_opt_low-6))/6.0)
    elif tmean7>=t_opt_high: therm = max(0.0, ((t_opt_high+6)-tmean7)/6.0)
    else: therm = 1.0
    # compat forest
    f = forest.lower()
    compat = 0.6
    if "castanea" in f or "fagus" in f: compat=1.0
    elif "quercus" in f: compat=0.8
    elif "pinus" in f or "abies" in f or "picea" in f: compat=0.9
    # aspect: preferisci N-NE-NW in secco; S a inizio stagione fredda
    oct = _octant_from_deg(aspect_deg)
    asp_bonus = 0.9 if oct in ["N","NE","NW"] else (1.0 if oct in ["E","W"] else 0.8)
    # quota/latitudine grezza
    lat_norm = max(0.0, min(1.0, (lat-36.5)/(46.8-36.5)))
    alt_opt = 800 + 600*(1.0-lat_norm)   # sud -> più alto
    alt_pen = max(0.0, 1.0 - abs(alt_m-alt_opt)/700.0)

    base = 100*water*0.45 + 100*therm*0.35 + 100*compat*0.15 + 100*asp_bonus*0.05
    base = base*alt_pen
    score = int(max(0,min(100, round(base))))
    breakdown = {
        "p14": p14, "et0_7": et0_7, "tmean7": tmean7,
        "water": water, "therm": therm, "compat": compat,
        "aspect": aspect_deg, "alt_pen": alt_pen
    }
    return score, breakdown

def _best_window(scores: List[int]) -> Tuple[int,int,int]:
    # media mobile su 3 giorni, scegli top window
    mus=[]
    for i in range(len(scores)-2):
        mus.append((i, sum(scores[i:i+3])//3))
    if not mus: return 0,0,0
    mus.sort(key=lambda x:x[1], reverse=True)
    i, m = mus[0]
    return i, i+2, m

def _practical_tips(day: Dict[str,Any], aspect_deg: float, forest: str) -> str:
    tips=[]
    s = day
    oct = _octant_from_deg(aspect_deg)
    # umidità/vento/sole (proxy: water vs therm)
    if s["score"]<40:
        tips.append("evita zone aride; cerca conche e impluvi con suolo profondo")
    if s["water"]>0.7: tips.append("prediligi margini di radure e faggi maturi dopo piogge recenti")
    if oct in ["N","NE","NW"]:
        tips.append("favorisci versanti ombreggiati ({}), microvalloni e tronchi a terra".format(oct))
    else:
        tips.append("cerca nelle ore fresche su versanti {} con copertura fogliare densa".format(oct))
    if "pinus" in forest.lower():
        tips.append("nel pino rosso controlla i margini muschiosi e i canaloni umidi")
    if "castanea" in forest.lower():
        tips.append("in castagneti maturi verifica i bordi sentieri e le scarpate fresche")
    return "; ".join(tips)

# ---------- Forest da OSM (semplificato) ----------
async def _forest_kind(lat: float, lon: float) -> str:
    # Overpass sfruttando natural=wood + leaf_type/wood tag
    q = f"""
    [out:json][timeout:25];
    (
      way(around:800,{lat},{lon})["natural"="wood"];
      relation(around:800,{lat},{lon})["natural"="wood"];
    );
    out center tags 20;
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post("https://overpass-api.de/api/interpreter", data={"data":q})
        r.raise_for_status()
        j=r.json()
    labels=[]
    for el in j.get("elements",[]):
        tags = el.get("tags",{})
        lt = tags.get("leaf_type","").lower()
        if "broad" in lt or lt=="broadleaved": labels.append("broadleaved")
        if "conif" in lt or lt=="needleleaved": labels.append("coniferous")
        if "wood" in tags:
            if tags["wood"] in ("deciduous","broadleaved"): labels.append("broadleaved")
            if tags["wood"] in ("coniferous","needleleaved"): labels.append("coniferous")
    if labels:
        return "broadleaved" if labels.count("broadleaved")>=labels.count("coniferous") else "coniferous"
    # fallback altitudine
    return "broadleaved"

def _forest_label(kind: str, alt_m: float) -> str:
    if kind=="coniferous":
        return "Pinus/Abies/Picea"
    # broadleaved con fallback altitudine
    if alt_m>900: return "Fagus sylvatica"
    if 500<alt_m<=900: return "Castanea sativa"
    return "Quercus spp."

# ---------- Endpoint ----------
@APP.get("/api/score")
async def api_score(lat: float = Query(...), lon: float = Query(...), day: int = Query(0)):
    """
    day = 0..9 (oggi..D+9). Restituisce:
    score, tips, perche, breakdown, specie, best_window, stima_kg3h, dati tecnici.
    """
    # DEM & aspect
    grid = await _elev_grid(lat, lon, step_m=30.0)
    slope_deg, aspect_deg = _fit_plane_aspect_slope(grid, cell_m=30.0)
    elev_m = float(grid[2][2])

    # forest
    kind = await _forest_kind(lat, lon)
    forest = _forest_label(kind, elev_m)

    # Meteo (Open-Meteo + OpenWeather opzionale)
    om = await _openmeteo(lat, lon)
    daily = om["daily"]
    precip = daily["precipitation_sum"]          # len = 25 (14 passati + 10 futuri + oggi)
    tmean  = daily["temperature_2m_mean"]
    et0    = daily.get("et0_fao_evapotranspiration", [0.0]*len(precip))

    # costruisci serie passato/futuro
    past14 = precip[:len(precip)-10]
    fut10  = precip[len(precip)-10:]
    pastT  = tmean[:len(tmean)-10]
    futT   = tmean[len(tmean)-10:]
    pastE  = et0[:len(et0)-10]

    P14 = sum(past14[-14:])
    ET07 = sum(pastE[-7:]) if pastE else 0.0
    last_rain_days = 0
    for i in range(1,15):
        if past14[-i] >= 0.5: last_rain_days = i-1; break

    # prossima pioggia
    next_rain_mm, next_rain_in_days = 0.0, None
    for i,mm in enumerate(fut10):
        if mm>=2.0:
            next_rain_in_days = i
            next_rain_mm = sum(fut10[i:i+3])
            break

    # calcola score 0..9
    scores=[]; infos=[]
    now = datetime.now(timezone.utc)
    month = int(now.astimezone().month)
    for d in range(10):
        t7 = sum((pastT+futT)[-7-d:len(pastT)+d]) / 7.0
        p14 = sum((past14+fut10)[-14-d:len(past14)+d])
        et7 = sum((pastE+[0]*10)[-7-d:len(pastE)+d])
        sc, br = _score_day(p14, et7, t7, elev_m, aspect_deg, forest, lat, (month + (d//30)) )
        scores.append(sc)
        infos.append({**br, "score": sc})

    s_today, breakdown_today = scores[0], infos[0]
    s_sel, info_sel = scores[day], infos[day]
    w_start, w_end, w_mean = _best_window(scores)

    # consigli/prché (dipendono dal giorno)
    tips = _practical_tips({**info_sel, "score":s_sel}, aspect_deg, forest)

    specie = _species_prob(lat, elev_m, forest, month+day)
    # stima kg 3h: funzione del punteggio, compat forest e incertezza semplice
    compat = info_sel["compat"]
    kg3h = round( max(0.0, (s_sel/100.0) * (0.7+0.6*compat)) , 1)

    # nota finestra da piogge future
    window_note = None
    if next_rain_in_days is not None and next_rain_mm>=10 and (12<=info_sel["tmean7"]<=22):
        window_note = f"Possibile finestra ~{next_rain_in_days+4}-{next_rain_in_days+8} giorni dopo pioggia prevista (~{int(next_rain_mm)} mm)."

    return {
        "elev_m": elev_m,
        "slope_deg": round(slope_deg,1),
        "aspect_deg": round(aspect_deg,1),
        "aspect_octant": _octant_from_deg(aspect_deg),
        "forest": forest,
        "P14_mm": round(P14,1),
        "ET0_7g_mm": round(ET07,1),
        "last_rain_days": last_rain_days,
        "next_rain_mm": round(next_rain_mm,1),
        "next_rain_in_days": next_rain_in_days,
        "scores": scores,
        "selected_day": day,
        "score_today": s_today,
        "score_selected": s_sel,
        "best_window": {"start": w_start, "end": w_end, "mean": w_mean},
        "tips": tips,
        "species_probable": specie,
        "kg3h_estimate": kg3h,
        "window_note": window_note,
        "breakdown": info_sel
    }




