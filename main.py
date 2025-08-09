from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import httpx, math, statistics

app = FastAPI(title="TrovaPorcini 4.0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- util DEM ----------
def slope_aspect_from_elev_grid(grid: List[List[float]], cell_m: float = 30.0) -> Tuple[float, float]:
    # grid 3x3 -> pendenza (deg) & aspect (deg, 0=N)
    # Horn method semplificato
    z = grid
    if len(z) != 3 or any(len(r) != 3 for r in z):
        return 0.0, 0.0
    dzdx = ((z[0][2]+2*z[1][2]+z[2][2]) - (z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_m)
    dzdy = ((z[2][0]+2*z[2][1]+z[2][2]) - (z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_m)
    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)
    aspect_rad = math.atan2(dzdx, -dzdy)
    aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
    return slope_deg, aspect_deg

def deg_to_octant(deg: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    i = int((deg/22.5)+0.5) % 16
    return dirs[i]

# ---------- external calls ----------
HEADERS = {"User-Agent": "TrovaPorcini/4.0 (+https://example.org)"}

async def open_meteo_daily(lat: float, lon: float) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join([
            "precipitation_sum",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration"
        ]),
        "timezone": "auto",
        "past_days": 14, "forecast_days": 10,
    }
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    # 3x3 intorno al punto
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m * deg_per_m_lat
    dlon = step_m * deg_per_m_lon
    coords = []
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            coords.append({"latitude": lat + dr*dlat, "longitude": lon + dc*dlon})
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        r = await client.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": coords})
        r.raise_for_status()
        j = r.json()["results"]
    els = [p["elevation"] for p in j]
    return [els[0:3], els[3:6], els[6:9]]

async def overpass_kind(lat: float, lon: float, radius_m: int = 800) -> Optional[str]:
    # prova a dedurre broadleaved/coniferous
    q = f"""
[out:json][timeout:25];
(
  way(around:{radius_m},{lat},{lon})[natural=wood];
  relation(around:{radius_m},{lat},{lon})[natural=wood];
);
out tags;
"""
    try:
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
            r = await client.post("https://overpass-api.de/api/interpreter", data=q)
            r.raise_for_status()
            j = r.json()
    except Exception:
        return None
    labels=[]
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        lt = (tags.get("leaf_type") or "").lower()
        if lt in ("broadleaved","broadleaf") or "deciduous" in tags.get("wood",""):
            labels.append("broadleaved")
        elif lt in ("coniferous","needleleaved") or "coniferous" in tags.get("wood",""):
            labels.append("coniferous")
    if not labels: 
        return None
    return "broadleaved" if labels.count("broadleaved") >= labels.count("coniferous") else "coniferous"

def forest_label(kind: Optional[str], alt_m: float) -> str:
    if kind == "coniferous": return "Pinus/Abies/Picea"
    if kind == "broadleaved":
        if alt_m > 800: return "Fagus sylvatica"
        if alt_m > 500: return "Castanea sativa"
        return "Quercus spp."
    # fallback altitudinale
    if alt_m > 1400: return "Pinus/Abies/Picea"
    if alt_m > 900: return "Fagus sylvatica"
    if alt_m > 500: return "Castanea sativa"
    return "Quercus spp."

# ---------- scoring ----------
def rolling_P14(past: List[float]) -> float:
    return float(sum(past[-14:]))

def last_rain_days(past: List[float]) -> int:
    for i, v in enumerate(reversed(past)):
        if v >= 1.0:
            return i
    return len(past)

def next_rain_in_days(future: List[float], thr: float = 3.0) -> Optional[int]:
    for i, v in enumerate(future, start=1):
        if v >= thr:
            return i
    return None

def alt_opt_by_lat(lat: float) -> float:
    # opt ~900m al 45°N; meno al sud, più al nord
    return max(300.0, min(1400.0, 900.0 + (lat-45.0)*(-25.0)))

def aspect_suitability(aspect_deg: float, hot: bool) -> float:
    # scala 0..1: N-NE-NW meglio in estate secco, S peggio in caldo
    if aspect_deg is None: return 0.5
    preferred = [337.5, 0, 22.5, 315] if hot else [0, 360]  # N nei periodi caldi
    d = min(abs((aspect_deg - 0 + 540) % 360 - 180),
            abs((aspect_deg - 337.5 + 540) % 360 - 180),
            abs((aspect_deg - 22.5 + 540) % 360 - 180))
    base = max(0.0, 1.0 - d/180.0)
    if hot:
        # penalizza S forte
        dS = abs((aspect_deg - 180 + 540) % 360 - 180)
        base -= max(0.0, 1.0 - dS/60.0)*0.3
    return max(0.0, min(1.0, base))

def moisture_index(p14: float, et0_7: float) -> float:
    # bilancio idrico grezzo normalizzato 0..1
    # più p14, meno et0
    # p14 ~20-60 mm ok; et0_7 ~14-35 mm penalizza
    m = (p14 - 20.0)/(60.0) - (et0_7 - 14.0)/(35.0)
    return max(0.0, min(1.0, 0.5 + m))

def temp_window_score(tmean7: float, lat: float, month: int) -> float:
    # optimum ~15-18°C estate, 10-14°C autunno
    if month in (6,7,8): lo, hi = 14.0, 19.0
    else: lo, hi = 9.0, 14.5
    if tmean7 <= lo-3 or tmean7 >= hi+3: return 0.0
    if tmean7 < lo: return (tmean7-(lo-3))/3.0
    if tmean7 > hi: return (hi+3 - tmean7)/3.0
    return 1.0

def humidity_boost(rh: float) -> float:
    if rh is None: return 0.0
    # sopra 75% bonus, sotto 55% malus
    return max(-0.2, min(0.25, (rh-65.0)/40.0))

def radiation_penalty(sw_MJ: float) -> float:
    # radiazione eccessiva disidrata
    if sw_MJ is None: return 0.0
    return -max(0.0, (sw_MJ-18.0)/30.0)  # -0..-0.4

def altitude_score(alt: float, lat: float) -> float:
    opt = alt_opt_by_lat(lat)
    d = abs(alt - opt)
    if d >= 800: return 0.0
    if d <= 150: return 1.0
    return max(0.0, 1.0 - (d-150)/650.0)

def composite_score(p14: float, et0_7: float, t7: float, alt: float,
                    aspect_deg: float, rh: float, sw_MJ: float,
                    lat: float, month: int) -> Tuple[float, Dict[str,float]]:
    hot = (t7 >= 18.0)
    s_moist = moisture_index(p14, et0_7)          # 35%
    s_temp  = temp_window_score(t7, lat, month)    # 25%
    s_alt   = altitude_score(alt, lat)             # 15%
    s_asp   = aspect_suitability(aspect_deg, hot)  # 10%
    s_rh    = 0.5 + humidity_boost(rh)             # 0.3..0.75 -> normalize
    s_rad   = 0.5 + radiation_penalty(sw_MJ)       # 0.1..0.5
    # normalizza extra a 0..1
    s_rh = max(0.0, min(1.0, s_rh))
    s_rad = max(0.0, min(1.0, s_rad))
    score = (
        0.35*s_moist + 0.25*s_temp + 0.15*s_alt +
        0.10*s_asp + 0.10*s_rh + 0.05*s_rad
    )*100.0
    breakdown = {
        "moist": s_moist, "temp": s_temp, "alt": s_alt,
        "asp": s_asp, "rh": s_rh, "rad": s_rad
    }
    return max(0.0, min(100.0, score)), breakdown

def caps3h_estimate(score: float, forest: str, uncert: float) -> Tuple[float,float]:
    # stima cappelli in 3h (intervallo): aumenta con score, riduci con incertezza
    base = max(0.0, (score-35.0)/6.0)   # 0..~10
    if "Castanea" in forest or "Fagus" in forest:
        base *= 1.15
    lo = max(0.0, base*(1.0-uncert))
    hi = base*(1.0+uncert)
    return lo, hi

def advice_from_state(day:int, s: Dict[str,float], last_rain:int, next_rain:Optional[int], forest:str, aspect:str) -> str:
    msgs=[]
    if s["moist"]<0.45: msgs.append("suolo secco: cerca impluvi e lettiere umide")
    if s["temp"]<0.45: msgs.append("temperature fuori finestra: preferisci quote/versanti più freschi")
    if "Castanea" in forest: msgs.append("bene castagneti maturi")
    if last_rain<=2 and s["moist"]>=0.5: msgs.append("pioggia recente favorevole")
    if next_rain and next_rain<=3: msgs.append(f"possibile apertura finestra tra ~{next_rain} gg")
    if aspect in ("N","NE","NW"): msgs.append("puntare versanti ombreggiati (N-NE-NW)")
    return "; ".join(msgs) or "vai presto, cerca lettiere umide e zone a microclima fresco"

def why_from_state(p14,t7,alt,et0_7,last_rain,next_rain,aspect,forest,rh,sw,lat,month)->str:
    parts = [
        f"P14={p14:.1f} mm",
        f"T7={t7:.1f} °C",
        f"quota={int(alt)} m",
        f"ET0_7={et0_7:.1f} mm",
        f"ult. pioggia ~{last_rain} gg fa",
        f"pross. 10 gg={sw:.1f} MJ·m⁻² rad" if sw is not None else None,
        f"RH={int(rh)}%" if rh is not None else None,
        f"asp={aspect}",
        f"bosco: {forest}"
    ]
    if next_rain: parts.append(f"pross. pioggia tra ~{next_rain} gg")
    return "; ".join([p for p in parts if p])

# ---------- API ----------
@app.get("/api/score")
async def api_score(lat: float = Query(...), lon: float = Query(...)):
    met = await open_meteo_daily(lat, lon)
    daily = met["daily"]
    precip = daily["precipitation_sum"]
    tmean = daily["temperature_2m_mean"]
    rh = daily.get("relative_humidity_2m_mean")
    sw = daily.get("shortwave_radiation_sum")
    et0 = daily.get("et0_fao_evapotranspiration")

    past14 = precip[:14]
    fut10 = precip[14:24]
    t_all = tmean[:24]
    rh_all = rh[:24] if rh else [None]*24
    sw_all = sw[:24] if sw else [None]*24
    et0_all = et0[:24] if et0 else [0.0]*24

    grid = await open_elevation_grid(lat, lon)
    elev_m = float(grid[1][1])
    slope_deg, aspect_deg = slope_aspect_from_elev_grid(grid)
    aspect_oct = deg_to_octant(aspect_deg)

    kind = await overpass_kind(lat, lon)
    forest = forest_label(kind, elev_m)

    # indicatori
    P14 = rolling_P14(past14)
    last_r = last_rain_days(past14)
    next_r = next_rain_in_days(fut10)
    month = datetime.now(timezone.utc).astimezone().month

    # precompute 10 giorni
    scores=[]; adv=[]; whys=[]; breakdowns=[]
    for d in range(10):
        # finestra mobile: 7 giorni centrati su d (naive)
        t7 = statistics.mean(t_all[7+d-3:7+d+4]) if 7+d+4 <= len(t_all) else t_all[14+d]
        rhd = rh_all[14+d] if rh_all and (14+d)<len(rh_all) else None
        swd = sw_all[14+d] if sw_all and (14+d)<len(sw_all) else None
        et07 = sum(et0_all[14+d-6:14+d+1]) if et0_all and (14+d+1)<=len(et0_all) else 0.0

        # P14 “efficace”: decadimento emivita 7 gg + quota parte futura fino al giorno d
        p_hist = list(past14)
        for i in range(min(d, len(fut10))):
            p_hist.append(fut10[i])
        # peso esponenziale sullo storico
        half=7.0; lam=math.log(2)/half
        eff=sum(p_hist[-14+i]*math.exp(-lam*(len(p_hist)-14-i)) for i in range(14))
        score, br = composite_score(eff, et07, t7, elev_m, aspect_deg, rhd, swd, lat, month)
        scores.append(int(round(score)))
        breakdowns.append(br)
        adv.append(advice_from_state(d, br, last_r, next_r, forest, aspect_oct))
        whys.append(why_from_state(eff, t7, elev_m, et07, last_r, next_r, aspect_oct, forest, rhd, swd, lat, month))

    # finestra migliore 3 gg
    best_mean = -1; s=0; e=2
    for i in range(0, 8):
        m = statistics.mean(scores[i:i+3])
        if m>best_mean:
            best_mean=m; s=i; e=i+2

    # incertezza semplice (varianza)
    uncert = min(0.5, statistics.pstdev(scores)/100.0 if len(scores)>1 else 0.2)
    lo, hi = caps3h_estimate(scores[0], forest, uncert)

    return {
        "lat": lat, "lon": lon,
        "elev_m": elev_m, "slope_deg": round(slope_deg,1),
        "aspect_deg": round(aspect_deg,1), "aspect_octant": aspect_oct,
        "forest": forest,
        "P14_mm": round(P14,1),
        "Tmean7_c": round(statistics.mean(t_all[11:18]),1) if len(t_all)>=18 else round(t_all[14],1),
        "humidity_mean": rh_all[14] if rh else None,
        "radiation_MJm2": sw_all[14] if sw else None,
        "score_today": scores[0],
        "scores_next10": scores,
        "best_window": {"start": s, "end": e, "mean": int(round(best_mean))},
        "caps3h_est": {"low": round(lo,1), "high": round(hi,1)},
        "why_today": whys[0], "why_all": whys,
        "advice_today": adv[0], "advice_all": adv,
        "tech": {
            "last_rain_days": last_r,
            "next_rain_in_days": next_r,
            "sum_next10mm": round(sum(fut10),1),
        }
    }

# geocode semplice (per la barra località)
@app.get("/api/geocode")
async def api_geocode(q: str):
    # usa Nominatim quando non sono coordinate
    async with httpx.AsyncClient(timeout=15, headers={**HEADERS, "Accept-Language":"it"}) as client:
        r = await client.get("https://nominatim.openstreetmap.org/search", params={
            "format":"json","q":q,"addressdetails":1,"limit":1
        })
        r.raise_for_status()
        j = r.json()
        if not j: 
            return {"error":"not_found"}
        return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0]["display_name"]}




