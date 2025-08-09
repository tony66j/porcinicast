from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, math, asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

from utils import slope_aspect_from_elev_grid, best_window_3day, deg_to_octant

APP_NAME = "PorciniCast/0.5 (+https://netlify.app)"
HEADERS = {"User-Agent": APP_NAME}

OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")  # optional

app = FastAPI(title="PorciniCast API (v0.5)", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Providers
# ------------------------
async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "q": q, "addressdetails": 1, "limit": 1}
    async with httpx.AsyncClient(timeout=15, headers={**HEADERS, "Accept-Language": "it"}) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    if not j:
        raise httpx.HTTPError("Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0].get("display_name","")}

async def open_meteo(lat: float, lon: float, past_days:int=30, forecast_days:int=10) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join([
            "precipitation_sum","temperature_2m_mean",
            "temperature_2m_min","temperature_2m_max",
            "et0_fao_evapotranspiration"
        ]),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
        r = await c.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    # 3x3 around point
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m*deg_per_m_lat
    dlon = step_m*deg_per_m_lon
    coords = [{"latitude": lat + dr*dlat, "longitude": lon + dc*dlon}
              for dr in (-1,0,1) for dc in (-1,0,1)]
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": coords})
        r.raise_for_status()
        j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]

async def overpass_forest(lat: float, lon: float, radius_m:int=800) -> Optional[str]:
    q = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})[natural=wood];
      relation(around:{radius_m},{lat},{lon})[natural=wood];
      way(around:{radius_m},{lat},{lon})[landuse=forest];
      relation(around:{radius_m},{lat},{lon})[landuse=forest];
    );
    out tags;
    """
    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as c:
        r = await c.get("https://overpass-api.de/api/interpreter", params={"data": q})
        r.raise_for_status()
        j = r.json()
    labels = []
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        if "leaf_type" in tags:
            lt = tags["leaf_type"].lower()
            if "broad" in lt or lt == "broadleaved": labels.append("broadleaved")
            elif "conifer" in lt or lt == "coniferous": labels.append("coniferous")
        elif "wood" in tags:
            if tags["wood"] in ("deciduous","broadleaved"): labels.append("broadleaved")
            elif tags["wood"] in ("coniferous","needleleaved"): labels.append("coniferous")
    if labels:
        return "broadleaved" if labels.count("broadleaved")>=labels.count("coniferous") else "coniferous"
    return None

# OpenWeather (optional): current + 5d/3h forecast
async def owm_current(lat:float, lon:float)->Optional[Dict[str,Any]]:
    if not OWM_KEY: return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat":lat,"lon":lon,"units":"metric","lang":"it","appid":OWM_KEY}
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def owm_forecast_5d(lat:float, lon:float)->Optional[Dict[str,Any]]:
    if not OWM_KEY: return None
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat":lat,"lon":lon,"units":"metric","lang":"it","appid":OWM_KEY}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()

# ------------------------
# Scoring helpers (v0.5)
# ------------------------
def k_from_half_life(days: float)->float:
    # exponential decay so that after 'days' weight halves
    return 0.5**(1.0/days)

def api_decay(precip: List[float], half_life_days: float=8.0)->float:
    """Antecedent precipitation with ~1 week half-life."""
    k = k_from_half_life(half_life_days)
    api = 0.0
    for p in precip:
        api = k*api + (p or 0.0)
    return api

def temperature_suitability(tmin: float, tmax: float, tmean: float) -> float:
    # Optimal 10–18 °C; penalties outside; returns 0..1
    if tmin < 1 or tmax > 32: return 0.0
    if tmean <= 6: base = max(0.0, (tmean-0)/6.0*0.3)
    elif tmean <= 10: base = 0.3 + 0.2*((tmean-6)/4.0)
    elif tmean <= 18: base = 0.5 + 0.5*((tmean-10)/8.0)
    elif tmean <= 22: base = 0.8 - 0.2*((tmean-18)/4.0)
    else: base = max(0.0, 0.6 - 0.6*((tmean-22)/10.0))
    if tmin < 6: base *= max(0.3, 1 - (6 - tmin)/8.0)
    if tmax > 24: base *= max(0.3, 1 - (tmax - 24)/10.0)
    return max(0.0, min(1.0, base))

def aspect_modifier(aspect_deg: float) -> float:
    octants = {'N':1.08,'NE':1.06,'E':1.0,'SE':0.92,'S':0.85,'SW':0.85,'W':0.92,'NW':1.03}
    return octants.get(deg_to_octant(aspect_deg), 1.0)

def slope_modifier(slope_deg: float)->float:
    if slope_deg < 2: return 0.96
    if slope_deg < 12: return 1.05
    if slope_deg < 25: return 1.0
    return 0.9

def api_to_moisture(api: float)->float:
    # 0..1 with sweet spot ~20–70 mm
    if api <= 5: return 0.05*api
    if api <= 20: return 0.25 + 0.35*((api-5)/15.0)   # 0.25->0.60
    if api <= 70: return 0.60 + 0.35*((api-20)/50.0)  # 0.60->0.95
    if api <= 120: return 0.95 - 0.25*((api-70)/50.0) # 0.95->0.70
    return 0.65

def final_score(api_val: float, et0_7: float, t_suit: float, elev_m: float, aspect_deg: float, slope_deg: float) -> Tuple[float, Dict[str,float]]:
    # Simple water balance: reduce moisture by ET0 of last 7 days (scaled)
    moisture = max(0.0, api_to_moisture(max(0.0, api_val - 0.6*et0_7)))
    elev_mod = 1.05 if 700<=elev_m<=1400 else (0.6 if (elev_m<150 or elev_m>2200) else 0.95)
    asp_m = aspect_modifier(aspect_deg)
    slp_m = slope_modifier(slope_deg)
    base = 0.6*moisture + 0.4*t_suit
    score = max(0.0, min(100.0, 100.0*base*elev_mod*asp_m*slp_m))
    parts = {"moisture":moisture,"temp":t_suit,"elev_mod":elev_mod,"aspect_mod":asp_m,"slope_mod":slp_m}
    return score, parts

def day_advice(score:int, humid_now:Optional[int]=None, rain24:Optional[float]=None)->str:
    if score >= 70: return "Vai: molto favorevole."
    if score >= 60: return "Buono: probabile ritrovamento."
    if score >= 50:
        if humid_now is not None and humid_now < 45: return "Incerto: aria secca, meglio attendere nuove piogge."
        if rain24 is not None and rain24>5: return "Incerto oggi, meglio 1–2 giorni dopo la pioggia."
        return "Incerto: attendi 1–2 giorni o piogge."
    return "Sconsigliato oggi."

# ------------------------
# API endpoints
# ------------------------
@app.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode(q)

@app.get("/api/score")
async def api_score(lat: float, lon: float, half: float = 8.0):
    tasks = [
        asyncio.create_task(open_meteo(lat, lon, past_days=30, forecast_days=10)),
        asyncio.create_task(open_elevation_grid(lat, lon)),
        asyncio.create_task(overpass_forest(lat, lon)),
        asyncio.create_task(owm_current(lat, lon)),
        asyncio.create_task(owm_forecast_5d(lat, lon)),
    ]
    meteo, elev_grid, forest_kind, owm_now, owm_fc = await asyncio.gather(*tasks)

    elev_m = float(elev_grid[1][1])
    slope_deg, aspect_deg, aspect_oct = slope_aspect_from_elev_grid(elev_grid, cell_size_m=30.0)

    daily = meteo["daily"]
    precip = daily["precipitation_sum"]      # length 30 past + 10 future
    tmean = daily["temperature_2m_mean"]
    tmin = daily["temperature_2m_min"]
    tmax = daily["temperature_2m_max"]
    et0  = daily.get("et0_fao_evapotranspiration", [0.0]*len(precip))

    pastN = 30
    pastP = precip[:pastN]
    pastTmin = tmin[:pastN]
    pastTmax = tmax[:pastN]
    pastTmean = tmean[:pastN]
    pastET0 = et0[:pastN]

    API_val = api_decay(pastP, half_life_days=half)
    ET0_7 = sum(pastET0[-7:]) if pastET0 else 0.0
    T_suit_today = temperature_suitability(pastTmin[-1], pastTmax[-1], pastTmean[-1])

    today_score, parts = final_score(API_val, ET0_7, T_suit_today, elev_m, aspect_deg, slope_deg)

    # Forecast next 10 days
    futP = precip[pastN:pastN+10]
    futTmean = tmean[pastN:pastN+10]
    futTmin = tmin[pastN:pastN+10]
    futTmax = tmax[pastN:pastN+10]
    futET0 = et0[pastN:pastN+10]

    scores = []
    rolling_api = API_val
    k = k_from_half_life(half)
    for i in range(10):
        rolling_api = k*rolling_api + (futP[i] or 0.0)
        t_s = temperature_suitability(futTmin[i], futTmax[i], futTmean[i])
        et7 = max(0.0, sum((futET0[max(0,i-6):i+1])) )  # approx rolling 7d using forecast slice
        sc, _ = final_score(rolling_api, et7, t_s, elev_m, aspect_deg, slope_deg)
        scores.append(int(round(sc)))

    s,e,m = best_window_3day(scores)

    # OpenWeather extras
    humid_now = None
    rain24 = None
    if owm_now:
        try: humid_now = int(owm_now["main"]["humidity"])
        except: pass
    if owm_fc:
        try:
            lst = owm_fc.get("list",[])[:8]  # ~24h (8 * 3h)
            rain24 = sum([v.get("rain",{}).get("3h",0.0) for v in lst])
        except: pass

    advice = day_advice(int(round(today_score)), humid_now, rain24)

    # forest label (informative only)
    def forest_label_from_osm_kind(kind: Optional[str], alt_m: float) -> str:
        if kind == "coniferous": return "Conifere (Pinus/Abies/Picea)"
        if kind == "broadleaved":
            if alt_m > 900: return "Fagus sylvatica (stima)"
            if 500 < alt_m <= 900: return "Castanea/Quercus (stima)"
            return "Quercus spp. (stima)"
        if alt_m > 1400: return "Conifere (Pinus/Abies/Picea)"
        if alt_m > 900: return "Fagus sylvatica (stima)"
        if alt_m > 500: return "Castanea sativa (stima)"
        return "Quercus spp. (stima)"

    forest_label = forest_label_from_osm_kind(forest_kind, elev_m)

    uncertainty = 1.0 - 0.5*(parts["moisture"]) - 0.3*(parts["temp"])
    uncertainty = max(0.1, min(0.9, uncertainty))

    return {
        "elevation_m": elev_m,
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct,
        "forest": forest_label,
        "API_star_mm": round(API_val,1),
        "ET0_7d_mm": round(ET0_7,1),
        "Tmean7_c": round(sum(pastTmean[-7:])/max(1,len(pastTmean[-7:])),1),
        "humidity_now": humid_now,
        "rain24h_forecast_mm": round(rain24,1) if rain24 is not None else None,
        "score_today": int(round(today_score)),
        "scores_next10": scores,
        "best_window": {"start": s, "end": e, "mean": m},
        "breakdown": parts,
        "advice": advice,
        "uncertainty": round(uncertainty,2)
    }
