from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx, math, asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

# ======= UTIL della versione funzionante (riusati) =======
# NB: lasciamo i nomi e le logiche come nella versione stabile
def k_from_half_life(days: float)->float:
    return 0.5**(1.0/days)

def api_decay(precip: List[float], half_life_days: float=8.0)->float:
    k = k_from_half_life(half_life_days)
    api = 0.0
    for p in precip:
        api = k*api + (p or 0.0)
    return api

def temperature_suitability(tmin: float, tmax: float, tmean: float) -> float:
    if tmin < 1 or tmax > 32: return 0.0
    if tmean <= 6: base = max(0.0, (tmean-0)/6.0*0.3)
    elif tmean <= 10: base = 0.3 + 0.2*((tmean-6)/4.0)
    elif tmean <= 18: base = 0.5 + 0.5*((tmean-10)/8.0)
    elif tmean <= 22: base = 0.8 - 0.2*((tmean-18)/4.0)
    else: base = max(0.0, 0.6 - 0.6*((tmean-22)/10.0))
    if tmin < 6: base *= max(0.3, 1 - (6 - tmin)/8.0)
    if tmax > 24: base *= max(0.3, 1 - (tmax - 24)/10.0)
    return max(0.0, min(1.0, base))

def final_score(api_val: float, et0_7: float, t_suit: float) -> int:
    moisture = max(0.0, min(1.0, api_val - 0.6*et0_7)/40.0 + 0.6)  # normalizzazione dolce
    base = 0.6*moisture + 0.4*t_suit
    return int(round(max(0.0, min(100.0, 100.0*base))))

def best_window_3day(scores: List[int]) -> Tuple[int,int,int]:
    if not scores: return (0,0,0)
    best=(0,0,0); m=-1
    for i in range(0, len(scores)-2):
        s = round((scores[i]+scores[i+1]+scores[i+2])/3)
        if s>m: m=s; best=(i,i+2,m)
    return best

def deg_to_octant(deg: float)->str:
    octs = ["N","NE","E","SE","S","SW","W","NW","N"]
    i = int(((deg%360)+22.5)//45)
    return octs[i]

def slope_aspect_from_elev_grid(grid: List[List[float]], cell_size_m: float=30.0) -> Tuple[float,float,str]:
    # stima semplice da griglia 3x3
    z = grid
    dzdx = ((z[0][2]+2*z[1][2]+z[2][2]) - (z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_size_m)
    dzdy = ((z[2][0]+2*z[2][1]+z[2][2]) - (z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_size_m)
    slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect = (math.degrees(math.atan2(dzdx, dzdy))+360)%360
    return (round(slope,1), round(aspect,1), deg_to_octant(aspect))

# ======= APP =======
app = FastAPI(title="Trovapolcini API (stabile+extra)", version="0.7.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

HEADERS = {"User-Agent": "Trovapolcini/0.7 (no-key)"}

# ========= PROVIDERS (Open-Meteo + Nominatim + Open-Elevation) =========
async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "q": q, "addressdetails": 1, "limit": 1}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.get(url, params=params); r.raise_for_status()
        j = r.json()
    if not j: raise HTTPException(404, detail="Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0].get("display_name","")}

async def open_meteo_daily(lat: float, lon: float, past_days:int=30, forecast_days:int=10) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "daily": ",".join([
            "precipitation_sum","temperature_2m_mean",
            "temperature_2m_min","temperature_2m_max",
            "et0_fao_evapotranspiration","relative_humidity_2m_mean"
        ]),
        "past_days": past_days, "forecast_days": forecast_days
    }
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
        r = await c.get(base, params=params); r.raise_for_status()
        return r.json()

async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    # griglia 3x3
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m*deg_per_m_lat; dlon = step_m*deg_per_m_lon
    coords = [{"latitude": lat + dr*dlat, "longitude": lon + dc*dlon}
              for dr in (-1,0,1) for dc in (-1,0,1)]
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": coords})
        r.raise_for_status(); j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]

# ========= API =========
@app.get("/api/health")
async def health():
    return {"ok": True, "app": "Trovapolcini", "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode(q)

@app.get("/api/score")
async def api_score(lat: float, lon: float, half: float = 8.0):
    # richieste in parallelo (come versione funzionante)
    tasks = [
        asyncio.create_task(open_meteo_daily(lat, lon, past_days=30, forecast_days=10)),
        asyncio.create_task(open_elevation_grid(lat, lon)),
    ]
    meteo_d, elev_grid = await asyncio.gather(*tasks)

    elev_m = float(elev_grid[1][1])
    slope_deg, aspect_deg, aspect_oct = slope_aspect_from_elev_grid(elev_grid, cell_size_m=30.0)

    daily = meteo_d["daily"]
    timev = daily["time"]
    precip = daily["precipitation_sum"]
    tmean = daily["temperature_2m_mean"]
    tmin = daily["temperature_2m_min"]
    tmax = daily["temperature_2m_max"]
    et0  = daily.get("et0_fao_evapotranspiration", [0.0]*len(precip))
    rh   = daily.get("relative_humidity_2m_mean", [None]*len(precip))

    pastN = 30
    pastP = precip[:pastN]
    pastTmin = tmin[:pastN]; pastTmax = tmax[:pastN]; pastTmean = tmean[:pastN]; pastET0 = et0[:pastN]

    # indicatori come prima
    P7 = sum(pastP[-7:]); P14 = sum(pastP[-14:])
    API_val = api_decay(pastP, half_life_days=half)
    ET0_7 = sum(pastET0[-7:]) if pastET0 else 0.0
    Tmean7 = sum(pastTmean[-7:])/max(1,len(pastTmean[-7:]))
    Tmin7 = min(pastTmin[-7:]) if pastTmin else 0.0
    Tmax7 = max(pastTmax[-7:]) if pastTmax else 0.0

    # indice oggi (come base funzionante)
    T_suit_today = temperature_suitability(pastTmin[-1], pastTmax[-1], pastTmean[-1])
    today_score = final_score(API_val, ET0_7, T_suit_today)

    # previsione 10 giorni (come base funzionante)
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
        et7 = max(0.0, sum((futET0[max(0,i-6):i+1])) )
        sc = final_score(rolling_api, et7, t_s)
        scores.append(int(round(sc)))

    s,e,m = best_window_3day(scores)

    # piogge passate/future (richiesta nuova)
    rain_past: Dict[str,float] = {}
    for d, mm in zip(timev[:pastN], pastP):
        rain_past[str(d)] = round(float(mm or 0.0), 1)
    rain_future: Dict[str,float] = {}
    for d, mm in zip(timev[pastN:pastN+10], futP):
        rain_future[str(d)] = round(float(mm or 0.0), 1)

    # stima raccolto 2h (basata su today_score + regime pioggia)
    def harvest_from_score(sc:int, p14:float)->str:
        if sc>=75 and p14>=20: return "6–10+ porcini (spot buoni)"
        if sc>=55 and p14>=10: return "2–5 porcini"
        if sc>=45 and p14>=8:  return "1–3 porcini"
        return "0–1 porcini"
    harvest = harvest_from_score(today_score, P14)

    # messaggi brevi (come base + estensioni)
    reasons = []
    if P14 < 20: reasons.append(f"Piogge scarse negli ultimi 14 giorni ({P14:.0f} mm).")
    if ET0_7 > 30: reasons.append(f"Evapotraspirazione elevata 7g ({ET0_7:.0f} mm) → disseccamento.")
    if Tmean7 < 10: reasons.append(f"Temperature medie basse (Tmed7 {Tmean7:.1f} °C).")
    if Tmean7 > 18: reasons.append(f"Temperature medie alte (Tmed7 {Tmean7:.1f} °C).")
    if not reasons: reasons.append("Piogge utili e termica favorevole; verifica spot noti.")
    tips = []
    if today_score>=70: tips.append("Finestra favorevole: controlla faggete/castagneti a 900–1500 m, versanti N–NE.")
    elif today_score>=55: tips.append("Condizioni discrete: cerca in conche umide, margini di bosco e lettiera profonda.")
    else: tips.append("Bassa probabilità: attendi pioggia >10–15 mm e 5–10 giorni di maturazione.")

    return {
        "elevation_m": round(elev_m),
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct,
        "API_star_mm": round(API_val,1),
        "P7_mm": round(P7,1),
        "P14_mm": round(P14,1),
        "ET0_7d_mm": round(ET0_7,1),
        "Tmean7_c": round(Tmean7,1),
        "Tmin7_c": round(Tmin7,1),
        "Tmax7_c": round(Tmax7,1),
        "RH7_mean_%": round(sum([v for v in rh[-7:] if v is not None])/max(1,len([v for v in rh[-7:] if v is not None]))) if any(rh[-7:]) else None,

        "score_today": int(round(today_score)),
        "scores_next10": scores,
        "best_window": {"start": s, "end": e, "mean": m},

        # NUOVO: piogge passate/future con date
        "rain_past": rain_past,
        "rain_future": rain_future,

        # NUOVO: stima raccolto 2h
        "harvest_estimate": harvest,

        "explanation": {"reasons": reasons},
        "tips": tips
    }







