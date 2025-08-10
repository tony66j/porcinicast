from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, math, asyncio
from typing import Dict, List, Tuple
from datetime import datetime, timezone

app = FastAPI(title="Trova Porcini API", version="0.7.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------- Helper functions (stesse logiche v0.6) ----------------
def half_life_coeff(days: float) -> float:
    return 0.5 ** (1.0 / days)

def api_index(precip: List[float], half_life: float) -> float:
    k = half_life_coeff(half_life)
    api = 0.0
    for p in precip:
        api = k * api + (p or 0.0)
    return api

def temperature_fit(tmin: float, tmax: float, tmean: float) -> float:
    if tmin < 1 or tmax > 32: return 0.0
    if tmean <= 6: base = max(0.0, (tmean) / 6 * 0.3)
    elif tmean <= 10: base = 0.3 + 0.2 * ((tmean - 6) / 4)
    elif tmean <= 18: base = 0.5 + 0.5 * ((tmean - 10) / 8)
    elif tmean <= 22: base = 0.8 - 0.2 * ((tmean - 18) / 4)
    else: base = max(0.0, 0.6 - 0.6 * ((tmean - 22) / 10))
    if tmin < 6: base *= max(0.3, 1 - (6 - tmin) / 8)
    if tmax > 24: base *= max(0.3, 1 - (tmax - 24) / 10)
    return max(0.0, min(1.0, base))

def final_index(api_val: float, et0_7: float, t_fit: float) -> int:
    moisture = max(0.0, min(1.0, (api_val - 0.6 * et0_7) / 40.0 + 0.6))
    return int(round(max(0.0, min(100.0, 100.0 * (0.6 * moisture + 0.4 * t_fit)))))

def best_window(values: List[int]) -> Tuple[int,int,int]:
    if len(values) < 3: return (0, 0, 0)
    best = (0, 2, sum(values[:3])//3)
    for i in range(1, len(values)-2):
        m = (values[i] + values[i+1] + values[i+2]) // 3
        if m > best[2]:
            best = (i, i+2, m)
    return best

def reliability_from_data(past: List[float], future: List[float]) -> float:
    if not past or not future: return 0.5
    mean_past = sum(past) / len(past)
    diffs = [f - mean_past for f in future]
    import statistics as stats
    stdev = stats.pstdev(diffs) if len(diffs) > 1 else 0.0
    return max(0.1, min(0.99, 1.0 / (1.0 + stdev)))

def yield_estimate(idx: int) -> str:
    if idx >= 80: return "6–10+ porcini"
    if idx >= 60: return "2–5 porcini"
    if idx >= 40: return "1–2 porcini"
    return "0–1 porcini"

def slope_aspect_from_elev_grid(grid: List[List[float]], cell_size_m: float = 30.0) -> Tuple[float,float,str]:
    dzdx = ((grid[0][2]+2*grid[1][2]+grid[2][2]) - (grid[0][0]+2*grid[1][0]+grid[2][0]))/(8*cell_size_m)
    dzdy = ((grid[2][0]+2*grid[2][1]+grid[2][2]) - (grid[0][0]+2*grid[0][1]+grid[0][2]))/(8*cell_size_m)
    slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect = (math.degrees(math.atan2(dzdx, dzdy))+360)%360
    octants = ["N","NE","E","SE","S","SW","W","NW","N"]
    octant = octants[int(((aspect%360)+22.5)//45)]
    return round(slope,1), round(aspect,0), octant

# ---------------- External data providers ----------------
HEADERS = {"User-Agent": "Trovaporcini/0.7 (+https://example.com)", "Accept-Language": "it"}

async def fetch_open_meteo(lat: float, lon: float, past: int, future: int) -> Dict[str,any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "daily": ",".join([
            "precipitation_sum",
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "et0_fao_evapotranspiration",
            "relative_humidity_2m_mean"
        ]),
        "past_days": past,
        "forecast_days": future
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

async def fetch_elevation_grid(lat: float, lon: float) -> List[List[float]]:
    deg_lat = 1/111320.0
    deg_lon = 1/(111320.0 * math.cos(math.radians(lat)))
    coords = [{"latitude": lat+dr*deg_lat*30, "longitude": lon+dc*deg_lon*30} for dr in (-1,0,1) for dc in (-1,0,1)]
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": coords})
        resp.raise_for_status()
        res = resp.json()["results"]
    vals = [p["elevation"] for p in res]
    return [vals[0:3], vals[3:6], vals[6:9]]

async def geocode(query: str) -> Dict[str,float]:
    """Ricerca Nominatim con User-Agent per evitare errori 403."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "q": query, "addressdetails": 1, "limit": 1}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    if not data:
        raise HTTPException(404, "Località non trovata")
    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]), "display": data[0].get("display_name","")}

# ---------------- API endpoints ----------------
@app.get("/api/health")
async def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode(q)

@app.get("/api/score")
async def api_score(
    lat: float = Query(...),
    lon: float = Query(...),
    half: float = Query(8.0, gt=3.0, lt=20.0)
):
    # Fetch data in parallel
    meteo_data, elev_grid = await asyncio.gather(
        fetch_open_meteo(lat, lon, past=30, future=10),
        fetch_elevation_grid(lat, lon)
    )
    daily = meteo_data["daily"]
    times = daily["time"]
    precip = [float(x or 0.0) for x in daily["precipitation_sum"]]
    tmean = daily["temperature_2m_mean"]
    tmin = daily["temperature_2m_min"]
    tmax = daily["temperature_2m_max"]
    et0 = daily.get("et0_fao_evapotranspiration",[0.0]*len(precip))
    rh = daily.get("relative_humidity_2m_mean",[None]*len(precip))

    # Past and future arrays
    past = 30; fut = 10
    p_past = precip[:past]; p_future = precip[past:past+fut]
    et_past = et0[:past]; et_future = et0[past:past+fut]
    tmean_past = tmean[:past]; tmin_past = tmin[:past]; tmax_past = tmax[:past]
    tmean_future = tmean[past:past+fut]; tmin_future = tmin[past:past+fut]; tmax_future = tmax[past:past+fut]

    # Current index
    api_val = api_index(p_past, half_life=half)
    et7 = sum(et_past[-7:]) if et_past else 0.0
    t_fit_today = temperature_fit(tmin_past[-1], tmax_past[-1], tmean_past[-1])
    idx_today = final_index(api_val, et7, t_fit_today)

    # Forecast indices next 10 days
    scores = []
    rolling_api = api_val
    coeff = half_life_coeff(half)
    for i in range(fut):
        rolling_api = coeff * rolling_api + (p_future[i] or 0.0)
        et7f = sum(et_future[max(0, i-6):i+1])
        tfit = temperature_fit(tmin_future[i], tmax_future[i], tmean_future[i])
        scores.append(final_index(rolling_api, et7f, tfit))
    wstart, wend, wmean = best_window(scores)

    # Reliability (varianza)
    reliability = reliability_from_data(p_past[-14:], p_future)

    # Rain maps for table
    rain_past = { times[i]: round(p_past[i],1) for i in range(past) }
    rain_future = { times[past+i]: round(p_future[i],1) for i in range(fut) }

    # Elevation, slope, aspect
    elevation = elev_grid[1][1]
    slope_deg, aspect_deg, aspect_oct = slope_aspect_from_elev_grid(elev_grid)

    # Reasons & tips
    reasons = []
    if sum(p_past[-14:]) < 20: reasons.append("Piogge scarse negli ultimi 14 giorni")
    if et7 > 30: reasons.append("Evapotraspirazione elevata (terreno tende a seccarsi)")
    if (sum(p_past[-7:]) + sum(p_future) < 10): reasons.append("Poche piogge passate e previste")
    if not reasons: reasons.append("Buona combinazione di umidità e temperatura")

    tips = []
    if idx_today >= 80: tips.append("Finestra ottima: cerca versanti nord e zone umide con lettiera spessa")
    elif idx_today >= 60: tips.append("Condizioni discrete: cerca nei fondovalle, margini di bosco e conche fresche")
    elif idx_today >= 40: tips.append("Moderato: monitora dopo eventuali piogge, prediligi zone in ombra")
    else: tips.append("Basso potenziale: attendi piogge > 10 mm e 5–10 giorni di maturazione")
    if reliability < 0.6: tips.append("Affidabilità bassa: ricontrolla le previsioni tra 12–24 h")
    if rh[-1] and rh[-1] < 60: tips.append("Umidità bassa: preferisci conche riparate e zone vicine a corsi d’acqua")

    return {
        "index": idx_today,
        "forecast": scores,
        "best_window": {"start": wstart, "end": wend, "mean": wmean},
        "rain_past": rain_past,
        "rain_future": rain_future,
        "harvest_estimate": yield_estimate(idx_today),
        "reliability": round(reliability, 3),
        "explanation": {"reasons": reasons},
        "tips": tips
    }


