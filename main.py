
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, math, asyncio
from typing import Dict, List, Tuple
from datetime import datetime, timezone

app = FastAPI(title="Trova Porcini API", version="0.7.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------- Helpers (manteniamo logica base 0.6) ----------------
def half_life_coeff(days: float) -> float:
    """Fattore di decadimento per P altrettanto alla ½ every days."""
    return 0.5 ** (1.0 / days)

def api_index(precip: List[float], half_life: float = 8.0) -> float:
    """Accumulated Precipitation Index (API*) con decadimento esponenziale."""
    k = half_life_coeff(half_life)
    api = 0.0
    for p in precip:
        api = k * api + (p or 0.0)
    return api

def temperature_fit(tmin: float, tmax: float, tmean: float) -> float:
    """Stima quanto la temperatura è ottimale per porcini (0-1)."""
    if tmin < 1 or tmax > 32: return 0.0
    if tmean <= 6: base = max(0.0, (tmean) / 6 * 0.3)
    elif tmean <= 10: base = 0.3 + 0.2 * ((tmean - 6) / 4)
    elif tmean <= 18: base = 0.5 + 0.5 * ((tmean - 10) / 8)
    elif tmean <= 22: base = 0.8 - 0.2 * ((tmean - 18) / 4)
    else: base = max(0.0, 0.6 - 0.6 * ((tmean - 22) / 10))
    if tmin < 6: base *= max(0.3, 1 - (6 - tmin) / 8)
    if tmax > 24: base *= max(0.3, 1 - (tmax - 24) / 10)
    return max(0.0, min(1.0, base))

def final_index(api_val: float, et0: float, t_fit: float) -> int:
    """Indice finale 0-100 combinando umidità (API-ET0) e finestra termica."""
    # normalizzazione umidità 0-1: 0 mm -> 0.2; 50 mm -> 1.0
    moisture = max(0.0, min(1.0, (api_val - 0.6 * et0) / 40.0 + 0.6))
    return int(round(max(0.0, min(100.0, 100.0 * (0.6 * moisture + 0.4 * t_fit)))))

def best_3day_window(values: List[int]) -> Tuple[int,int,int]:
    """Restituisce (start_index, end_index, mean) per finestra 3 giorni migliore."""
    if len(values) < 3: return (0, 0, 0)
    best = (0, 2, sum(values[:3])//3)
    for i in range(1, len(values)-2):
        m = (values[i] + values[i+1] + values[i+2]) // 3
        if m > best[2]:
            best = (i, i+2, m)
    return best

def reliability_from_data(past: List[float], forecast: List[float]) -> float:
    """
    Stima affidabilità (0-1) in base a quanto la pioggia futura differisce dalla media passata.
    Maggiore discrepanza => meno affidabile.
    """
    import statistics as stats
    if not past or not forecast: return 0.5
    mean_past = sum(past) / len(past)
    diffs = [(f - mean_past) for f in forecast]
    stdev = stats.pstdev(diffs) if len(diffs) > 1 else 0.0
    # normalizza: varianza bassa -> alta affidabilità; varianza alta -> bassa affidabilità
    return max(0.1, min(0.99, 1.0 / (1.0 + stdev)))

def yield_estimate(idx: int) -> str:
    """Stima numero di porcini in 2h di raccolta in base all'indice."""
    if idx >= 80: return "6–10+ porcini"
    if idx >= 60: return "2–5 porcini"
    if idx >= 40: return "1–2 porcini"
    return "0–1 porcini"

# ---------------- API Calls ----------------
async def get_open_meteo(lat: float, lon: float, past: int = 30, future: int = 10) -> Dict[str, any]:
    """Richiede dati climatici (precip, temp, et0, radiazione, umidità) per passato e futuro."""
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
        "past_days": past, "forecast_days": future
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def get_elevation_grid(lat: float, lon: float) -> List[List[float]]:
    """Richiede griglia 3x3 da Open Elevation per pendenza/esposizione."""
    deg_lat = 1/111320.0
    deg_lon = 1/(111320.0 * math.cos(math.radians(lat)))
    coords = [{"latitude": lat+dr*deg_lat*30, "longitude": lon+dc*deg_lon*30} for dr in (-1,0,1) for dc in (-1,0,1)]
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": coords})
        r.raise_for_status()
        res = r.json()["results"]
    vals = [p["elevation"] for p in res]
    return [vals[0:3], vals[3:6], vals[6:9]]

async def geocode(q: str) -> Dict[str, float]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    if not j: raise HTTPException(404, "Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0]["display_name"]}

# ---------------- Endpoints ----------------
@app.get("/api/geocode")
async def api_geocode_endpoint(q: str):
    return await geocode(q)

@app.get("/api/score")
async def api_score(
    lat: float = Query(...),
    lon: float = Query(...),
    half: float = Query(8.0, gt=3.0, lt=15.0)
):
    # recupera dati
    meteo, elev = await asyncio.gather(get_open_meteo(lat, lon), get_elevation_grid(lat, lon))
    daily = meteo["daily"]
    times = daily["time"]
    p = [float(x or 0.0) for x in daily["precipitation_sum"]]
    tmean = daily["temperature_2m_mean"]
    tmin = daily["temperature_2m_min"]
    tmax = daily["temperature_2m_max"]
    et0 = daily.get("et0_fao_evapotranspiration",[0.0]*len(p))
    rh = daily.get("relative_humidity_2m_mean",[None]*len(p))

    pastN = 30
    p_past = p[:pastN]; p_future = p[pastN:pastN+10]
    et_past = et0[:pastN]; et_future = et0[pastN:pastN+10]
    tmean_past = tmean[:pastN]; tmin_past = tmin[:pastN]; tmax_past = tmax[:pastN]
    tmean_future = tmean[pastN:pastN+10]; tmin_future = tmin[pastN:pastN+10]; tmax_future = tmax[pastN:pastN+10]

    api_val = api_index(p_past, half_life=half)
    et7 = sum(et_past[-7:]) if et_past else 0.0
    t_suit_today = temperature_fit(tmin_past[-1], tmax_past[-1], tmean_past[-1])
    idx_today = final_index(api_val, et7, t_suit_today)

    # calcola indici futuri (10 giorni)
    scores = []
    rolling_api = api_val
    coeff = half_life_coeff(half)
    for i in range(10):
        rolling_api = coeff * rolling_api + (p_future[i] or 0.0)
        et7f = sum(et_future[max(0,i-6):i+1])
        tfit = temperature_fit(tmin_future[i], tmax_future[i], tmean_future[i])
        scores.append(final_index(rolling_api, et7f, tfit))

    s,e,m = best_3day_window(scores)

    # stima affidabilità: varianza delle piogge future vs media passata
    reliability = reliability_from_data(p_past[-14:], p_future)

    # piogge passate/future mappate
    rain_past = { times[i]: round(p_past[i],1) for i in range(pastN) }
    rain_future = { times[pastN+i]: round(p_future[i],1) for i in range(10) }

    elev_m = elev[1][1]
    slope_deg, aspect_deg, aspect_oct = slope_aspect_from_elev_grid(elev)

    # spiegazioni base
    reasons=[]
    if sum(p_past[-14:]) < 20: reasons.append("Piogge scarse negli ultimi 14 giorni")
    if et7 > 30: reasons.append("Evapotraspirazione elevata (terreno tende a seccarsi)")
    if (sum(p_past[-7:]) < 5 and sum(p_future) < 5): reasons.append("Piogge insufficienti in passato e previste")
    if not reasons: reasons.append("Condizioni idonee (buona umidità e termica)")

    # consigli dinamici
    tips=[]
    if idx_today >= 80: tips.append("Finestra ottima: cerca zone ombrose con lettiera spessa, specialmente su versanti N-NE.")
    elif idx_today >= 60: tips.append("Condizioni discrete: privilegia i fondovalle, margini di bosco e conche umide.")
    elif idx_today >= 40: tips.append("Moderato: monitora dopo eventuali piogge e preferisci aree con suolo profondo.")
    else: tips.append("Basso potenziale: attendi piogge ≥10 mm e un paio di giorni di maturazione.")
    if reliability < 0.6: tips.append("Affidabilità bassa: ricontrolla le previsioni tra 12–24 h.")
    if rh[-1] is not None and rh[-1] < 60: tips.append("Umidità dell’aria bassa: prediligi conche riparate e vicino a corsi d’acqua.")

    return {
        "index": idx_today,
        "forecast": scores,
        "best_window": { "start": s, "end": e, "mean": m },
        "rain_past": rain_past,
        "rain_future": rain_future,
        "harvest_estimate": yield_estimate(idx_today),
        "reliability": round(reliability,3),
        "explanation": { "reasons": reasons },
        "tips": tips
    }

