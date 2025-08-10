# main.py
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx

APP_NAME = "Trova Porcini"
USER_AGENT = "TrovaPorcini/1.2 (contact: antonio@example.com)"  # personalizza

app = FastAPI(title=APP_NAME, version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Utility
# -------------------------

COORD_RE = re.compile(
    r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*[,;\s]\s*([+-]?\d+(?:[.,]\d+)?)\s*$",
    re.IGNORECASE,
)

def parse_coords(text: str) -> Optional[Dict[str, float]]:
    if not text:
        return None
    m = COORD_RE.match(text.strip())
    if not m:
        return None
    lat = float(m.group(1).replace(",", "."))
    lon = float(m.group(2).replace(",", "."))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return {"lat": lat, "lon": lon}

async def http_get_json(client: httpx.AsyncClient, url: str, params: Dict, max_retries: int = 3) -> dict:
    delay = 0.7
    for _ in range(max_retries):
        try:
            r = await client.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                pass  # ritenta
            else:
                r.raise_for_status()
        except Exception:
            pass
        await _sleep(delay)
        delay *= 1.8
    raise HTTPException(status_code=502, detail=f"Upstream error contacting {url}")

async def _sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)

# -------------------------
# Geocoding
# -------------------------

NOMINATIM = "https://nominatim.openstreetmap.org/search"

async def geocode(query: str, client: httpx.AsyncClient) -> Dict:
    params = {"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1}
    data = await http_get_json(client, NOMINATIM, params)
    if not data:
        raise HTTPException(404, detail="Località non trovata")
    hit = data[0]
    return {"name": hit.get("display_name", query), "lat": float(hit["lat"]), "lon": float(hit["lon"])}

# -------------------------
# Meteo (Open-Meteo multi-modello)
# -------------------------

OPENMETEO = "https://api.open-meteo.com/v1/forecast"

MODEL_LIST = [
    "ecmwf_ifs04",
    "icon_seamless",
    "gfs_seamless",
]

DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "relative_humidity_2m_mean",
]

def today_iso() -> str:
    return date.today().isoformat()

async def fetch_model(lat: float, lon: float, model: str, client: httpx.AsyncClient) -> Optional[Dict]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
        "models": model,
        "past_days": 30,        # per P14, ultima pioggia, medie 7gg
        "forecast_days": 10,    # previsione futura 10 giorni
    }
    try:
        data = await http_get_json(client, OPENMETEO, params)
        d = data.get("daily", {})
        if not d or "time" not in d:
            return None
        # helper per estrarre serie numeriche con fallback a None
        def arr(key):
            vals = d.get(key, [])
            out = []
            for v in vals:
                try:
                    out.append(float(v) if v is not None else None)
                except Exception:
                    out.append(None)
            return out
        return {
            "model": model,
            "time": d["time"],
            "precip": arr("precipitation_sum"),
            "tmean": arr("temperature_2m_mean"),
            "et0": arr("et0_fao_evapotranspiration"),
            "swrad": arr("shortwave_radiation_sum"),
            "rh": arr("relative_humidity_2m_mean"),
            "elevation": data.get("elevation"),
        }
    except Exception:
        return None

# -------------------------
# Scoring
# -------------------------

def rolling_sum(vals: List[Optional[float]], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    acc = 0.0
    q: List[float] = []
    for v in vals:
        q.append(0.0 if v is None else v)
        acc += q[-1]
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc if len(q) == window else None)
    return out

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def moisture_index(p14: Optional[float], et014: Optional[float]) -> float:
    if p14 is None or et014 is None:
        return 0.5
    surplus = p14 - et014
    m = sigmoid(0.12 * surplus)  # ~10 mm -> ~0.6
    return clamp(m, 0.0, 1.0)

def thermo_index(tmean: Optional[float]) -> float:
    if tmean is None:
        return 0.5
    if tmean < 6 or tmean > 24:
        return 0.0
    if 12 <= tmean <= 16:
        return 1.0
    if 6 <= tmean < 12:
        return (tmean - 6) / 6.0
    return 1.0 - (tmean - 16) / 8.0

def daily_score(moist: float, thermo: float) -> float:
    w_m, w_t = 0.6, 0.4
    return clamp(w_m * moist + w_t * thermo, 0.0, 1.0) * 100.0

def reliability_from_spread(series_by_model: List[List[Optional[float]]]) -> float:
    import statistics as stats
    if len(series_by_model) < 2:
        return 0.65
    days = list(zip(*series_by_model))
    cvs = []
    for day_vals in days:
        vals = [v for v in day_vals if (v is not None and v >= 0)]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        if mean <= 0.2:
            continue
        try:
            stdev = stats.pstdev(vals)
            cv = stdev / (mean if mean != 0 else 1.0)
            cvs.append(cv)
        except Exception:
            pass
    if not cvs:
        return 0.7
    cv_avg = sum(cvs) / len(cvs)
    rel = 1.0 / (1.0 + 1.8 * cv_avg)
    return clamp(rel, 0.2, 0.98)

def mean_ignore_none(vals: List[Optional[float]]) -> Optional[float]:
    vs = [v for v in vals if v is not None]
    return (sum(vs) / len(vs)) if vs else None

# -------------------------
# API
# -------------------------

@app.get("/api/health")
async def health():
    return {"ok": True, "app": APP_NAME, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/score")
async def api_score(
    q: Optional[str] = Query(None, description="Nome località oppure 'lat,lon'"),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
):
    if (lat is None or lon is None) and not q:
        raise HTTPException(400, detail="Fornisci 'q' oppure 'lat' e 'lon'")
    coord = None
    loc_name = None

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        if lat is not None and lon is not None:
            coord = {"lat": float(lat), "lon": float(lon)}
            loc_name = f"{lat:.5f}, {lon:.5f}"
        else:
            maybe = parse_coords(q or "")
            if maybe:
                coord = maybe
                loc_name = f"{coord['lat']:.5f}, {coord['lon']:.5f}"
            else:
                g = await geocode(q, client)
                coord = {"lat": g["lat"], "lon": g["lon"]}
                loc_name = g["name"]

        # multi-modello
        results: List[Dict] = []
        for model in MODEL_LIST:
            data = await fetch_model(coord["lat"], coord["lon"], model, client)
            if data:
                results.append(data)
        if not results:
            raise HTTPException(502, detail="Meteo non disponibile per questa posizione")

        base_time = results[0]["time"]
        L = len(base_time)

        # indici per i 10 giorni di previsione (ultimi 10)
        last_10_idx = list(range(max(0, L - 10), L))

        # per affidabilità e fattori
        precip_models: List[List[Optional[float]]] = []
        model_names: List[str] = []
        elevation_vals: List[Optional[float]] = []

        # calcolo score giornalieri media-modello
        # prima computiamo per ciascun modello
        scores_by_model: List[List[float]] = []
        p14_by_model: List[Optional[float]] = []
        et014_by_model: List[Optional[float]] = []
        t7_by_model: List[Optional[float]] = []
        rh7_by_model: List[Optional[float]] = []
        sw7_by_model: List[Optional[float]] = []
        last_rain_candidates: List[Optional[Dict]] = []

        for res in results:
            model_names.append(res["model"])
            elevation_vals.append(res.get("elevation"))

            p = res["precip"]
            t = res["tmean"]
            et0 = res["et0"]
            rh = res["rh"]
            sw = res["swrad"]

            p14_series = rolling_sum(p, 14)
            et014_series = rolling_sum(et0, 14)

            # calcola score per ciascun giorno (per la serie completa)
            scores = []
            for i in range(L):
                mi = moisture_index(p14_series[i], et014_series[i])
                th = thermo_index(t[i] if i < len(t) else None)
                scores.append(daily_score(mi, th))
            scores_by_model.append(scores)

            # medie per fattori (ultimi 7 giorni osservati disponibili, non forecast)
            # i








