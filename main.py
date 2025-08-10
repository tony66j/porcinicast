
# main.py
import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx

APP_NAME = "PorciniCast"
USER_AGENT = "PorciniCast/1.0 (contact: app@example.com)"  # personalizza se vuoi

app = FastAPI(title=APP_NAME, version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Utility / parsing
# -------------------------

COORD_RE = re.compile(
    r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*[,;\s]\s*([+-]?\d+(?:[.,]\d+)?)\s*$",
    re.IGNORECASE,
)

def parse_coords(text: str) -> Optional[Dict[str, float]]:
    """
    Accetta "lat, lon" anche con virgola decimale italiana.
    """
    if not text:
        return None
    m = COORD_RE.match(text.strip())
    if not m:
        return None
    lat = float(m.group(1).replace(",", "."))
    lon = float(m.group(2).replace(",", "."))
    # limiti rapidi
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return {"lat": lat, "lon": lon}

async def http_get_json(client: httpx.AsyncClient, url: str, params: Dict, max_retries: int = 3) -> dict:
    """
    GET con retry/backoff semplice su errori di rete o 5xx/429.
    """
    delay = 0.7
    for attempt in range(max_retries):
        try:
            r = await client.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                await httpx.AsyncClient().aclose()
                # backoff
            else:
                # 4xx diversi da 429: errore "duro"
                r.raise_for_status()
        except Exception:
            pass
        await asyncio_sleep(delay)
        delay *= 1.8
    raise HTTPException(status_code=502, detail=f"Upstream error contacting {url}")

async def asyncio_sleep(seconds: float):
    """piccolo wrapper per evitare import di asyncio all'inizio."""
    import asyncio
    await asyncio.sleep(seconds)

# -------------------------
# Geocoding (Nominatim)
# -------------------------

NOMINATIM = "https://nominatim.openstreetmap.org/search"

async def geocode(query: str, client: httpx.AsyncClient) -> Dict:
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }
    data = await http_get_json(client, NOMINATIM, params)
    if not data:
        raise HTTPException(404, detail="Località non trovata")
    hit = data[0]
    return {
        "name": hit.get("display_name", query),
        "lat": float(hit["lat"]),
        "lon": float(hit["lon"]),
    }

# -------------------------
# Meteo (Open-Meteo, multi-modello)
# -------------------------

OPENMETEO = "https://api.open-meteo.com/v1/forecast"

# elenco modelli; se uno non è disponibile viene ignorato
MODEL_LIST = [
    "ecmwf_ifs04",     # ECMWF HRES ~0.4°
    "icon_seamless",   # DWD ICON (global/eur) seamless
    "gfs_seamless",    # NOAA GFS
]

DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "et0_fao_evapotranspiration",
]

def today_iso() -> str:
    return date.today().isoformat()

def plus_days_iso(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()

async def fetch_model(lat: float, lon: float, model: str, client: httpx.AsyncClient) -> Optional[Dict]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
        "models": model,
        "past_days": 14,      # per avere finestra P14/ET0_14
        "forecast_days": 10,  # barra 10 giorni
    }
    try:
        data = await http_get_json(client, OPENMETEO, params)
        d = data.get("daily", {})
        # validazione base
        if not d or "time" not in d or "precipitation_sum" not in d:
            return None
        return {
            "model": model,
            "time": d["time"],
            "precip": d.get("precipitation_sum", []),
            "tmean": d.get("temperature_2m_mean", []),
            "et0": d.get("et0_fao_evapotranspiration", []),
            "elevation": data.get("elevation"),
        }
    except Exception:
        return None

# -------------------------
# Scoring micologico (semplice ma fisicamente sensato)
# -------------------------

def rolling_sum(vals: List[float], window: int) -> List[float]:
    out = []
    acc = 0.0
    q = []
    for v in vals:
        q.append(v or 0.0)
        acc += q[-1]
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc if len(q) == window else None)
    return out

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def moisture_index(p14: float, et014: float) -> float:
    """
    Indice umidità 0..1: usa surplus idrico (P14-ET0_14) in mm,
    poi mappa con sigmoide. Centro ~ +10 mm => 0.5-0.6
    """
    if p14 is None or et014 is None:
        return 0.4  # fallback neutro
    surplus = p14 - et014
    # scala: 0.12 ~ 8–12 mm / 14gg
    m = sigmoid(0.12 * (surplus))
    return clamp(m, 0.0, 1.0)

def thermo_index(tmean: float) -> float:
    """
    Finestra termica dolce per Boletus edulis s.l.
    Ottimo 12–16°C, utile 8–20°C, calo oltre.
    """
    if tmean is None:
        return 0.5
    # triangolare "morbida" con plateau tra 12-16
    if tmean < 6 or tmean > 24:
        return 0.0
    if 12 <= tmean <= 16:
        return 1.0
    if 6 <= tmean < 12:
        return (tmean - 6) / 6.0
    # 16 < t <= 24
    return 1.0 - (tmean - 16) / 8.0

def daily_score(moist: float, thermo: float) -> float:
    """
    Combina umidità e termica; leggera enfatizzazione dell'umidità.
    """
    w_m, w_t = 0.6, 0.4
    s = w_m * moist + w_t * thermo
    return clamp(s, 0.0, 1.0) * 100.0

def reliability_from_spread(series_by_model: List[List[float]]) -> float:
    """
    Affidabilità 0..1 dalla dispersione tra modelli su precipitazione totale 10 giorni.
    Bassa varianza => alta affidabilità. Calcoliamo CV medio dei giorni (escludiamo zeri assoluti).
    """
    import statistics as stats
    if len(series_by_model) < 2:
        return 0.65  # fallback
    # trasponi sui giorni
    days = list(zip(*series_by_model))
    cvs = []
    for day_vals in days:
        vals = [v for v in day_vals if v is not None]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        if mean <= 0.2:  # piogge trascurabili => poco informative
            continue
        try:
            stdev = stats.pstdev(vals)
            cv = stdev / mean  # coefficiente di variazione
            cvs.append(cv)
        except Exception:
            pass
    if not cvs:
        return 0.7
    cv_avg = sum(cvs) / len(cvs)
    # mappa CV a 0..1 (cv 0.15 => ~0.9; cv 0.6 => ~0.5)
    rel = 1.0 / (1.0 + 1.8 * cv_avg)
    return clamp(rel, 0.2, 0.98)

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
    """
    Restituisce:
      - location {name, lat, lon}
      - today {date, score}
      - forecast [{date, score}]
      - reliability 0..1
      - models_used [...]
    """
    if (lat is None or lon is None) and not q:
        raise HTTPException(400, detail="Fornisci 'q' oppure 'lat' e 'lon'")
    coord = None
    loc_name = None

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        if lat is not None and lon is not None:
            coord = {"lat": float(lat), "lon": float(lon)}
            loc_name = f"{lat:.5f}, {lon:.5f}"
        else:
            # prova a capire se q sono coordinate
            maybe = parse_coords(q or "")
            if maybe:
                coord = {"lat": maybe["lat"], "lon": maybe["lon"]}
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

        # normalizza timeline: usiamo l'ordinamento della prima serie
        base_time = results[0]["time"]
        L = len(base_time)

        # estrai vettori per ciascun modello
        model_names = []
        precip_models = []
        score_models = []

        # calcola P14/ET0_14 e score per ogni giorno utile
        forecast_out = []
        idx_today = None

        # individua indici per i 10 gg di previsione (ultimi 10 della serie)
        # perché abbiamo past_days=14 + forecast_days=10
        # Esempio: time[0..23] => ultimi 10 sono la previsione
        last_10_idx = list(range(max(0, L - 10), L))

        for res in results:
            model_names.append(res["model"])
            p = [float(x) if x is not None else 0.0 for x in res["precip"]]
            t = [float(x) if x is not None else None for x in res["tmean"]]
            et0 = [float(x) if x is not None else 0.0 for x in res["et0"]]
            p14 = rolling_sum(p, 14)
            et014 = rolling_sum(et0, 14)

            # costruisci score per ogni giorno
            scores = []
            for i in range(L):
                mi = moisture_index(p14[i] if i < len(p14) else None, et014[i] if i < len(et014) else None)
                th = thermo_index(t[i] if i < len(t) else None)
                s = daily_score(mi, th)
                scores.append(s)

            # raccogli per affidabilità (precip solo sui 10gg)
            precip_models.append([p[i] for i in last_10_idx])
            score_models.append([scores[i] for i in last_10_idx])

        # affidabilità dalla dispersione precipitazioni
        reliability = reliability_from_spread(precip_models)

        # output forecast (media tra modelli)
        forecast_out = []
        for k, i in enumerate(last_10_idx):
            day = base_time[i]
            # media tra modelli
            vals = [scores[i] for scores in score_models]
            mean_score = sum(vals) / len(vals)
            forecast_out.append({"date": day, "score": round(mean_score, 1)})

        # today = primo della finestra forecast se corrisponde a data odierna, altrimenti ultimo della storia
        today_str = today_iso()
        today_obj = None
        today_entry = None
        for item in forecast_out:
            if item["date"] == today_str:
                today_entry = item
                break
        if today_entry is None:
            # prendi quello con data minima > oggi? altrimenti l'ultimo disponibile
            try:
                today_entry = next((x for x in forecast_out if x["date"] >= today_str), forecast_out[0])
            except StopIteration:
                today_entry = forecast_out[-1]

        return {
            "location": {"name": loc_name, "lat": coord["lat"], "lon": coord["lon"]},
            "today": today_entry,
            "forecast": forecast_out,
            "reliability": round(reliability, 3),
            "models_used": model_names,
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)







