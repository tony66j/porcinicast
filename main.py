from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import httpx
import math
import os

app = FastAPI(title="TrovaPorcini API v3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- utilità semplici ----------
def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def score_from_rain(p14: float) -> float:
    # 0→0, 30mm→50, 60mm→80, 100mm→100
    if p14 <= 0: return 0.0
    if p14 >= 100: return 100.0
    # log dolce
    return clamp(25.0 * math.log1p(p14), 0.0, 100.0)

def score_from_temp(tmean7: float) -> float:
    # fascia ideale 13–19°C, scende fuori
    if tmean7 is None: return 0.0
    if 13 <= tmean7 <= 19:
        return 100.0
    dist = min(abs(tmean7-13), abs(tmean7-19))
    return clamp(100.0 - dist*18.0, 0.0, 100.0)

def est_caps_from_index(idx: float, compat: float, uncertainty: float) -> int:
    # cappelli ≈ scala non lineare dell’indice con compatibilità bosco e incertezza
    base = (idx/100.0)**1.6 * 20.0   # 0..~20
    base *= clamp(compat, 0.6, 1.2)  # 0.6..1.2
    base *= clamp(1.0 - 0.5*uncertainty, 0.5, 1.0)
    return int(round(base))

async def open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join(["precipitation_sum", "temperature_2m_mean"]),
        "past_days": 14, "forecast_days": 10,
        "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

def p14_from_daily(daily: Dict[str, Any]) -> float:
    prec = daily.get("precipitation_sum") or []
    return float(sum(prec[:14])) if len(prec) >= 14 else float(sum(prec))

def tmean7_from_daily(daily: Dict[str, Any]) -> Optional[float]:
    t = daily.get("temperature_2m_mean") or []
    vals = t[:7] if len(t) >= 7 else t
    try:
        return sum(vals)/len(vals) if vals else None
    except Exception:
        return None

def last_rain_days(daily: Dict[str, Any]) -> Optional[int]:
    prec = daily.get("precipitation_sum") or []
    for i, mm in enumerate(prec):
        if mm and mm >= 1.0:
            return i  # i giorni fa (0=today)
    return None

def next_rain_info(daily: Dict[str, Any]) -> Dict[str, Any]:
    prec = daily.get("precipitation_sum") or []
    if not prec: return {"next_rain_in_days": None, "sum_next10mm": 0.0}
    for d, mm in enumerate(prec[1:], start=1):
        if mm and mm >= 1.0:
            return {"next_rain_in_days": d, "sum_next10mm": float(sum(prec[1:]))}
    return {"next_rain_in_days": None, "sum_next10mm": float(sum(prec[1:]))}

def advice_from_conditions(idx: float, tmean7: Optional[float], last_rain: Optional[int]) -> List[str]:
    tips = []
    if idx >= 70:
        tips.append("Vai presto e punta i faggi o castagneti (lettiera umida).")
    elif idx >= 40:
        tips.append("Cerca zone ombrose e fresche, con suolo che trattiene umidità.")
    else:
        tips.append("Probabilità bassa: esplora quote/versanti diversi o attendi piogge.")
    if tmean7 is not None:
        if tmean7 > 20: tips.append("Temperature alte: privilegia esposizioni N–NE, fondovalle e suoli freschi.")
        elif tmean7 < 11: tips.append("Temperature basse: scegli ore centrali e versanti S–SE riparati dal vento.")
    if last_rain is not None and last_rain >= 10:
        tips.append("Piogge lontane: dai priorità a suoli profondi e muscosi.")
    return tips

@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/api/score")
async def api_score(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    day: int = Query(0, ge=0, le=9)  # 0=oggi, 1..9
) -> Dict[str, Any]:
    """
    Ritorna indice 0..100 e spiegazioni. Robusto: nessun 500, solo fallback.
    """
    try:
        data = await open_meteo(lat, lon)
        daily = data.get("daily") or {}
    except Exception as e:
        # fallback “sicuro”: indice 0 con spiegazione, mai 500
        return {
            "index": 0, "caps3h": 0,
            "advice": ["Servizio meteo non disponibile. Riprova tra poco."],
            "why": "Errore nel recupero dati meteo.",
            "tech": {"lat": lat, "lon": lon, "source": "open-meteo", "error": str(e)},
            "uncertainty": 1.0
        }

    P14 = p14_from_daily(daily)
    T7  = tmean7_from_daily(daily)
    lr  = last_rain_days(daily)
    nr  = next_rain_info(daily)

    # indice “oggi” e per giorno selezionato (se day>0, usa piogge future cumulative)
    rain_component = score_from_rain(P14 + (daily.get("precipitation_sum")[1:1+day] and sum(daily["precipitation_sum"][1:1+day]) or 0.0))
    temp_component = score_from_temp(T7 if day == 0 else (daily.get("temperature_2m_mean")[day] if daily.get("temperature_2m_mean") else T7))
    idx = clamp(0.55*rain_component + 0.45*temp_component, 0.0, 100.0)

    # compatibilità bosco “stimata” (senza OSM qui): neutra 1.0
    compat = 1.0
    # incertezza: cresce se mancano T7 o se piogge quasi nulle
    uncertainty = 0.3 + (0.2 if T7 is None else 0.0) + (0.2 if P14 < 5 else 0.0)
    uncertainty = clamp(uncertainty, 0.2, 0.9)

    caps = est_caps_from_index(idx, compat, uncertainty)

    why_parts = []
    why_parts.append(f"P14 ultimi 14 gg: {round(P14,1)} mm")
    if T7 is not None:
        why_parts.append(f"Tmedia 7 gg: {round(T7,1)} °C")
    if lr is not None:
        why_parts.append(f"Ultima pioggia: ~{lr} gg fa")
    if nr["next_rain_in_days"] is not None:
        why_parts.append(f"Prossima pioggia tra ~{nr['next_rain_in_days']} gg (tot prossimi 10 gg: {round(nr['sum_next10mm'],1)} mm)")

    adv = advice_from_conditions(idx, T7, lr)

    return {
        "index": int(round(idx)),
        "caps3h": caps,
        "advice": adv,
        "why": "; ".join(why_parts) if why_parts else "Dati incompleti.",
        "tech": {
            "lat": lat, "lon": lon,
            "P14mm": round(P14,1),
            "Tmean7C": None if T7 is None else round(T7,1),
            "last_rain_days": lr,
            "next_rain_in_days": nr["next_rain_in_days"],
            "sum_next10mm": round(nr["sum_next10mm"],1),
            "day": day, "source": "open-meteo"
        },
        "uncertainty": round(uncertainty,2),
        "species": ["Boletus edulis (stima)"]
    }




