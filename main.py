# main.py
import math, os, re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

APP_NAME = "Trova Porcini"
USER_AGENT = "TrovaPorcini/1.3 (owner: Antonio Pio D'aloia)"

app = FastAPI(title=APP_NAME, version="1.3.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# -------------------- util --------------------
COORD_RE = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*[,;\s]\s*([+-]?\d+(?:[.,]\d+)?)\s*$")

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

async def _sleep(s: float):
    import asyncio; await asyncio.sleep(s)

async def http_get_json(client: httpx.AsyncClient, url: str, params: Dict, retries: int = 3) -> dict:
    backoff = 0.8
    for _ in range(retries):
        try:
            r = await client.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                pass
            else:
                r.raise_for_status()
        except Exception:
            pass
        await _sleep(backoff); backoff *= 1.8
    raise HTTPException(502, detail=f"Errore contattando {url}")

# -------------------- geocoding --------------------
NOMINATIM = "https://nominatim.openstreetmap.org/search"

async def geocode(q: str, client: httpx.AsyncClient) -> Dict:
    data = await http_get_json(client, NOMINATIM, {
        "q": q, "format": "jsonv2", "limit": 1, "addressdetails": 1
    })
    if not data:
        raise HTTPException(404, detail="Località non trovata")
    hit = data[0]
    return {"name": hit.get("display_name", q), "lat": float(hit["lat"]), "lon": float(hit["lon"])}

# -------------------- meteo --------------------
OPENMETEO = "https://api.open-meteo.com/v1/forecast"
MODELS = ["ecmwf_ifs04", "icon_seamless", "gfs_seamless"]
DAILY = [
    "precipitation_sum", "temperature_2m_mean",
    "et0_fao_evapotranspiration", "shortwave_radiation_sum",
    "relative_humidity_2m_mean", "temperature_2m_min", "temperature_2m_max",
    "wind_speed_10m_max"
]

def _arr(d: dict, key: str) -> List[Optional[float]]:
    vals = d.get(key, [])
    out = []
    for v in vals:
        try: out.append(float(v) if v is not None else None)
        except Exception: out.append(None)
    return out

async def fetch_model(lat: float, lon: float, model: str, client: httpx.AsyncClient) -> Optional[Dict]:
    data = await http_get_json(client, OPENMETEO, {
        "latitude": lat, "longitude": lon, "timezone": "auto", "models": model,
        "daily": ",".join(DAILY), "past_days": 30, "forecast_days": 10
    })
    d = data.get("daily", {})
    if not d or "time" not in d:
        return None
    return {
        "model": model, "time": d["time"],
        "p": _arr(d, "precipitation_sum"),
        "t": _arr(d, "temperature_2m_mean"),
        "tmin": _arr(d, "temperature_2m_min"),
        "tmax": _arr(d, "temperature_2m_max"),
        "et0": _arr(d, "et0_fao_evapotranspiration"),
        "rh": _arr(d, "relative_humidity_2m_mean"),
        "sw": _arr(d, "shortwave_radiation_sum"),
        "wmax": _arr(d, "wind_speed_10m_max"),
        "elev": data.get("elevation")
    }

# -------------------- scoring --------------------
def rolling_sum(vals: List[Optional[float]], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []; acc = 0.0; q: List[float] = []
    for v in vals:
        q.append(0.0 if v is None else v); acc += q[-1]
        if len(q) > window: acc -= q.pop(0)
        out.append(acc if len(q) == window else None)
    return out

def clamp(x, lo, hi): return max(lo, min(hi, x))
def sigmoid(x): return 1/(1+math.exp(-x))
def mean_ignore_none(v):
    vv = [x for x in v if x is not None]
    return sum(vv)/len(vv) if vv else None

def moisture_index(p14, et014):
    if p14 is None or et014 is None: return 0.5
    surplus = p14 - et014
    return clamp(sigmoid(0.12*surplus), 0, 1)  # ~10 mm => ~0.6

def thermo_index(tmean):
    if tmean is None: return 0.5
    if tmean < 6 or tmean > 24: return 0.0
    if 12 <= tmean <= 16: return 1.0
    if tmean < 12: return (tmean-6)/6.0
    return 1.0 - (tmean-16)/8.0

def night_humidity_bonus(rh7, tmin7):
    if rh7 is None or tmin7 is None: return 0.0
    bonus = 0.0
    if rh7 >= 85: bonus += 0.12
    if 6 <= tmin7 <= 14: bonus += 0.08
    return clamp(bonus, 0.0, 0.25)

def drying_penalty(sw7, wmax7):
    pen = 0.0
    if sw7 is not None and sw7 > 18000:  # ~18 MJ/m2/giorno medio (7gg)
        pen += 0.08
    if wmax7 is not None and wmax7 > 10: # >10 m/s massimi
        pen += 0.07
    return clamp(pen, 0.0, 0.2)

def daily_score(moist, thermo, bonus=0.0, penalty=0.0):
    s = 0.6*moist + 0.4*thermo
    s = s + bonus - penalty
    return clamp(s, 0.0, 1.0)*100.0

def reliability_from_spread(series_by_model: List[List[Optional[float]]]) -> float:
    import statistics as stats
    if len(series_by_model) < 2: return 0.65
    days = list(zip(*series_by_model)); cvs=[]
    for d in days:
        vals = [v for v in d if v is not None and v >= 0]
        if len(vals) < 2: continue
        m = sum(vals)/len(vals)
        if m <= 0.2: continue
        try:
            s = stats.pstdev(vals); cv = s/(m if m else 1.0); cvs.append(cv)
        except Exception: pass
    if not cvs: return 0.7
    cv_avg = sum(cvs)/len(cvs)
    return clamp(1/(1+1.8*cv_avg), 0.2, 0.98)

def make_tips(score_today, reliab, last_rain_mm, factors):
    tips = []
    tips.append("Raccogli solo esemplari adulti e integri; rispetta i limiti locali e recidi il gambo con coltellino.")
    tips.append("Usa un cestino areato; evita sacchetti chiusi. Pulisci sul posto per ridurre contaminazioni.")
    if score_today >= 75:
        tips.append("Finestra ottima: faggete/castagneti 900–1500 m, versanti N–NE più umidi.")
    elif score_today >= 55:
        tips.append("Condizioni discrete: cerca in ombra, fondovalle e lettiere spesse.")
    else:
        tips.append("Basso potenziale: attendi pioggia >10–15 mm e verifica dopo 5–10 giorni.")
    if reliab < 0.6:
        tips.append("Affidabilità bassa: modelli discordi; ricontrolla tra 12–24 ore.")
    if last_rain_mm is not None:
        if last_rain_mm >= 10: tips.append("Pioggia utile recente: 5–8 gg a quote medio-alte, 3–5 gg se fa caldo.")
        elif last_rain_mm < 3: tips.append("Pioggia scarsa: punta a impluvi e micro-ristagni.")
    if factors.get("T7_mean_C") and factors["T7_mean_C"] > 20:
        tips.append("Temperatura alta: privilegia esposizioni nord e quote maggiori.")
    if factors.get("RH7_mean_%") and factors["RH7_mean_%"] < 60:
        tips.append("Umidità bassa: cerca in conche e vicino a sorgenti o corsi d’acqua.")
    return tips[:6]

# -------------------- API --------------------
@app.get("/api/health")
async def health():
    return {"ok": True, "app": APP_NAME, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/score")
async def api_score(q: Optional[str] = Query(None), lat: Optional[float] = Query(None), lon: Optional[float] = Query(None)):
    if (lat is None or lon is None) and not q:
        raise HTTPException(400, detail="Fornisci 'q' oppure 'lat' e 'lon'")
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        if lat is not None and lon is not None:
            coord = {"lat": float(lat), "lon": float(lon)}; name = f"{lat:.5f}, {lon:.5f}"
        else:
            maybe = parse_coords(q or "")
            if maybe: coord = maybe; name = f"{coord['lat']:.5f}, {coord['lon']:.5f}"
            else:
                g = await geocode(q, client); coord = {"lat": g["lat"], "lon": g["lon"]}; name = g["name"]

        results = []
        for m in MODELS:
            try:
                r = await fetch_model(coord["lat"], coord["lon"], m, client)
                if r: results.append(r)
            except Exception:
                pass
        if not results:
            raise HTTPException(502, detail="Dati meteo non disponibili")

        base_time = results[0]["time"]; L = len(base_time)
        idx_fc = list(range(max(0, L-10), L))  # ultimi 10 gg = previsione

        # medie 7 gg precedenti all'inizio previsione
        i_end = max(0, L-10); i_start = max(0, i_end-7)

        def avg_across_models(key, slice_):
            vals=[]
            for r in results:
                arr = r[key]
                chunk = [x for x in arr[slice_[0]:slice_[1]] if x is not None]
                if chunk: vals.append(sum(chunk)/len(chunk))
            return sum(vals)/len(vals) if vals else None

        # P14/ET0_14 al giorno di riferimento (ultimo osservato)
        p14_all=[]; et014_all=[]; precip_fc_models=[]; scores_models=[]
        for r in results:
            p14 = rolling_sum(r["p"], 14); et014 = rolling_sum(r["et0"], 14)
            idx_ref = max(0, (L-10)-1)
            p14_all.append(p14[idx_ref] if idx_ref < len(p14) else None)
            et014_all.append(et014[idx_ref] if idx_ref < len(et014) else None)
            precip_fc_models.append([r["p"][i] if i < len(r["p"]) else None for i in idx_fc])

        # indicatori medi 7gg
        t7 = avg_across_models("t", (i_start, i_end))
        rh7 = avg_across_models("rh", (i_start, i_end))
        sw7 = avg_across_models("sw", (i_start, i_end))
        tmin7 = avg_across_models("tmin", (i_start, i_end))
        wmax7 = avg_across_models("wmax", (i_start, i_end))
        elev = round(sum([r["elev"] for r in results if r["elev"] is not None])/len([r for r in results if r["elev"] is not None])) if any(r["elev"] for r in results) else None

        # score giornaliero medio sui 10gg
        scores_models = []
        for r in results:
            p14 = rolling_sum(r["p"], 14); et014 = rolling_sum(r["et0"], 14)
            model_scores=[]
            for i in range(L):
                moist = moisture_index(p14[i], et014[i])
                thermo = thermo_index(r["t"][i] if i < len(r["t"]) else None)
                bonus = night_humidity_bonus(rh7, tmin7)
                pen   = drying_penalty(sw7, wmax7)
                model_scores.append(daily_score(moist, thermo, bonus, pen))
            scores_models.append(model_scores)

        forecast=[]
        for i in idx_fc:
            vals=[sm[i] for sm in scores_models]
            forecast.append({"date": base_time[i], "score": round(sum(vals)/len(vals), 1)})

        # today entry
        today_str = date.today().isoformat()
        today_entry = next((x for x in forecast if x["date"] == today_str), None) or forecast[0]

        # affidabilità
        reliability = reliability_from_spread(precip_fc_models)

        # ultima pioggia >=1 mm negli ultimi 30 gg osservati (media modelli)
        last_rain = None
        for i in range(i_end-1, -1, -1):
            mm = [r["p"][i] for r in results if r["p"][i] is not None]
            if not mm: continue
            mavg = sum(mm)/len(mm)
            if mavg >= 1.0:
                last_rain = {"date": base_time[i], "amount_mm": round(mavg, 1)}
                break

        factors = {
            "latitude": coord["lat"], "longitude": coord["lon"], "elevation_m": elev,
            "P14_mm": round(mean_ignore_none(p14_all) or 0, 1),
            "ET0_14_mm": round(mean_ignore_none(et014_all) or 0, 1),
            "T7_mean_C": round(t7 or 0, 1) if t7 is not None else None,
            "RH7_mean_%": round(rh7 or 0) if rh7 is not None else None,
            "SWrad7_MJm2": round((sw7 or 0)/1000.0, 2) if sw7 is not None else None,
            "Tmin7_C": round(tmin7 or 0, 1) if tmin7 is not None else None,
            "WindMax7_ms": round(wmax7 or 0, 1) if wmax7 is not None else None,
            "models_used": [r["model"] for r in results],
        }

        tips = make_tips(today_entry["score"], reliability, last_rain["amount_mm"] if last_rain else None, factors)

        return {
            "location": {"name": name, "lat": coord["lat"], "lon": coord["lon"]},
            "today": today_entry, "forecast": forecast, "reliability": round(reliability, 3),
            "last_rain": last_rain, "factors": factors, "tips": tips
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)







