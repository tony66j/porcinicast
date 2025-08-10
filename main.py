from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, math, asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

from utils import slope_aspect_from_elev_grid, best_window_3day, deg_to_octant

APP_NAME = "Trovaporcini/0.7 (+https://netlify.app)"
HEADERS = {"User-Agent": APP_NAME, "Accept-Language": "it"}

OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")  # opzionale

app = FastAPI(title="Trova Porcini API", version="0.7.3")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------- Providers ----------------
async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "q": q, "addressdetails": 1, "limit": 1}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    if not j:
        raise HTTPException(404, "Località non trovata")
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
        r = await c.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def open_meteo_hourly(lat: float, lon: float) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "hourly": "precipitation_probability", "forecast_days": 3, "timezone": "auto"}
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
        r = await c.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    deg_lat = 1.0/111320.0
    deg_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat, dlon = step_m*deg_lat, step_m*deg_lon
    coords = [{"latitude": lat + dr*dlat, "longitude": lon + dc*dlon} for dr in (-1,0,1) for dc in (-1,0,1)]
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": coords})
        r.raise_for_status()
        j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]

async def owm_current(lat:float, lon:float)->Optional[Dict[str,Any]]:
    if not OWM_KEY: return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat":lat,"lon":lon,"units":"metric","lang":"it","appid":OWM_KEY}
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()

# ---------------- Scoring helpers (base v0.6 migliorata) ----------------
def k_from_half_life(days: float)->float: return 0.5**(1.0/days)

def api_decay(precip: List[float], half_life_days: float=8.0)->float:
    k = k_from_half_life(half_life_days); api = 0.0
    for p in precip: api = k*api + (p or 0.0)
    return api

def temperature_suitability(tmin: float, tmax: float, tmean: float) -> float:
    if tmin < 1 or tmax > 32: return 0.0
    if tmean <= 6: base = max(0.0, (tmean)/6.0*0.3)
    elif tmean <= 10: base = 0.3 + 0.2*((tmean-6)/4.0)
    elif tmean <= 18: base = 0.5 + 0.5*((tmean-10)/8.0)
    elif tmean <= 22: base = 0.8 - 0.2*((tmean-18)/4.0)
    else: base = max(0.0, 0.6 - 0.6*((tmean-22)/10.0))
    if tmin < 6: base *= max(0.3, 1 - (6 - tmin)/8.0)
    if tmax > 24: base *= max(0.3, 1 - (tmax - 24)/10.0)
    return max(0.0, min(1.0, base))

def api_to_moisture(api: float)->float:
    if api <= 5: return 0.05*api
    if api <= 20: return 0.25 + 0.35*((api-5)/15.0)
    if api <= 70: return 0.60 + 0.35*((api-20)/50.0)
    if api <= 120: return 0.95 - 0.25*((api-70)/50.0)
    return 0.65

def aspect_modifier(aspect_deg: float) -> float:
    octants = {'N':1.08,'NE':1.06,'E':1.0,'SE':0.92,'S':0.85,'SW':0.85,'W':0.92,'NW':1.03}
    return octants.get(deg_to_octant(aspect_deg), 1.0)

def slope_modifier(slope_deg: float)->float:
    if slope_deg < 2: return 0.96
    if slope_deg < 12: return 1.05
    if slope_deg < 25: return 1.0
    return 0.9

def final_score(api_val: float, et0_7: float, t_suit: float, elev_m: float, aspect_deg: float, slope_deg: float) -> Tuple[float, Dict[str,float]]:
    moisture = max(0.0, api_to_moisture(max(0.0, api_val - 0.6*et0_7)))
    elev_mod = 1.05 if 700<=elev_m<=1400 else (0.6 if (elev_m<150 or elev_m>2200) else 0.95)
    asp_m = aspect_modifier(aspect_deg); slp_m = slope_modifier(slope_deg)
    base = 0.6*moisture + 0.4*t_suit
    score = max(0.0, min(100.0, 100.0*base*elev_mod*asp_m*slp_m))
    return score, {"moisture":moisture,"temp":t_suit,"elev_mod":elev_mod,"aspect_mod":asp_m,"slope_mod":slp_m}

def reliability_from_data(past14: List[float], fut10: List[float]) -> float:
    if not past14 or not fut10: return 0.5
    mean_p = sum(past14)/len(past14)
    diffs = [f-mean_p for f in fut10]
    import statistics as st
    stdev = st.pstdev(diffs) if len(diffs)>1 else 0.0
    return max(0.1, min(0.99, 1.0/(1.0+stdev)))

def yield_estimate(idx: int) -> str:
    if idx >= 80: return "6–10+ porcini"
    if idx >= 60: return "2–5 porcini"
    if idx >= 40: return "1–2 porcini"
    return "0–1 porcini"

# ---------------- Endpoints ----------------
@app.get("/api/health")
async def health(): return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/geocode")
async def api_geocode(q: str): return await geocode(q)

@app.get("/api/score")
async def api_score(lat: float = Query(...), lon: float = Query(...), half: float = Query(8.0, gt=3, lt=20)):
    meteo_d, meteo_h, elev_grid, owm_now = await asyncio.gather(
        open_meteo_daily(lat, lon, 30, 10),
        open_meteo_hourly(lat, lon),
        open_elevation_grid(lat, lon),
        owm_current(lat, lon),
    )

    elev_m = float(elev_grid[1][1])
    slope_deg, aspect_deg, aspect_oct = slope_aspect_from_elev_grid(elev_grid, cell_size_m=30.0)

    d = meteo_d["daily"]; times = d["time"]
    P = [float(x or 0.0) for x in d["precipitation_sum"]]
    Tm, Tmin, Tmax = d["temperature_2m_mean"], d["temperature_2m_min"], d["temperature_2m_max"]
    ET0 = d.get("et0_fao_evapotranspiration",[0.0]*len(P))
    RH  = d.get("relative_humidity_2m_mean",[None]*len(P))

    pastN=30; futN=10
    P_past, P_fut = P[:pastN], P[pastN:pastN+futN]
    ET_past, ET_fut = ET0[:pastN], ET0[pastN:pastN+futN]
    Tmin_p, Tmin_f = Tmin[:pastN], Tmin[pastN:pastN+futN]
    Tmax_p, Tmax_f = Tmax[:pastN], Tmax[pastN:pastN+futN]
    Tm_p,  Tm_f   = Tm[:pastN],  Tm[pastN:pastN+futN]

    P7, P14 = sum(P_past[-7:]), sum(P_past[-14:])
    API_val = api_decay(P_past, half_life_days=half)
    ET7 = sum(ET_past[-7:]) if ET_past else 0.0
    Tmean7 = sum(Tm_p[-7:])/max(1,len(Tm_p[-7:])); Tmin7=min(Tmin_p[-7:]); Tmax7=max(Tmax_p[-7:])
    last_rain_days = next((i for i,pv in enumerate(reversed(P_past)) if pv>1.0), -1)

    T_suit_today = temperature_suitability(Tmin_p[-1], Tmax_p[-1], Tm_p[-1])
    today_score, parts = final_score(API_val, ET7, T_suit_today, elev_m, aspect_deg, slope_deg)

    # forecast 10d
    scores=[]; k=k_from_half_life(half); rolling=API_val
    for i in range(futN):
        rolling = k*rolling + (P_fut[i] or 0.0)
        tfit = temperature_suitability(Tmin_f[i], Tmax_f[i], Tm_f[i])
        et7f = sum(ET_fut[max(0,i-6):i+1])
        sc,_ = final_score(rolling, et7f, tfit, elev_m, aspect_deg, slope_deg)
        scores.append(int(round(sc)))
    s,e,m = best_window_3day(scores)

    # Affidabilità
    reliability = reliability_from_data(P_past[-14:], P_fut)

    # Hourly rain prob (media 72h)
    prob = meteo_h.get("hourly", {}).get("precipitation_probability", [])
    prob_rain_next3d = round(sum(prob)/max(1,len(prob)),1) if prob else None

    humid_now = None
    if owm_now:
        try: humid_now = int(owm_now["main"]["humidity"])
        except: pass

    reasons=[]
    if P14 < 20: reasons.append(f"Piogge scarse 14g ({P14:.0f} mm).")
    if API_val < 20: reasons.append("Bilancio idrico basso (API*).")
    if ET7 > 30: reasons.append(f"ET0 7g elevata ({ET7:.0f} mm).")
    if last_rain_days > 10: reasons.append("Ultima pioggia utile oltre 10 giorni fa.")
    if Tmean7 < 10: reasons.append(f"Tmed7 bassa ({Tmean7:.1f} °C).")
    if Tmean7 > 18: reasons.append(f"Tmed7 alta ({Tmean7:.1f} °C).")
    if humid_now is not None and humid_now < 45: reasons.append(f"UR bassa ora ({humid_now}%).")
    if not reasons: reasons.append("Buona combinazione di umidità e termica.")

    tips=[]
    if today_score >= 80: tips.append("Finestra ottima: versanti N–NE, conche umide, lettiera spessa.")
    elif today_score >= 60: tips.append("Discreto: cerca fondovalle e bordi bosco ombreggiati.")
    elif today_score >= 40: tips.append("Moderato: prova dopo piogge ≥10 mm e 2–3 giorni di maturazione.")
    else: tips.append("Basso: attendi nuove piogge e raffreddamento.")
    if reliability < 0.6: tips.append("Affidabilità bassa: ricontrolla tra 12–24 h.")

    rain_past = {times[i]: round(P_past[i],1) for i in range(pastN)}
    rain_fut  = {times[pastN+i]: round(P_fut[i],1) for i in range(futN)}

    return {
        "elevation_m": round(elev_m),
        "slope_deg": round(slope_deg,1),
        "aspect_deg": round(aspect_deg,0),
        "aspect_octant": deg_to_octant(aspect_deg),
        "API_star_mm": round(API_val,1),
        "P7_mm": round(P7,1), "P14_mm": round(P14,1), "ET0_7d_mm": round(ET7,1),
        "Tmean7_c": round(Tmean7,1), "Tmin7_c": round(Tmin7,1), "Tmax7_c": round(Tmax7,1),
        "humidity_now": humid_now, "prob_rain_next3d": prob_rain_next3d,
        "last_rain_days": last_rain_days,
        "score_today": int(round(today_score)),
        "scores_next10": scores,
        "best_window": {"start": s, "end": e, "mean": m},
        "harvest_estimate": yield_estimate(int(round(today_score))),
        "reliability": round(reliability,3),
        "explanation": {"reasons": reasons},
        "tips": tips,
        "rain_past": rain_past, "rain_future": rain_fut
    }



