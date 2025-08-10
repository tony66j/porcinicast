from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone

app = FastAPI(title="Trova Porcini API (stable+fallback)", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

HEADERS = {"User-Agent":"Trovaporcini/1.1 (+site)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")  # imposta su Render

# ----------------- Utilità numeriche/scientifiche -----------------
def half_life_coeff(days: float) -> float:
    return 0.5 ** (1.0 / max(1.0, days))

def api_index(precip: List[float], half_life: float = 8.0) -> float:
    k = half_life_coeff(half_life)
    api = 0.0
    for p in precip:
        api = k * api + (p or 0.0)
    return api

def temperature_fit(tmin: float, tmax: float, tmean: float) -> float:
    if tmin < 1 or tmax > 32: return 0.0
    if tmean <= 6: base = max(0.0, (tmean)/6.0*0.3)
    elif tmean <= 10: base = 0.3 + 0.2*((tmean-6)/4.0)
    elif tmean <= 18: base = 0.5 + 0.5*((tmean-10)/8.0)
    elif tmean <= 22: base = 0.8 - 0.2*((tmean-18)/4.0)
    else: base = max(0.0, 0.6 - 0.6*((tmean-22)/10.0))
    if tmin < 6:  base *= max(0.3, 1 - (6 - tmin)/8.0)
    if tmax > 24: base *= max(0.3, 1 - (tmax - 24)/10.0)
    return max(0.0, min(1.0, base))

def final_index(api_val: float, et0_7: float, t_fit: float) -> int:
    moisture = max(0.0, min(1.0, (api_val - 0.6*et0_7)/40.0 + 0.6))
    return int(round(max(0.0, min(100.0, 100.0*(0.6*moisture + 0.4*t_fit)))))

def best_window(values: List[int]) -> Tuple[int,int,int]:
    if len(values) < 3: return (0,0,0)
    best = (0,2, round((values[0]+values[1]+values[2])/3))
    for i in range(1, len(values)-2):
        m = round((values[i]+values[i+1]+values[i+2])/3)
        if m > best[2]: best = (i, i+2, m)
    return best

def reliability_from_sources(ow_vals: List[float], om_vals: List[float]) -> float:
    # Concordanza tra modelli: più concordano, più alta l’affidabilità
    n = min(len(ow_vals), len(om_vals))
    if n == 0: return 0.55
    diffs = [abs((ow_vals[i] or 0.0) - (om_vals[i] or 0.0)) for i in range(n)]
    avg_diff = sum(diffs)/n
    return max(0.25, min(0.95, 0.95/(1.0 + avg_diff/6.0)))

def yield_estimate(idx: int) -> str:
    if idx >= 80: return "6–10+ porcini"
    if idx >= 60: return "2–5 porcini"
    if idx >= 40: return "1–2 porcini"
    return "0–1 porcini"

def slope_aspect_from_grid(z: List[List[float]], cell_size_m: float=30.0)->Tuple[float,float,str]:
    dzdx = ((z[0][2]+2*z[1][2]+z[2][2]) - (z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_size_m)
    dzdy = ((z[2][0]+2*z[2][1]+z[2][2]) - (z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_size_m)
    slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect = (math.degrees(math.atan2(dzdx, dzdy))+360)%360
    octs = ["N","NE","E","SE","S","SW","W","NW","N"]
    return round(slope,1), round(aspect,0), octs[int(((aspect%360)+22.5)//45)]

# ----------------- Providers (con fallback) -----------------
async def fetch_open_meteo(lat: float, lon: float, past: int=15, future: int=10) -> Dict[str,Any]:
    url="https://api.open-meteo.com/v1/forecast"
    params={
        "latitude":lat, "longitude":lon, "timezone":"auto",
        "daily": ",".join([
            "precipitation_sum","temperature_2m_mean",
            "temperature_2m_min","temperature_2m_max",
            "et0_fao_evapotranspiration","relative_humidity_2m_mean"
        ]),
        "past_days": past, "forecast_days": future
    }
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
        r=await c.get(url, params=params); r.raise_for_status()
        return r.json()

async def fetch_openweather(lat: float, lon: float) -> Dict[str,Any]:
    if not OWM_KEY: return {}
    url="https://api.openweathermap.org/data/3.0/onecall"
    params={
        "lat":lat, "lon":lon, "exclude":"minutely,hourly,current,alerts",
        "units":"metric", "lang":"it", "appid":OWM_KEY
    }
    try:
        async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
            r=await c.get(url, params=params); r.raise_for_status()
            return r.json()
    except Exception:
        # fallback silenzioso: nessun dato OW
        return {}

async def fetch_elevation_grid(lat: float, lon: float, step_m: float=30.0)->Optional[List[List[float]]]:
    try:
        deg_lat = 1/111320.0
        deg_lon = 1/(111320.0*math.cos(math.radians(lat)))
        dlat, dlon = step_m*deg_lat, step_m*deg_lon
        coords=[{"latitude":lat+dr*dlat,"longitude":lon+dc*dlon} for dr in(-1,0,1) for dc in(-1,0,1)]
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
            r=await c.post("https://api.open-elevation.com/api/v1/lookup", json={"locations":coords})
            r.raise_for_status(); j=r.json()
        vals=[p["elevation"] for p in j["results"]]
        return [vals[0:3], vals[3:6], vals[6:9]]
    except Exception:
        return None  # fallback: niente pendenza/esposizione

async def geocode_nominatim(q: str)->Dict[str,Any]:
    url="https://nominatim.openstreetmap.org/search"
    params={"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r=await c.get(url, params=params); r.raise_for_status()
        data=r.json()
    if not data: raise HTTPException(404, "Località non trovata")
    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]), "display": data[0].get("display_name","")}

# ----------------- Endpoints -----------------
@app.get("/api/health")
async def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode_nominatim(q)

@app.get("/api/score")
async def api_score(
    lat: float = Query(...),
    lon: float = Query(...),
    half: float = Query(8.0, gt=3.0, lt=20.0)
):
    # 1) acquisizione dati in parallelo (con fallback silenziosi)
    om_task  = asyncio.create_task(fetch_open_meteo(lat,lon,past=15,future=10))
    ow_task  = asyncio.create_task(fetch_openweather(lat,lon))
    dem_task = asyncio.create_task(fetch_elevation_grid(lat,lon))
    om, ow, elev_grid = await asyncio.gather(om_task, ow_task, dem_task)

    # 2) parsing Open-Meteo (storico + futuro)
    d = om["daily"]
    timev = d["time"]
    P_om  = [float(x or 0.0) for x in d["precipitation_sum"]]
    Tm_om = d["temperature_2m_mean"]
    Tmin_om = d["temperature_2m_min"]
    Tmax_om = d["temperature_2m_max"]
    ET0_om  = d.get("et0_fao_evapotranspiration", [0.0]*len(P_om))
    RH_om   = d.get("relative_humidity_2m_mean", [None]*len(P_om))

    pastN = 15
    futN  = 10
    P_past_om = P_om[:pastN]
    P_fut_om  = P_om[pastN:pastN+futN]
    Tmin_p_om, Tmax_p_om, Tm_p_om = Tmin_om[:pastN], Tmax_om[:pastN], Tm_om[:pastN]
    Tmin_f_om, Tmax_f_om, Tm_f_om = Tmin_om[pastN:pastN+futN], Tmax_om[pastN:pastN+futN], Tm_om[pastN:pastN+futN]
    ET0_p_om = ET0_om[:pastN]; ET0_f_om = ET0_om[pastN:pastN+futN]

    # 3) parsing OpenWeather (se disponibile)
    P_fut_ow: List[float] = []
    Tmin_f_ow: List[float] = []
    Tmax_f_ow: List[float] = []
    Tm_f_ow:   List[float] = []
    if ow and "daily" in ow:
        for day in ow["daily"]:
            P_fut_ow.append(float(day.get("rain", 0.0)))
            t = day.get("temp", {})
            Tmin_f_ow.append(float(t.get("min", 0.0)))
            Tmax_f_ow.append(float(t.get("max", 0.0)))
            Tm_f_ow.append(float(t.get("day", (t.get("min",0.0)+t.get("max",0.0))/2.0)))
    ow_len = min(len(P_fut_ow), futN)

    # 4) BLENDING previsioni (OW 60% + OM 40% dove disponibile; altrimenti OM)
    w_ow, w_om = 0.60, 0.40
    def blend(arr_ow, arr_om, i):
        if i < ow_len:
            return w_ow*arr_ow[i] + w_om*arr_om[i]
        return arr_om[i]

    P_fut_blend, Tmin_f_blend, Tmax_f_blend, Tm_f_blend = [], [], [], []
    for i in range(futN):
        P_fut_blend.append(blend(P_fut_ow, P_fut_om, i))
        Tmin_f_blend.append(blend(Tmin_f_ow, Tmin_f_om, i) if ow_len else Tmin_f_om[i])
        Tmax_f_blend.append(blend(Tmax_f_ow, Tmax_f_om, i) if ow_len else Tmax_f_om[i])
        Tm_f_blend.append(  blend(Tm_f_ow,   Tm_f_om,   i) if ow_len else Tm_f_om[i])

    # 5) Orografia (fallback se DEM non disponibile)
    if elev_grid:
        elev_m = float(elev_grid[1][1])
        slope_deg, aspect_deg, aspect_oct = slope_aspect_from_grid(elev_grid)
    else:
        elev_m, slope_deg, aspect_deg, aspect_oct = 800.0, 5.0, 0.0, "N"  # valori neutri

    # 6) Indice oggi (storico OM)
    API_val = api_index(P_past_om, half_life=half)
    ET7     = sum(ET0_p_om[-7:]) if ET0_p_om else 0.0
    Tfit_today = temperature_fit(Tmin_p_om[-1], Tmax_p_om[-1], Tm_p_om[-1])
    idx_today  = final_index(API_val, ET7, Tfit_today)

    # 7) Previsione 10gg
    scores=[]
    k = half_life_coeff(half); rolling = API_val
    for i in range(futN):
        rolling = k*rolling + (P_fut_blend[i] or 0.0)
        et7f = sum(ET0_f_om[max(0,i-6):i+1]) if ET0_f_om else 0.0
        tfit = temperature_fit(Tmin_f_blend[i], Tmax_f_blend[i], Tm_f_blend[i])
        scores.append(final_index(rolling, et7f, tfit))
    s,e,m = best_window(scores)

    # 8) Affidabilità
    reliability = reliability_from_sources(P_fut_ow[:ow_len], P_fut_om[:ow_len]) if ow_len else 0.6

    # 9) Tabelle piogge
    rain_past = { timev[i]: round(P_past_om[i],1) for i in range(min(pastN,len(timev))) }
    rain_future = { timev[pastN+i] if pastN+i < len(timev) else f"+{i+1}d": round(P_fut_blend[i],1) for i in range(futN) }

    # 10) Spiegazioni/consigli
    P7 = sum(P_past_om[-7:]); P15 = sum(P_past_om[-15:])
    Tm7 = sum(Tm_p_om[-7:])/max(1,len(Tm_p_om[-7:]))
    reasons=[]
    if P15 < 20: reasons.append(f"Piogge scarse ultimi 15 giorni ({P15:.0f} mm).")
    if ET7  > 30: reasons.append(f"ET0 elevata su 7 giorni ({ET7:.0f} mm).")
    if Tm7 < 10: reasons.append(f"Temperature medie basse (Tmed7 {Tm7:.1f} °C).")
    if Tm7 > 18: reasons.append(f"Temperature medie alte (Tmed7 {Tm7:.1f} °C).")
    if not reasons: reasons.append("Bilancio idrico e termico favorevoli.")
    tips=[]
    if idx_today >= 80: tips.append("Finestra ottima: versanti N–NE, conche umide, lettiera spessa.")
    elif idx_today >= 60: tips.append("Discreto: fondovalle e margini ombreggiati; suoli profondi.")
    elif idx_today >= 40: tips.append("Moderato: attendi piogge ≥10–15 mm e 2–3 giorni di maturazione.")
    else: tips.append("Basso potenziale: meglio aspettare precipitazioni significative.")
    if reliability < 0.6: tips.append("Affidabilità bassa: modelli in disaccordo; ricontrolla tra 12–24 h.")

    return {
        "lat": lat, "lon": lon,
        "elevation_m": round(elev_m),
        "slope_deg": slope_deg, "aspect_deg": aspect_deg, "aspect_octant": aspect_oct,
        "API_star_mm": round(API_val,1),
        "P7_mm": round(P7,1), "P15_mm": round(P15,1),
        "ET0_7d_mm": round(ET7,1),
        "Tmean7_c": round(Tm7,1),
        "index": idx_today,
        "forecast": [int(x) for x in scores],
        "best_window": {"start": s, "end": e, "mean": m},
        "harvest_estimate": yield_estimate(idx_today),
        "reliability": round(reliability,3),
        "rain_past": rain_past,
        "rain_future": rain_future,
        "explanation": {"reasons": reasons},
        "tips": tips
    }


