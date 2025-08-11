# main.py (advanced) — Backend FastAPI per previsione porcini
# — Fix Render: rimossa dipendenza da `python-dateutil` (ModuleNotFoundError)
# — Modello: SMI (bilancio P-PET) + VPD/UR + cold-shock + microtopografia (TWI proxy)
# — Blend: Open‑Meteo (free) + OpenWeather (free con chiave)
# — Output compatibile con la tua UI (stesse chiavi principali)
#
# Dipendenze: fastapi, uvicorn, httpx, numpy, pydantic
# Avvio locale: uvicorn main:app --reload --host 0.0.0.0 --port 8000

from __future__ import annotations
import os
import math
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# —— Config base ——
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
NOMINATIM_EMAIL = os.getenv("NOMINATIM_EMAIL", "info@example.com")

app = FastAPI(title="Porcini Forecast API — Advanced Model")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chiudi in produzione
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# —— Utility tempo ——
UTC = timezone.utc
TODAY = lambda: datetime.now(tz=UTC).date()

# —— Helper meteo ——
async def fetch_open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    """
    Scarica dati orari/daily Open‑Meteo per finestra: 21 giorni passati + 10 giorni futuri.
    Variabili: T2m, UR, precipitazione, shortwave_rad, Tmin/Tmax.
    """
    client_timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=client_timeout) as client:
        base = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "shortwave_radiation",
                "surface_pressure",
            ],
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
            ],
            "past_days": 21,
            "forecast_days": 10,
            "timezone": "UTC",
        }
        r = await client.get(base, params=params)
        r.raise_for_status()
        data = r.json()
        return data

async def fetch_openweather(lat: float, lon: float) -> Dict[str, Any]:
    """Usa endpoint forecast 3h/5gg + current di OpenWeather, se è presente la chiave."""
    if not OPENWEATHER_API_KEY:
        return {"ok": False, "reason": "missing_key"}

    client_timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=client_timeout) as client:
        try:
            r_now = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            )
            r_now.raise_for_status()
            now = r_now.json()
        except Exception as e:
            return {"ok": False, "reason": f"current_failed: {e}"}
        try:
            r_fc = await client.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            )
            r_fc.raise_for_status()
            fc = r_fc.json()
        except Exception as e:
            return {"ok": False, "reason": f"forecast_failed: {e}"}

    return {"ok": True, "now": now, "forecast": fc}

# —— Geocoding (per compatibilità con UI) ——
async def geocode_nominatim(q: str) -> List[Dict[str, Any]]:
    ua = {
        "User-Agent": f"porcini-app/1.0 ({NOMINATIM_EMAIL})",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=15.0, headers=ua) as client:
        r = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "addressdetails": 1, "limit": 5},
        )
        r.raise_for_status()
        return r.json()

async def reverse_nominatim(lat: float, lon: float) -> Dict[str, Any]:
    ua = {
        "User-Agent": f"porcini-app/1.0 ({NOMINATIM_EMAIL})",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=15.0, headers=ua) as client:
        r = await client.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 14, "addressdetails": 1},
        )
        r.raise_for_status()
        return r.json()

# —— Elevation & TWI proxy ——
async def fetch_elevation_patch(lat: float, lon: float, ddeg: float = 0.0009) -> Tuple[np.ndarray, float, float]:
    """
    Chiede un 3x3 a Open‑Elevation attorno al punto, ritorna griglia elevazioni e risoluzione approssimata (m/px).
    ddeg ~ 0.0009 ~ 100 m circa (dipende dalla latitudine).
    """
    lats = [lat - ddeg, lat, lat + ddeg]
    lons = [lon - ddeg, lon, lon + ddeg]
    coords = [{"latitude": la, "longitude": lo} for la in lats for lo in lons]
    payload = {"locations": coords}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post("https://api.open-elevation.com/api/v1/lookup", json=payload)
        if r.status_code != 200:
            arr = np.zeros((3, 3), dtype=float)
            return arr, 0.0, 0.0
        js = r.json().get("results", [])
        if len(js) != 9:
            arr = np.zeros((3, 3), dtype=float)
            return arr, 0.0, 0.0
        zs = [it["elevation"] for it in js]
        grid = np.array(zs, dtype=float).reshape(3, 3)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
        dx = ddeg * m_per_deg_lon
        dy = ddeg * m_per_deg_lat
        return grid, dx, dy

def slope_aspect_from_grid(grid: np.ndarray, dx: float, dy: float) -> Tuple[float, float]:
    """Calcola pendenza (°) e aspect (gradi azimut) con kernel Horn su 3x3."""
    if grid.shape != (3, 3) or dx == 0.0 or dy == 0.0:
        return 0.0, float("nan")
    z = grid
    dzdx = ((z[0, 2] + 2*z[1, 2] + z[2, 2]) - (z[0, 0] + 2*z[1, 0] + z[2, 0])) / (8.0 * dx)
    dzdy = ((z[2, 0] + 2*z[2, 1] + z[2, 2]) - (z[0, 0] + 2*z[0, 1] + z[0, 2])) / (8.0 * dy)
    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)
    aspect_rad = math.atan2(dzdy, -dzdx)  # Horn convention → 0=N
    aspect_deg = (math.degrees(aspect_rad) + 180.0) % 360.0
    if slope_deg < 0.8:
        return 0.0, float("nan")
    return slope_deg, aspect_deg

def concavity_proxy(grid: np.ndarray) -> float:
    if grid.shape != (3, 3):
        return 0.0
    center = grid[1, 1]
    neigh = np.delete(grid.flatten(), 4)
    val = float(np.mean(neigh) - center)
    return max(-1.0, min(1.0, val / 10.0))

def twi_proxy_from_slope_concavity(slope_deg: float, concav: float) -> float:
    beta = math.radians(max(0.1, slope_deg))
    tanb = math.tan(beta)
    c = max(0.0, concav + 0.2)
    twi = math.log(1.0 + 4.0 * c) - math.log(max(0.05, tanb))
    return max(0.0, min(1.0, (twi + 2.0) / 4.0))

# —— Fisica semplificata ——
def saturation_vapor_pressure_hpa(Tc: float) -> float:
    return 6.112 * math.exp((17.67 * Tc) / (Tc + 243.5))

def vpd_hpa(Tc: float, RH: float) -> float:
    es = saturation_vapor_pressure_hpa(Tc)
    return es * (1.0 - max(0.0, min(100.0, RH)) / 100.0)

# —— PET (Hargreaves-Samani) ——
def extraterrestrial_radiation_MJm2d(lat_deg: float, doy: int) -> float:
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    ws = math.acos(-math.tan(phi) * math.tan(delta))
    Gsc = 0.0820  # MJ m^-2 min^-1
    Ra = (24*60/math.pi) * Gsc * dr * (
        ws*math.sin(phi)*math.sin(delta) + math.cos(phi)*math.cos(delta)*math.sin(ws)
    )
    return max(0.0, Ra)

def pet_hargreaves(Tmin: float, Tmax: float, Tmean: float, lat_deg: float, doy: int) -> float:
    Ra = extraterrestrial_radiation_MJm2d(lat_deg, doy)
    return max(0.0, 0.0023 * Ra * max(0.0, Tmean + 17.8) * math.sqrt(max(0.0, Tmax - Tmin)))

# —— SMI (Soil Moisture Index) ——
def compute_smi_series(dates: List[datetime], precip_mm: List[float], tmin: List[float], tmax: List[float], tmean: List[float], lat: float) -> List[float]:
    alpha = 0.18  # memoria idrica ~10 gg
    S = 0.0
    smi_raw = []
    for i, d in enumerate(dates):
        doy = d.timetuple().tm_yday
        pet = pet_hargreaves(tmin[i], tmax[i], tmean[i], lat, doy)
        forcing = precip_mm[i] - pet
        S = (1 - alpha) * S + alpha * forcing
        smi_raw.append(S)
    arr = np.array(smi_raw)
    if len(arr) >= 20:
        p5, p95 = np.percentile(arr, [5, 95])
    else:
        p5, p95 = float(arr.min()), float(arr.max() if arr.max()!=arr.min() else arr.min()+1)
    norm = (arr - p5) / max(1e-6, (p95 - p5))
    norm = np.clip(norm, 0.0, 1.0)
    return norm.tolist()

# —— Cold-shock ——
def cold_shock_score(tmin_series: List[float]) -> float:
    if len(tmin_series) < 7:
        return 0.0
    last3 = float(np.nanmean(tmin_series[-3:]))
    prev3 = float(np.nanmean(tmin_series[-6:-3]))
    drop = last3 - prev3
    if drop >= -1.0:
        return 0.0
    return max(0.0, min(1.0, (-drop - 1.0) / 3.0))

# —— VPD penalty/gating ——
def vpd_penalty(vpd_hpa_daily_max: float) -> float:
    if vpd_hpa_daily_max <= 6.0:
        return 1.0
    if vpd_hpa_daily_max >= 12.0:
        return 0.4
    return 1.0 - 0.6 * (vpd_hpa_daily_max - 6.0) / 6.0

# —— Lag stocastico dipendente da SMI & cold-shock ——
def stochastic_lag_days(smi: float, shock: float) -> float:
    base_mean = 12.0 - 5.0 * smi - 2.0 * shock
    base_mean = min(16.0, max(7.0, base_mean))
    return base_mean

# —— Energy index da aspect/pendenza e stagione ——
def aspect_energy_index(aspect_deg: float, slope_deg: float, month: int) -> float:
    if math.isnan(aspect_deg) or slope_deg < 0.8:
        return 0.5
    summer_bias = 1.0 if month in (7,8,9) else 0.5
    def angdiff(a,b):
        d = abs(a-b) % 360.0
        return d if d<=180 else 360-d
    dN = angdiff(aspect_deg, 0.0)
    dS = angdiff(aspect_deg, 180.0)
    bonusN = max(0.0, 1.0 - dN/180.0)
    malusS = max(0.0, 1.0 - dS/180.0) * summer_bias
    e = 0.5 + 0.3*bonusN - 0.3*malusS
    e *= 1.0 + min(0.2, slope_deg/90.0)
    return float(np.clip(e, 0.2, 0.9))

# —— Generazione indice giornaliero (10 giorni) ——
def gaussian(x, mu, sigma):
    return math.exp(-0.5*((x-mu)/sigma)**2)

def build_daily_index(dates: List[datetime], smi: List[float], tmin: List[float], tmax: List[float], tmean: List[float],
                      vpd_max: List[float], energy_idx: float, shock_score: float) -> List[float]:
    n = len(dates)
    idx = np.zeros(n, dtype=float)
    thr = float(np.percentile(smi, 60)) if n>=10 else 0.6
    sigma = 2.5
    for i, _ in enumerate(dates):
        if smi[i] >= thr:
            lag = stochastic_lag_days(smi[i], shock_score)
            peak_day = i + int(round(lag))
            for j in range(i, min(n, peak_day+8)):
                amp = 0.6 + 0.7*smi[i] + 0.3*shock_score
                pen = vpd_penalty(vpd_max[j])
                idx[j] += amp * gaussian(j, peak_day, sigma) * pen
    idx *= energy_idx
    if idx.max() > 0:
        idx = 100.0 * idx / idx.max()
    return idx.tolist()

# —— Taglia attesa (diametro cappello) ——
def cap_growth_rate_cm_per_day(Tmean: float) -> float:
    mu, sig = 15.0, 4.5
    base = math.exp(-0.5*((Tmean-mu)/sig)**2)
    return 1.8 * base

def estimate_cap_size_today(recent_T: List[float], recent_URmin: List[float]) -> Dict[str, float]:
    days = min(4, len(recent_T))
    size = 0.0
    halted = False
    for i in range(days):
        if recent_URmin[i] < 40.0:
            halted = True
            continue
        size += cap_growth_rate_cm_per_day(recent_T[i])
    return {
        "mean_cm": round(size, 1),
        "range_cm": [round(max(1.0, 0.7*size),1), round(max(1.5, 1.3*size),1)],
        "growth_halted": halted,
    }

# —— Mapping indice → raccolto atteso ——
def expected_harvest_from_index(index_today: float, hours: int, ndvi_scaler: float = 1.0) -> Dict[str, Any]:
    if index_today >= 85:
        base = (8, 14)
    elif index_today >= 70:
        base = (5, 9)
    elif index_today >= 55:
        base = (3, 6)
    elif index_today >= 40:
        base = (1, 3)
    else:
        base = (0, 2)
    factor = hours/2.0
    lo = max(0, int(round(base[0]*factor*ndvi_scaler)))
    hi = max(lo, int(round(base[1]*factor*ndvi_scaler)))
    return {"min": lo, "max": hi}

# —— Affidabilità ——
def reliability_score(source_agreement: float, coverage_ok: bool, had_keys: bool) -> float:
    base = 0.55
    base += 0.25*source_agreement
    if coverage_ok:
        base += 0.1
    if had_keys:
        base += 0.05
    return float(np.clip(base, 0.2, 0.95))

# —— Schemi risposta ——
class PredictResponse(BaseModel):
    index_today: float
    expected_harvest: Dict[str, Any]
    size_today: Dict[str, Any]
    forecast_10d: List[Dict[str, Any]]
    rain_table: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    reliability: float

@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.now(tz=UTC).isoformat()}

@app.get("/geocode")
async def geocode(q: str = Query(..., min_length=2)):
    res = await geocode_nominatim(q)
    return res

@app.get("/reverse")
async def reverse(lat: float, lon: float):
    return await reverse_nominatim(lat, lon)

@app.get("/predict", response_model=PredictResponse)
async def predict(
    lat: float,
    lon: float,
    hours: int = 2,
):
    om_task = asyncio.create_task(fetch_open_meteo(lat, lon))
    ow_task = asyncio.create_task(fetch_openweather(lat, lon))
    elev_task = asyncio.create_task(fetch_elevation_patch(lat, lon))

    om = await om_task
    ow = await ow_task
    grid, dx, dy = await elev_task

    hourly = om.get("hourly", {})
    htimes = [datetime.fromisoformat(t.replace("Z","+00:00")) for t in hourly.get("time", [])]
    T = hourly.get("temperature_2m", [])
    RH = hourly.get("relative_humidity_2m", [])
    PR = hourly.get("precipitation", [])
    SW = hourly.get("shortwave_radiation", [])

    from collections import defaultdict
    by_day = defaultdict(lambda: {"T":[], "RH":[], "PR":[], "SW":[]})
    for t, tt, rh, pr, sw in zip(htimes, T, RH, PR, SW):
        d = datetime(t.year, t.month, t.day, tzinfo=UTC)
        by_day[d]["T"].append(tt)
        by_day[d]["RH"].append(rh)
        by_day[d]["PR"].append(pr)
        by_day[d]["SW"].append(sw)

    days_sorted = sorted(by_day.keys())
    tmin, tmax, tmean, prsum, urmin, vpdmax = [], [], [], [], [], []
    for d in days_sorted:
        Ts = np.array(by_day[d]["T"]) if by_day[d]["T"] else np.array([np.nan])
        RHs = np.array(by_day[d]["RH"]) if by_day[d]["RH"] else np.array([np.nan])
        PRs = np.array(by_day[d]["PR"]) if by_day[d]["PR"] else np.array([0.0])
        tmin.append(float(np.nanmin(Ts)))
        tmax.append(float(np.nanmax(Ts)))
        tmean.append(float(np.nanmean(Ts)))
        prsum.append(float(np.nansum(PRs)))
        # URmin e VPDmax (calcolo robusto; coppie T/RH valide)
        urmin.append(float(np.nanmin(RHs)) if np.any(~np.isnan(RHs)) else 60.0)
        vpds = []
        min_len = min(len(Ts), len(RHs))
        for i in range(min_len):
            if not (math.isnan(Ts[i]) or math.isnan(RHs[i])):
                vpds.append(vpd_hpa(float(Ts[i]), float(RHs[i])))
        vpdmax.append(float(np.nanmax(vpds)) if len(vpds) else 6.0)

    slope_deg, aspect_deg = slope_aspect_from_grid(grid, dx, dy)
    concav = concavity_proxy(grid)
    twi_px = twi_proxy_from_slope_concavity(slope_deg, concav)
    month_now = (TODAY()).month
    energy_idx = aspect_energy_index(aspect_deg, slope_deg, month_now)

    smi = compute_smi_series(days_sorted, prsum, tmin, tmax, tmean, lat)
    shock = cold_shock_score(tmin)

    today_dt = datetime(TODAY().year, TODAY().month, TODAY().day, tzinfo=UTC)
    try:
        i0 = days_sorted.index(today_dt)
    except ValueError:
        diffs = [abs((d - today_dt).days) for d in days_sorted]
        i0 = int(np.argmin(diffs))
    i1 = min(len(days_sorted), i0 + 10)

    idx_10d = build_daily_index(
        days_sorted[i0:i1],
        smi[i0:i1],
        tmin[i0:i1],
        tmax[i0:i1],
        tmean[i0:i1],
        vpdmax[i0:i1],
        energy_idx,
        shock,
    )

    index_today = float(idx_10d[0]) if idx_10d else 0.0

    recent_T = list(reversed(tmean[max(0, i0-3):i0+1]))
    recent_URmin = list(reversed(urmin[max(0, i0-3):i0+1]))
    size_today = estimate_cap_size_today(recent_T, recent_URmin)

    expected = expected_harvest_from_index(index_today, hours, ndvi_scaler=1.0)

    rain_table = []
    i2 = min(len(days_sorted), i0 + 10)
    start_rain = max(0, i0-10)
    for d, pr in zip(days_sorted[start_rain:i2], prsum[start_rain:i2]):
        rain_table.append({"date": d.date().isoformat(), "precip_mm": round(pr, 1)})

    had_keys = bool(OPENWEATHER_API_KEY)
    agreement = 0.5
    if isinstance(ow, dict) and ow.get("ok"):
        try:
            future_hours = [t for t in htimes if t >= today_dt and t < today_dt + timedelta(days=3)]
            if future_hours:
                idxs = [htimes.index(t) for t in future_hours]
                om_T = float(np.nanmean([T[i] for i in idxs]))
                om_PR = float(np.nansum([PR[i] for i in idxs]))
            else:
                om_T, om_PR = float("nan"), float("nan")
            fcl = ow["forecast"].get("list", [])
            ow_T = float(np.nanmean([it["main"]["temp"] for it in fcl[:24]])) if fcl else float("nan")
            rain_vals = [it.get("rain", {}).get("3h", 0.0) for it in fcl[:24]] if fcl else []
            ow_PR = float(np.nansum(rain_vals)) if rain_vals else float("nan")
            aT = 0.5 if math.isnan(om_T) or math.isnan(ow_T) else max(0.0, 1.0 - abs(om_T - ow_T)/6.0)
            aP = 0.5 if math.isnan(om_PR) or math.isnan(ow_PR) else max(0.0, 1.0 - abs(om_PR - ow_PR)/20.0)
            agreement = 0.5 * (aT + aP)
        except Exception:
            agreement = 0.5

    coverage_ok = (len(days_sorted) >= 25)
    reliab = reliability_score(agreement, coverage_ok, had_keys)

    forecast_10d = []
    for j, d in enumerate(days_sorted[i0:i1]):
        forecast_10d.append({
            "date": d.date().isoformat(),
            "index": round(idx_10d[j], 1),
        })

    aspect_str = None
    if not math.isnan(aspect_deg):
        dirs = ["N","NE","E","SE","S","SW","W","NW"]
        k = int(((aspect_deg + 22.5) % 360) / 45)
        aspect_str = dirs[k]
    analysis = {
        "microtopography": {
            "slope_deg": round(slope_deg, 1),
            "aspect_deg": None if math.isnan(aspect_deg) else round(aspect_deg, 1),
            "aspect_octant": aspect_str,
            "concavity_proxy": round(concav, 2),
            "twi_proxy": round(twi_px, 2),
            "energy_index": round(energy_idx, 2),
        },
        "drivers": {
            "smi_today": round(smi[i0], 2) if i0 < len(smi) else None,
            "cold_shock": round(shock, 2),
            "vpd_max_today_hPa": round(vpdmax[i0], 1) if i0 < len(vpdmax) else None,
            "urmin_today_%": round(urmin[i0], 1) if i0 < len(urmin) else None,
        },
        "sources": {
            "open_meteo": True,
            "open_weather": bool(isinstance(ow, dict) and ow.get("ok")),
            "nominatim_email": NOMINATIM_EMAIL,
        }
    }

    resp = PredictResponse(
        index_today=round(index_today, 1),
        expected_harvest=expected,
        size_today=size_today,
        forecast_10d=forecast_10d,
        rain_table=rain_table,
        analysis=analysis,
        reliability=round(reliab, 2),
    )
    return resp


       
