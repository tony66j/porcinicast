# main.py — Trova Porcini API (v2.3.0)
# Novità v2.3:
#  • "Raccolto atteso" SEMPRE mostrato (stima prudenziale, ponderata da reliability/SMI/VPD/evento osservato).
#  • Profilo SPECIE per il gruppo "porcini" (B. aereus, B. reticulatus, B. edulis, B. pinophilus):
#      - selezione specie in base a habitat (OSM/manuale), mese, quota e (lievemente) esposizione;
#      - per-specie: Tm7 ottimale, finestra stagionale, lag base, sensibilità a VPD e a SMI.
#  • Tutto il resto (blend OM+OW passato/futuro, SMI P-ET0 o ERA5-Land, VPD, TWI, energy index) resta compatibile.
#
# Avvio: uvicorn main:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio, tempfile, time
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta

app = FastAPI(title="Trova Porcini API (v2.3.0)", version="2.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

HEADERS = {"User-Agent":"Trovaporcini/2.3.0 (+site)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")
CDS_API_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")

# ----------------------------- UTIL -----------------------------
def clamp(v,a,b): return a if v<a else b if v>b else v

def half_life_coeff(days: float) -> float:
    return 1.0 - 0.5**(1.0/max(1.0,days))

def api_index(precip: List[float], half_life: float=8.0) -> float:
    k=half_life_coeff(half_life); api=0.0
    for p in precip: api=(1-k)*api + k*(p or 0.0)
    return api

def stddev(xs: List[float]) -> float:
    if not xs: return 0.0
    m=sum(xs)/len(xs)
    return (sum((x-m)**2 for x in xs)/len(xs))**0.5

# ---- VPD ----
def saturation_vapor_pressure_hpa(Tc: float) -> float:
    return 6.112 * math.exp((17.67 * Tc) / (Tc + 243.5))

def vpd_hpa(Tc: float, RH: float) -> float:
    RHc = clamp(RH, 0.0, 100.0)
    return saturation_vapor_pressure_hpa(Tc) * (1.0 - RHc/100.0)

def vpd_penalty(vpd_max_hpa: float, species_vpd_sens: float = 1.0) -> float:
    """
    Penalità su 0.4..1.0; species_vpd_sens: 0.8 (tollerante) .. 1.2 (sensibile)
    """
    if vpd_max_hpa <= 6.0: base = 1.0
    elif vpd_max_hpa >= 12.0: base = 0.4
    else: base = 1.0 - 0.6 * (vpd_max_hpa - 6.0) / 6.0
    # specie più sensibile → amplifica malus
    return clamp(1.0 - (1.0-base)*species_vpd_sens, 0.35, 1.0)

# ---- Shock termico ΔTmin(3d) ----
def cold_shock_from_tmin_series(tmin: List[float]) -> float:
    if len(tmin) < 7: return 0.0
    last3 = sum(tmin[-3:]) / 3.0
    prev3 = sum(tmin[-6:-3]) / 3.0
    drop = last3 - prev3
    if drop >= -1.0: return 0.0
    return clamp((-drop - 1.0) / 3.0, 0.0, 1.0)  # 0..1

# ---- TWI proxy + energy index ----
def twi_proxy_from_slope_concavity(slope_deg: float, concavity: float) -> float:
    beta = max(0.1, math.radians(max(0.1, slope_deg)))
    tanb = max(0.05, math.tan(beta))
    conc = max(0.0, concavity + 0.02)
    twi = math.log(1.0 + conc / tanb)
    return clamp((twi - 0.5) / (3.0 - 0.5), 0.0, 1.0)

def microclimate_energy(aspect_octant: str, slope_deg: float, month: int) -> float:
    # molto semplice: N/NE più fresco/umido, S/SW più caldo/secco; modulato da mese
    aspect_octant = (aspect_octant or "NA").upper()
    base = {"N":0.4,"NE":0.5,"E":0.6,"SE":0.7,"S":0.8,"SW":0.8,"W":0.6,"NW":0.5}.get(aspect_octant,0.6)
    seasonal = 0.6 if month in (11,12,1,2) else 0.7 if month in (9,10,3,4) else 0.8  # estate più "energetica"
    e = base * (0.8 + 0.4*min(1.0, slope_deg/30.0)) * seasonal
    return clamp(e, 0.3, 1.1)

# ------------------------ FONTI METEO ------------------------
async def fetch_openmeteo(lat: float, lon: float, past: int = 15, future: int = 10) -> Dict[str, Any]:
    """
    Open-Meteo: passato (ri-analisi) + previsioni.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,shortwave_radiation,et0_fao_evapotranspiration",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "past_days": past,
        "forecast_days": min(16, future),
        "timezone": "UTC",
        "models": "best_match"
    }
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        j = r.json()

    h = j.get("hourly", {})
    timev = h.get("time", [])
    T = h.get("temperature_2m", [])
    RH = h.get("relative_humidity_2m", [])
    P = h.get("precipitation", [])
    SW = h.get("shortwave_radiation", [])
    ET0 = h.get("et0_fao_evapotranspiration", [])
    return {"time": timev, "T": T, "RH": RH, "P": P, "SW": SW, "ET0": ET0}

async def fetch_openweather(lat: float, lon: float) -> Dict[str, Any]:
    """
    OpenWeather: passato recente (onecall/timemachine-like proxy) + 7-10gg.
    """
    if not OWM_KEY:
        return {"current": {}, "daily": []}
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric", "lang": "it"}
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    # normalizza
    daily = j.get("daily", [])
    pastP = []
    futP = []
    futT = []
    futRH = []
    now = int(time.time())
    for d in daily:
        dt = d.get("dt", now)
        mm = float(d.get("rain", 0.0) or 0.0)
        tmean = (float(d.get("temp", {}).get("min", 0.0)) + float(d.get("temp", {}).get("max", 0.0))) / 2.0
        rh = float(d.get("humidity", 60.0))
        if dt < now:
            pastP.append(mm)
        else:
            futP.append(mm)
            futT.append(tmean)
            futRH.append(rh)
    return {"pastP": pastP[-15:], "futP": futP[:10], "futT": futT[:10], "futRH": futRH[:10]}

# --------------------- DEM / OROGRAFIA ---------------------
async def fetch_elevation_grid_multiscale(lat: float, lon: float) -> Tuple[float, float, float, str, float]:
    """
    Placeholder DEM: restituisce quota, pendenza, aspetto (gradi e ottante), concavità.
    In produzione: SRTM/ALOS/Mapzen + derivati.
    """
    # Semplificazione: valori fittizi coerenti col punto
    elev = 1000 + 500 * math.sin(math.radians(lat%90))
    slope = 12 + 8 * math.cos(math.radians(lon%90))
    aspect_deg = (int(abs(lat*lon)) % 360)
    aspect_oct = ["N","NE","E","SE","S","SW","W","NW"][int((aspect_deg%360)/45)]
    concavity = 0.02 * math.sin(math.radians((lat+lon)%180))
    return round(elev), round(slope,1), aspect_deg, aspect_oct, round(concavity,3)

# --------------------- OSM / HABITAT ----------------------
async def fetch_osm_habitat(lat: float, lon: float) -> Tuple[str, float, Dict[str,float]]:
    """
    Query OSM per inferire macro-habitat (faggio/castagno/quercia/conifere/misto).
    """
    # qui useremmo Overpass; per ora un semplice stub deterministico
    def _score_tags(tags: Dict[str, Any]) -> Dict[str, float]:
        # stub
        return {"faggio":0.6, "castagno":0.1, "quercia":0.1, "conifere":0.1, "misto":0.1}
    # finto giro su 3 tile nell'intorno
    scores = {"faggio":0,"castagno":0,"quercia":0,"conifere":0,"misto":0}
    for _ in range(3):
        local = _score_tags({})
        for k,v in local.items(): scores[k]+=v
    # normalizza
    for k in scores: scores[k]/=3.0
    # sceglie massimo
    habitat, conf = max(scores.items(), key=lambda kv: kv[1])[0], max(scores.values())
    return habitat, conf, scores

def _choose_habitat(scores: Dict[str,float]) -> Tuple[str, float]:
    if not scores: return "misto", 0.0
    k = max(scores, key=lambda x: scores[x])
    return k, scores[k]

async def try_habitat_auto(lat: float, lon: float) -> Tuple[str, float, Dict[str,float]]:
    # wrapper robusto
    try:
        for _ in range(2):
            # in produzione: query Overpass con bbox progressiva
            # qui: usa stub
            scores = {"faggio":0.55, "castagno":0.15, "quercia":0.1, "conifere":0.1, "misto":0.1}
            hab, conf = _choose_habitat(scores)
            return hab, conf, scores
        # fallback
        return "misto", 0.15, {"castagno":0,"faggio":0,"quercia":0,"conifere":0,"misto":1}
    except Exception:
        return "misto", 0.15, {"castagno":0,"faggio":0,"quercia":0,"conifere":0,"misto":1}

# ----------------------- SMI (P-ET0 + ERA5-Land opzionale) -----------------------
SM_CACHE: Dict[str, Dict[str, Any]] = {}

async def _prefetch_era5l_sm(lat: float, lon: float, days: int = 40) -> None:
    if not CDS_API_KEY: return
    key = f"{round(lat,3)},{round(lon,3)}"
    if key in SM_CACHE and (time.time() - SM_CACHE[key].get("ts", 0)) < 12*3600:
        return
    try:
        import cdsapi  # type: ignore
        from netCDF4 import Dataset, num2date  # type: ignore
    except Exception:
        return
    def _blocking_download():
        try:
            c = cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY, quiet=True, verify=1)
            end = datetime.utcnow().date()
            start = end - timedelta(days=days-1)
            years = sorted({start.year, end.year})
            months = [f"{m:02d}" for m in range(1,13)] if len(years)>1 else [f"{m:02d}" for m in range(start.month, end.month+1)]
            days_list = [f"{d:02d}" for d in range(1,31)]
            bbox = [lat+0.05, lon-0.05, lat-0.05, lon+0.05]
            req = {
                "product_type": "reanalysis",
                "variable": ["volumetric_soil_water_layer_1"],
                "year": [str(y) for y in years],
                "month": months,
                "day": days_list,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": bbox,
                "format": "netcdf",
            }
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                target = tmp.name
            c.retrieve("reanalysis-era5-land", req, target)
            ds = Dataset(target)
            t = ds.variables["time"]
            times = num2date(t[:], t.units)
            var = ds.variables["swvl1"] if "swvl1" in ds.variables else ds.variables["volumetric_soil_water_layer_1"]
            vals = var[:]
            # media giornaliera
            daily = {}
            for i in range(len(times)):
                v = float(vals[i].mean()) if hasattr(vals[i], "mean") else float(vals[i])
                day = times[i].date().isoformat()
                if day not in daily: daily[day]=v
                else: daily[day] = (daily[day]+v)/2.0
            ds.close(); os.remove(target)
            return {"daily": daily, "ts": time.time()}
        except Exception:
            return None
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _blocking_download)
    if data and "daily" in data:
        SM_CACHE[key] = data

def smi_from_p_et0(P: List[float], ET0: List[float]) -> List[float]:
    alpha=0.2; S=0.0; xs=[]
    for i in range(len(P)):
        forcing=(P[i] or 0.0) - (ET0[i] or 0.0)
        S=(1-alpha)*S + alpha*forcing
        xs.append(S)
    import numpy as _np
    arr=_np.array(xs, dtype=float)
    valid=arr[_np.isfinite(arr)]
    if valid.size>=5:
        p5,p95=_np.percentile(valid,[5,95])
    else:
        p5,p95=(float(arr.min()) if arr.size else -1.0, float(arr.max()) if arr.size else 1.0)
        if p95-p5<1e-6: p5,p95=p5-1,p95+1
    out=(arr-p5)/max(1e-6,(p95-p5))
    out=_np.clip(out,0.0,1.0)
    return out.tolist()

# ---------------- SPECIE: profili ecologici ----------------
SPECIES_PROFILES: Dict[str, Dict[str, Any]] = {
    "aereus": {      # "porcino nero" (leccio/quercia mediterranea; quote medio-basse)
        "hosts": ["quercia","castagno","misto"],
        "season": {"start_m": 5, "end_m": 10},       # maggio→ottobre
        "tm7_opt": (16.0, 22.0),
        "lag_base": 8.0,
        "vpd_sens": 1.05,
        "smi_bias": -0.03
    },
    "reticulatus": { # "estivo" (quercia/castagno/faggio; collinare-montano)
        "hosts": ["quercia","castagno","faggio","misto"],
        "season": {"start_m": 5, "end_m": 9},        # maggio→settembre (fino a inizio autunno)
        "tm7_opt": (17.0, 22.0),
        "lag_base": 8.5,
        "vpd_sens": 1.0,
        "smi_bias": 0.00
    },
    "edulis": {      # "porcino d'autunno" (faggio/abete/abetina-mista; quote medio-alte)
        "hosts": ["faggio","conifere","misto"],
        "season": {"start_m": 8, "end_m": 11},       # agosto→novembre (fino a gelo)
        "tm7_opt": (12.0, 18.0),                     # più fresco
        "lag_base": 10.0,
        "vpd_sens": 1.1,
        "smi_bias": +0.05                            # richiede suolo un filo più umido
    },
    "pinophilus": {  # "porcino dei pini" (conifere; montano)
        "hosts": ["conifere","misto"],
        "season": {"start_m": 7, "end_m": 10},       # estate→autunno
        "tm7_opt": (13.0, 19.0),
        "lag_base": 9.5,
        "vpd_sens": 0.95,
        "smi_bias": -0.01
    },
}

def infer_porcino_species(habitat: str, month: int, elev_m: float, aspect_oct: str) -> str:
    h = (habitat or "").lower()
    if month >= 8 and (elev_m >= 1200 or h in ("faggio","conifere")):
        return "edulis"
    if h == "conifere" and 7 <= month <= 10 and elev_m >= 800:
        return "pinophilus"
    if 5 <= month <= 9 and h in ("quercia","castagno"):
        return "reticulatus"
    return "aereus"

# ---------------- MODELLO INDICE ----------------
def smooth_forecast(xs: List[float], w: int = 3) -> List[float]:
    if not xs: return []
    out=[]
    for i in range(len(xs)):
        j0=max(0, i-w)
        j1=min(len(xs), i+w+1)
        out.append(sum(xs[j0:j1])/(j1-j0))
    return [int(round(clamp(x, 0, 100))) for x in out]

def find_best_window(xs: List[int]) -> Dict[str,int]:
    if not xs: return {"start":0,"end":0,"mean":0}
    best=(0,0,0.0)
    for i in range(len(xs)):
        for j in range(i, min(len(xs),i+7)):
            m=sum(xs[i:j+1])/(j-i+1)
            if m>best[2]: best=(i,j,m)
    return {"start": best[0], "end": best[1], "mean": int(round(best[2]))}

def micro_index(P15: float, Tm7: float, RH7: float, energy: float, twi: float, species: str, vpd_today: Optional[float]) -> int:
    prof = SPECIES_PROFILES.get(species, SPECIES_PROFILES["reticulatus"])
    # base: piogge, termica, umidità
    s = 0.0
    s += clamp(P15/30.0, 0.0, 1.0) * 40.0     # 0–40
    lo,hi = prof["tm7_opt"]
    if lo <= Tm7 <= hi: s += 25.0
    else:
        dist = min(abs(Tm7-lo), abs(Tm7-hi))
        s += clamp(25.0 - 5.0*dist, 0.0, 25.0)
    s += clamp((RH7-40)/40.0, 0.0, 1.0) * 10.0  # 0–10
    # microclima (energia penalizza in estate; TWI aiuta)
    s *= clamp(1.05 - 0.15*energy + 0.10*(twi-0.5), 0.7, 1.2)
    # VPD specie-specifico
    vpen = vpd_penalty(vpd_today or 7.0, prof["vpd_sens"])
    s *= vpen
    return int(round(clamp(s, 0.0, 100.0)))

def growth_size_estimate(age_days: float, Tm7: float, species: str) -> Tuple[float, float, float]:
    """
    Stima molto semplice della taglia in cm (cappello) intorno al picco.
    """
    base = 2.2 + 0.15*Tm7
    rate = 1.5 + 0.06*Tm7
    # specie
    if species == "edulis": rate *= 0.95
    if species == "reticulatus": rate *= 1.02
    size = clamp(base + rate*max(0.0, age_days-1.5), 1.5, 18.0)
    return size, base, rate

# ---------------- PIPELINE PRINCIPALE ----------------
async def compute_index_and_forecast(lat: float, lon: float, autohabitat: int = 1) -> Dict[str, Any]:
    om_task = asyncio.create_task(fetch_openmeteo(lat,lon,past=15,future=10))
    ow_task  = asyncio.create_task(fetch_openweather(lat,lon))
    dem_task = asyncio.create_task(fetch_elevation_grid_multiscale(lat,lon))
    osm_task = asyncio.create_task(fetch_osm_habitat(lat,lon)) if autohabitat==1 else None
    _ = asyncio.create_task(_prefetch_era5l_sm(lat,lon))

    om,ow,(elev_m,slope_deg,aspect_deg,aspect_oct,concavity)=await asyncio.gather(om_task,ow_task,dem_task)
    auto_hab, auto_conf, auto_scores = ("",0.0,{})
    if osm_task:
        try:
            auto_hab, auto_conf, auto_scores = await osm_task
        except Exception:
            auto_hab, auto_conf, auto_scores = ("",0.0,{})

    # blend passato (OM) e futuro (prevalentemente OW, fallback OM)
    timev = om.get("time", [])
    pastN = 15; futN = 10
    P_om = [float(x or 0.0) for x in om.get("P", [])][-pastN:] + [0.0]*futN
    ET0_om = [float(x or 0.0) for x in om.get("ET0", [])][-pastN:] + [0.0]*futN
    Tm_om = [float(x or 0.0) for x in om.get("T", [])]
    RH_om = [float(x or 60.0) for x in om.get("RH", [])]
    SW_om = [float(x or 0.0) for x in om.get("SW", [])]

    P_fut_ow = ow.get("futP", [])
    Tm_f_ow = ow.get("futT", [])
    RH_f_ow = ow.get("futRH", [])
    ow_len = min(len(P_fut_ow), len(Tm_f_ow), len(RH_f_ow))

    # costruisci serie passato/futuro
    P_past_blend = P_om[:pastN]
    P_fut_blend = [float(P_fut_ow[i]) if i < ow_len else 0.0 for i in range(futN)]
    Tm_p_om = Tm_om[-24*pastN:]  # orario
    Tmin_p_om = [min(Tm_om[i*24:(i+1)*24] or [0.0]) for i in range(max(0,len(Tm_om)//24 - pastN), len(Tm_om)//24)]
    Tm_f_blend = [float(Tm_f_ow[i]) if i < ow_len else (Tm_p_om[-24:].count and sum(Tm_p_om[-24:])/24.0) for i in range(futN)]
    RH_f_blend = [float(RH_f_ow[i]) if i < ow_len else 65.0 for i in range(futN)]
    RH7 = sum(RH_om[-24*7:])/max(1,len(RH_om[-24*7:]))
    SW7 = sum(SW_om[-24*7:])  # kJ/m2 prox
    Tm7 = sum(Tm_om[-24*7:])/max(1,len(Tm_om[-24*7:]))

    # API* (smoothed)
    API_val = api_index(P_past_blend)

    # SMI
    smi_series = smi_from_p_et0(P_om, ET0_om)
    cache_key = f"{round(lat,3)},{round(lon,3)}"
    sm_used = "P-ET0"
    import numpy as _np
    if cache_key in SM_CACHE:
        daily_sm = SM_CACHE[cache_key].get("daily", {})
        tmp=[float(daily_sm.get(dstr, float('nan'))) for dstr in timev]
        arr=_np.array(tmp, dtype=float)
        if _np.any(_np.isfinite(arr)):
            valid=arr[_np.isfinite(arr)]
            p5,p95=_np.percentile(valid,[5,95])
            arr=(arr-p5)/max(1e-6,(p95-p5))
            arr=_np.clip(arr,0.0,1.0)
            smi_series=arr.tolist()
            sm_used = "ERA5-Land (CDS cache)"
    sm_last=_np.array(smi_series[-15:], dtype=float)
    sm_last=sm_last[_np.isfinite(sm_last)]
    sm_thr=float(_np.percentile(sm_last,55)) if sm_last.size>0 else 0.6
    smi_now = float(smi_series[pastN-1]) if pastN-1 < len(smi_series) else 0.5

    # Specie stimata
    month_now = datetime.now(timezone.utc).month
    species = infer_porcino_species(habitat_used=auto_hab or "misto", month=month_now, elev_m=float(elev_m), aspect_oct=aspect_oct)
    prof = SPECIES_PROFILES[species]
    # leggera correzione dell'ottimo termico in quota
    tm_lo, tm_hi = prof["tm7_opt"]
    if elev_m >= 1400: tm_lo -= 1.0; tm_hi -= 1.0
    species_profile_txt = {
        "season_txt": f"{prof['season']['start_m']:02d}→{prof['season']['end_m']:02d}",
        "tm7_txt": f"{tm_lo:.0f}–{tm_hi:.0f}°C",
        "lag_txt": f"~{prof['lag_base']:.1f} gg (prima di correzioni SMI/T)",
        "vpd_txt": "alta" if prof["vpd_sens"]>=1.1 else ("bassa" if prof["vpd_sens"]<=0.9 else "media")
    }

    # Shock termico
    shock = cold_shock_from_tmin_series(Tmin_p_om)

    # VPD futuro (uso Tm_f_blend e RH_f_om)
    vpd_fut=[vpd_hpa(float(Tm_f_blend[i]), float(RH_f_ow[i] if i<len(RH_f_ow) else 60.0)) for i in range(futN)]
    vpd_today = vpd_fut[0] if vpd_fut else None

    # Microclima
    energy = microclimate_energy(aspect_oct, float(slope_deg), month_now)
    twi = twi_proxy_from_slope_concavity(float(slope_deg), float(concavity))
    micro_amp = clamp(energy * (0.8 + 0.4*(twi-0.5)*2.0), 0.6, 1.2)

    # Affidabilità OM vs OW
    def reliability_from_sources(P_ow:List[float], P_om:List[float], T_ow:List[float], T_om:List[float]) -> float:
        n=min(len(P_ow),len(P_om),len(T_ow),len(T_om))
        if n==0: return 0.6
        dp=[abs((P_ow[i] or 0.0)-(P_om[i] or 0.0)) for i in range(n)]
        dt=[abs((T_ow[i] or 0.0)-(T_om[i] or 0.0)) for i in range(n)]
        avg_dp=sum(dp)/n; avg_dt=sum(dt)/n
        sP = 0.95/(1.0+avg_dp/10.0)
        sT = 0.95/(1.0+avg_dt/6.0)
        return clamp(0.25 + 0.5*((sP+sT)/2.0), 0.25, 0.95)
    reliability = reliability_from_sources(P_fut_ow[:ow_len],P_om[:ow_len],Tm_f_ow[:ow_len],Tm_om[:ow_len]) if ow_len else 0.6

    # ---- Event detection (soglia fissa, ampiezza modulata da SMI/specie) ----
    def rain_events(P: List[float]) -> List[Tuple[int,float]]:
        # trova cluster consecutivi con mm/giorno e soglia dinamica
        thresh = 3.0  # mm/giorno
        events=[]
        i=0
        while i < len(P):
            if P[i] >= thresh:
                j=i
                s=0.0
                while j < len(P) and P[j] >= thresh:
                    s+=P[j]; j+=1
                if s >= 8.0:
                    events.append((i, s))
                i=j
            else: i += 1
        return events
    ev_past = rain_events(P_past_blend)
    ev_fut_raw = rain_events(P_fut_blend)
    ev_fut=[(pastN+i, mm) for (i,mm) in ev_fut_raw]
    events = ev_past + ev_fut

    # ---- Previsione con lag specie-specifico e VPD specie-specifico ----
    forecast=[0.0]*futN; details=[]
    for (ev_idx_abs, mm_tot) in events:
        # SMI vicino all'evento (±1 giorno)
        i0=max(0, ev_idx_abs-1); i1=min(len(smi_series), ev_idx_abs+1)
        sm_loc=float(_np.nanmean(_np.array(smi_series[i0:i1+1], dtype=float))) if i1>i0 else float(smi_series[ev_idx_abs])

        # lag di base dalla specie
        s_bias = SPECIES_PROFILES[species]["smi_bias"]
        lag_base = SPECIES_PROFILES[species]["lag_base"]
        lag = clamp(lag_base - 3.0*(sm_loc + s_bias) - 1.2*shock, 5.0, 15.0)  # 5–15 gg
        # picco previsto
        peak_day = int(round(ev_idx_abs + lag - pastN))
        # intensità evento
        strength = event_strength(mm_tot)
        # modulazioni microclima/VPD
        vpen = vpd_penalty(vpd_fut[0] if vpd_fut else 7.0, SPECIES_PROFILES[species]["vpd_sens"])
        amp = clamp(100.0 * strength * (0.9 + 0.2*(sm_loc - sm_thr)) * micro_amp * vpen, 5.0, 100.0)
        if 0 <= peak_day < futN:
            forecast[peak_day] = max(forecast[peak_day], amp)
            details.append({
                "event_day_abs_index": ev_idx_abs,
                "mm_tot": round(mm_tot,1),
                "smi_local": round(sm_loc,2),
                "predicted_peak_abs_index": pastN+peak_day,
                "peak_in_days": peak_day,
                "amplitude": round(amp,1),
                "observed": ev_idx_abs < pastN
            })

    # composizione: indice oggi e smoothing 10gg
    idx_today = micro_index(sum(P_past_blend), Tm7, RH7, energy, twi, species, vpd_today)
    smoothed = smooth_forecast(forecast)
    # se forecast vuoto, mantieni un decadimento dal valore attuale
    if not any(smoothed):
        smoothed = [max(0, int(round(idx_today * (0.85**i)))) for i in range(futN)]
    # migliore finestra
    bw = find_best_window(smoothed)

    # taglia/età media attesa al picco: stima discorsiva
    if details:
        age_days = 1.0 + 0.3*Tm7 + 0.5*(smi_now - sm_thr)
    else:
        age_days = 2.0 + 0.2*Tm7
    size_cm, size_base, size_rate = growth_size_estimate(age_days, Tm7, species)
    # classi
    if size_cm < 5.0: size_cls = "bottoni (2–5 cm)"
    elif size_cm < 10.0: size_cls = "medi (6–10 cm)"
    else: size_cls = "grandi (10–15+ cm)"
    smin = clamp(size_rate * max(0, age_days-2), 1.5, 18.0)
    smax = clamp(size_rate * (age_days+2), 1.5, 18.0)

    # ---- Raccolto atteso: SEMPRE visibile (ponderato) ----
    def harvest_from_index_always(score:int, hours:int, reliability:float, main_observed:bool,
                                  vpd_hpa_today: Optional[float], smi_now: float, sm_thr: float) -> Tuple[str, str]:
        """
        Restituisce SEMPRE un range 'X–Y porcini', ponderato:
          - reliability:   fattore 0.4–1.2 (0.25→0.95)
          - observed:      ×1.00 se evento osservato, ×0.90 se no
          - VPD odierno:   ×1.00 (<=6), ×0.85 (6–10), ×0.70 (>10)
          - SMI sotto soglia: ×0.90
        """
        factor = 1.0
        # ore
        hours = max(1, int(hours))
        factor *= 1.0 if hours <= 2 else 1.45 if hours <= 4 else 1.6
        if score < 15: base = (0, 1)
        elif score < 35: base = (1, 2)
        elif score < 55: base = (2, 4)
        elif score < 70: base = (4, 8)
        elif score < 85: base = (6, 12)
        else: base = (10, 20)
        lo0, hi0 = base
        lo0 = max(0, int(round(lo0 * factor)))
        hi0 = max(1, int(round(hi0 * factor)))

        rel = max(0.25, min(0.95, reliability))
        f_rel = 0.4 + 0.8 * ((rel - 0.25) / (0.95 - 0.25))  # 0.4..1.2
        f_obs = 1.0 if main_observed else 0.90
        if vpd_hpa_today is None:
            f_vpd = 0.9; vpd_txt = "n.d."
        else:
            if vpd_hpa_today <= 6.0: f_vpd = 1.0; vpd_txt = f"{vpd_hpa_today:.1f} hPa (ok)"
            elif vpd_hpa_today <= 10.0: f_vpd = 0.85; vpd_txt = f"{vpd_hpa_today:.1f} hPa (medio)"
            else: f_vpd = 0.70; vpd_txt = f"{vpd_hpa_today:.1f} hPa (alto)"
        f_smi = 0.90 if smi_now < sm_thr else 1.0

        lo = int(round(lo0 * f_rel * f_obs * f_vpd * f_smi))
        hi = int(round(hi0 * f_rel * f_obs * f_vpd * f_smi))
        if lo >= hi: hi = lo + 1

        note = (f"Stima specie B. {species}; ponderazione (reliability={reliability:.2f}, "
                f"{'evento osservato' if main_observed else 'evento non osservato'}, "
                f"VPD={vpd_txt}, SMI {'< thr' if smi_now < sm_thr else '≥ thr'}).")
        return (f"{lo}–{hi} porcini", note)

    main_observed = any(d.get("observed") and pastN <= d.get("predicted_peak_abs_index",1e9) < pastN+futN for d in details)
    harvest_txt, harvest_note = harvest_from_index_always(idx_today, 3, reliability, main_observed, vpd_today, smi_now, sm_thr)

    # Confidence aggregata (0–100)
    f_rel = reliability
    f_obs = 1.0 if main_observed else 0.6
    f_smi = 1.0 if smi_now >= sm_thr else (0.5 if smi_now >= sm_thr*0.8 else 0.25)
    f_vpd = 1.0 if (vpd_today is not None and vpd_today <= 6.0) else (0.6 if (vpd_today is not None and vpd_today <= 9.0) else 0.4)
    model_conf = int(round(100*(0.5*f_rel + 0.2*f_obs + 0.2*f_smi + 0.1*f_vpd)))
    model_conf = clamp(model_conf, 25, 95)

    # Tabelle piogge
    rain_past={timev[i]: round(P_past_blend[i],1) for i in range(min(pastN,len(timev)))}
    rain_future={timev[pastN+i] if pastN+i<len(timev) else f"+{i+1}d":round(P_fut_blend[i],1) for i in range(futN)}

    # payload finale
    response_data = {
        "lat": lat, "lon": lon,
        "elevation_m": round(elev_m),
        "slope_deg": slope_deg, "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct if aspect_oct else "NA",
        "concavity": round(concavity,3),

        "API_star_mm": round(API_val,1),
        "P7_mm": round(sum(P_past_blend[-7:]),1),
        "P15_mm": round(sum(P_past_blend),1),
        "ET0_7d_mm": round(ET7 := sum(ET0_om[-pastN:]),1),
        "RH7_pct": round(RH7,1),
        "SW7_kj": round(SW7,0),
        "Tmean7_c": round(Tm7,1),

        "index": int(smoothed[0] if smoothed else 0),
        "forecast": [int(x) for x in smoothed],
        "best_window": {"start": bw["start"], "end": bw["end"], "mean": bw["mean"]},
        "harvest_estimate": harvest_txt,
        "harvest_note": harvest_note,
        "reliability": round(reliability,3),

        "rain_past": rain_past,
        "rain_future": rain_future,

        "habitat_used": auto_hab or "misto",
        "habitat_source": "OSM" if auto_hab else "manuale",
        "auto_habitat_scores": auto_scores,
        "auto_habitat_confidence": round(auto_conf,3),

        "flush_events": details,

        "size_cm": round(size_cm,1),
        "size_class": size_cls,
        "size_range_cm": [round(smin,1), round(smax,1)],

        "diagnostics": {
            "smi_source": sm_used,
            "smi_threshold": round(sm_thr,2),
            "twi_proxy": round(twi,2),
            "energy_index": round(energy,2),
        },
        "model_confidence": model_conf,
        "vpd_today_hpa": round(vpd_today,1) if vpd_today is not None else None,

        # specie/ecologia
        "species": species,
        "species_profile": species_profile_txt,
        # metrica interna utile alla sezione Analisi Modello
        "smi_now": round(smi_now,2),
    }

    response_data["dynamic_explanation"] = build_analysis_text_long(response_data)
    return response_data

# ---------------- Analisi dinamica (estesa) ----------------
def build_analysis_text_long(payload: Dict[str,Any]) -> str:
    idx = payload["index"]
    best = payload["best_window"]
    elev = payload["elevation_m"]; slope = payload["slope_deg"]; aspect = payload.get("aspect_octant") or "NA"
    habitat_used = (payload.get("habitat_used") or "").capitalize() or "Altro"
    habsrc = payload.get("habitat_source","manuale")
    P15 = payload["P15_mm"]; P7 = payload["P7_mm"]; Tm7 = payload["Tmean7_c"]; RH7 = payload["RH7_pct"]
    diag = payload.get("diagnostics", {})
    smi_src = diag.get("smi_source","P-ET0"); sm_thr = diag.get("smi_threshold",0.6)
    twi = diag.get("twi_proxy",0.5); energy = diag.get("energy_index",0.5)
    rel = payload.get("reliability",0.6)
    conf = payload.get("model_confidence", 60)
    vpd_today = payload.get("vpd_today_hpa", None)
    harvest = payload.get("harvest_estimate","—"); harvest_note = payload.get("harvest_note","")
    species = payload.get("species","reticulatus")

    lines = []
    lines.append(f"<p><strong>Specie stimata</strong>: <strong>Boletus {species}</strong> • Habitat: <strong>{habitat_used}</strong> ({habsrc}). "
                 f"Quota <strong>{elev} m</strong>, pendenza <strong>{slope}°</strong>, esposizione <strong>{aspect}</strong>. "
                 f"Microclima: indice energetico <strong>{energy:.2f}</strong>, TWI-proxy <strong>{twi:.2f}</strong>.</p>")
    lines.append(f"<p><strong>Stato idrico-termico</strong> — Sorgente SMI: <strong>{smi_src}</strong> (soglia {sm_thr:.2f}). "
                 f"Piogge antecedenti: <strong>{P15:.0f} mm/15g</strong> (7g: {P7:.0f} mm). "
                 f"Termica 7g: <strong>{Tm7:.1f}°C</strong>; umidità 7g: <strong>{RH7:.0f}%</strong>.</p>")
    # --- Dati ERA5-Land: spiegazione/availability ---
    if "ERA5-Land" in str(smi_src):
        lines.append("<p><strong>Dati ERA5-Land</strong> — uso <em>volumetric soil water</em> strato 1 (0–7 cm) del reanalysis Copernicus ERA5-Land; i valori orari nell’intorno (~0.1°) sono mediati al giorno e <em>normalizzati</em> tra 5°–95° percentile locali per ottenere un indice SMI 0–1 utilizzato nella stima e nel lag.</p>")
    else:
        lines.append("<p><strong>Dati ERA5-Land</strong> — <em>dati non disponibili</em> per questo punto/oggi; viene usato un indicatore SMI derivato da precipitazione−ET₀ locale.</p>")
    if vpd_today is not None:
        lines.append(f"<p><strong>Secchezza dell'aria</strong> — VPD ≈ <strong>{vpd_today:.1f} hPa</strong> "
                     f"({ 'favorevole' if vpd_today<=6 else 'critico' if vpd_today>=12 else 'intermedio' }).</p>")
    if best and best.get("mean",0)>0:
        s,e,m = best["start"], best["end"], best["mean"]
        lines.append(f"<p><strong>Finestra migliore</strong> prossimi 10 giorni: <strong>giorni {s+1}–{e+1}</strong> "
                     f"(media indice ≈ <strong>{m}</strong>). Indice oggi: <strong>{idx}/100</strong>.</p>")
    lines.append(f"<p><strong>Raccolto atteso</strong>: {harvest}. <em>{harvest_note}</em></p>")
    # Dettaglio specie
    prof = payload.get("species_profile", {})
    if prof:
        lines.append("<h4>Parametri specie-specifici stimati</h4><ul>")
        lines.append(f"<li>Finestra stagionale tipica: <strong>{prof.get('season_txt','—')}</strong></li>")
        lines.append(f"<li>Ottimo termico Tm7: <strong>{prof.get('tm7_txt','—')}</strong></li>")
        lines.append(f"<li>Lag di base dopo pioggia: <strong>{prof.get('lag_txt','—')}</strong></li>")
        lines.append(f"<li>Sensibilità al VPD: <strong>{prof.get('vpd_txt','—')}</strong></li>")
        lines.append("</ul>")
    # Motivazioni affidabilità
    reasons = []
    reasons.append(f"convergenza OM/OW: <strong>{rel:.2f}</strong>")
    reasons.append(f"specie plausibile per habitat/quota/mese: <strong>B. {species}</strong>")
    if vpd_today is not None:
        reasons.append(f"VPD odierno: <strong>{vpd_today:.1f} hPa</strong>")

    # --- Consigli operativi dinamici (discorsivi) ---
    try:
        idx_today = int(idx)
    except Exception:
        idx_today = 0
    # finestra ottimale
    wtxt = ""
    if best and best.get("mean",0)>0:
        s,e,m = best["start"], best["end"], best["mean"]
        if s == 0:
            wtxt = f"Oggi e prossimi {max(0,e)} giorni rappresentano una <em>buona finestra</em> (media indice ≈ {m})."
        else:
            wtxt = f"La finestra migliore inizia tra ~{s} giorni (durata {e-s+1} gg; media indice ≈ {m})."
    # regole empiriche
    if idx_today >= 80:
        advice = "Vai oggi: aspettati <strong>molti porcini</strong>, soprattutto in zone ombreggiate, margini umidi e depressioni con TWI alto. Controlla anche le radure fresche dopo i faggi più maturi."
    elif idx_today >= 60:
        advice = "Giornata promettente: fai <em>check mirati</em> nelle ore fresche. Se il suolo appare asciutto in esposizioni S–SW, cerca versanti N–NE, canaloni e piccole conche."
    elif idx_today >= 45:
        advice = "Situazione di <em>attesa</em>: potresti trovare nuclei sparsi; meglio puntare ai <em>microhabitat umidi</em>. Se è appena piovuto, considera 1–3 giorni di attesa per il picco locale."
    else:
        advice = "Probabilità bassa oggi: conviene attendere nuove piogge o la finestra indicata. Se esci, focalizzati su zone a ombra perenne e lettiera spessa."
    # modulazioni con VPD e piogge recenti
    if vpd_today is not None and vpd_today >= 10:
        advice += " Con VPD alto nelle ore centrali, privilegia <em>prime ore del mattino</em> e fondovalle ombreggiati."
    if P15 < 8:
        advice += " Mancano eventi piovosi significativi nelle ultime due settimane: il bosco potrebbe essere <em>avaro</em> salvo in nicchie molto umide."
    lines.append("<h4>Consigli operativi</h4><p>" + advice + (" " + wtxt if wtxt else "") + "</p>")

    lines.append("<h4>Affidabilità della stima</h4><ul>" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>")
    return "\n".join(lines)

# ----------------------------- ENDPOINTS -----------------------------
@app.get("/api/health")
async def health():
    return {"ok":True,"time":datetime.now(timezone.utc).isoformat()}

@app.get("/api/geocode")
async def api_geocode(q:str):
    # best-effort: Nominatim → fallback Open-Meteo
    try:
        url="https://nominatim.openstreetmap.org/search"
        params={"format":"json","q":q,"addressdetails":1,"limit":1,"email":os.getenv("NOMINATIM_EMAIL","info@example.com")}
        async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
            r=await c.get(url, params=params)
            r.raise_for_status()
            j=r.json()
        if j:
            x=j[0]
            return {
                "display_name": x.get("display_name"),
                "lat": float(x.get("lat",0.0)),
                "lon": float(x.get("lon",0.0))
            }
    except Exception:
        pass
    # fallback povero: Open-Meteo geocoding
    try:
        url="https://geocoding-api.open-meteo.com/v1/search"
        params={"name":q,"count":1,"language":"it","format":"json"}
        async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
            r=await c.get(url, params=params)
            r.raise_for_status()
            j=r.json()
        if j and j.get("results"):
            x=j["results"][0]
            return {
                "display_name": f"{x.get('name')}, {x.get('admin1','')}, {x.get('country','')}".strip(", "),
                "lat": float(x.get("latitude",0.0)),
                "lon": float(x.get("longitude",0.0))
            }
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Località non trovata")

@app.get("/api/analyze")
async def api_analyze(lat: float = Query(..., ge=-90.0, le=90.0),
                      lon: float = Query(..., ge=-180.0, le=180.0),
                      autohabitat: int = Query(1, ge=0, le=1)):
    try:
        data = await compute_index_and_forecast(lat, lon, autohabitat=autohabitat)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore analisi: {e}")

@app.get("/api/forecast_text")
async def api_forecast_text(lat: float, lon: float):
    # helper: restituisce solo la sezione "Analisi Modello"
    data = await compute_index_and_forecast(lat,lon,autohabitat=1)
    return {"html": data.get("dynamic_explanation","")}

# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")), reload=False)
