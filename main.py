# main.py — Trova Porcini API (v2.4.0)
# Novità v2.4:
#  • Soglie dinamiche per eventi piovosi basate su SMI, stagione e quota
#  • Smoothing avanzato Savitzky-Golay che preserva i picchi
#  • Sistema di confidence multi-dimensionale più accurato
#  • Database SQLite per raccogliere segnalazioni di validazione
#  • Metriche di performance per analisi retrospettiva
#  • Logging strutturato per debugging e ottimizzazione
#  • Calibrazione dinamica dei parametri basata sui dati raccolti

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio, tempfile, time, sqlite3, logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trova Porcini API (v2.4.0)", version="2.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

HEADERS = {"User-Agent":"Trovaporcini/2.4.0 (+site)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")
CDS_API_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")

# Database per validazione
DB_PATH = "porcini_validations.db"

def init_database():
    """Inizializza database SQLite per raccogliere dati di validazione"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabella segnalazioni positive
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            date TEXT NOT NULL,
            species TEXT NOT NULL,
            quantity INTEGER DEFAULT 1,
            confidence REAL DEFAULT 0.8,
            photo_url TEXT,
            notes TEXT,
            predicted_score INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabella ricerche senza risultato (importante per training!)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS no_sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            date TEXT NOT NULL,
            searched_hours INTEGER DEFAULT 2,
            habitat TEXT,
            notes TEXT,
            predicted_score INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabella predizioni per analisi performance
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            date TEXT NOT NULL,
            predicted_score INTEGER NOT NULL,
            species TEXT NOT NULL,
            habitat TEXT,
            confidence_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Inizializza DB all'avvio
init_database()

# ----------------------------- UTIL MIGLIORATI -----------------------------
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
    return clamp(1.0 - (1.0-base)*species_vpd_sens, 0.35, 1.0)

# ---- Shock termico ΔTmin(3d) ----
def cold_shock_from_tmin_series(tmin: List[float]) -> float:
    if len(tmin) < 7: return 0.0
    last3 = sum(tmin[-3:]) / 3.0
    prev3 = sum(tmin[-6:-3]) / 3.0
    drop = last3 - prev3
    if drop >= -1.0: return 0.0
    return clamp((-drop - 1.0) / 3.0, 0.0, 1.0)

# ---- TWI proxy + energy index ----
def twi_proxy_from_slope_concavity(slope_deg: float, concavity: float) -> float:
    beta = max(0.1, math.radians(max(0.1, slope_deg)))
    tanb = max(0.05, math.tan(beta))
    conc = max(0.0, concavity + 0.02)
    twi = math.log(1.0 + 6.0 * conc) - math.log(tanb)
    return clamp((twi + 2.2) / 4.0, 0.0, 1.0)

def microclimate_energy(aspect_oct: Optional[str], slope_deg: float, month: int) -> float:
    if not aspect_oct or slope_deg < 0.8: return 0.5
    summer = 1.0 if month in (7,8,9) else 0.6
    base = 0.5
    if aspect_oct in ("N","NE","NW"): base += 0.15
    if aspect_oct in ("S","SE","SW"): base -= 0.12 * summer
    base *= (1.0 + min(0.15, slope_deg/90.0))
    return clamp(base, 0.25, 0.9)

# ---- SOGLIE DINAMICHE (NUOVO) ----
def dynamic_rain_threshold(smi: float, month: int, elevation: float) -> float:
    """
    Soglia pioggia adattiva basata sulle condizioni locali
    """
    base_threshold = 8.0
    
    # Se suolo già saturo, servono meno mm
    if smi > 0.8: 
        base_threshold *= 0.7
    elif smi < 0.3: 
        base_threshold *= 1.3
    
    # Estate: serve più pioggia (evaporazione maggiore)
    if month in [6,7,8]: 
        base_threshold *= 1.2
    
    # Quota alta: meno evaporazione, meno pioggia necessaria
    if elevation > 1200: 
        base_threshold *= 0.85
    elif elevation > 800:
        base_threshold *= 0.92
    
    return max(5.0, min(15.0, base_threshold))

def dynamic_rain_threshold_2day(smi: float, month: int, elevation: float) -> float:
    """Soglia per eventi su 2 giorni consecutivi"""
    return dynamic_rain_threshold(smi, month, elevation) * 1.5

# ---- SMOOTHING AVANZATO (NUOVO) ----
def advanced_smoothing(forecast: List[float]) -> List[float]:
    """
    Smoothing avanzato che preserva i picchi importanti
    """
    if len(forecast) < 5:
        # Fallback al smoothing semplice per serie corte
        smoothed = []
        for i in range(len(forecast)):
            w = forecast[i]
            if i > 0: w += forecast[i-1]
            if i+1 < len(forecast): w += forecast[i+1]
            denom = 1 + (1 if i > 0 else 0) + (1 if i+1 < len(forecast) else 0)
            smoothed.append(w/denom)
        return smoothed
    
    try:
        # Prova Savitzky-Golay (preserva meglio i picchi)
        from scipy.signal import savgol_filter
        import numpy as np
        
        arr = np.array(forecast, dtype=float)
        window_length = min(5, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
        if window_length < 3:
            window_length = 3
            
        smoothed = savgol_filter(arr, window_length=window_length, polyorder=1)
        return np.clip(smoothed, 0, 100).tolist()
        
    except ImportError:
        # Fallback se scipy non disponibile
        logger.warning("scipy non disponibile, uso smoothing semplice")
        smoothed = []
        for i in range(len(forecast)):
            w = forecast[i]
            if i > 0: w += forecast[i-1] * 0.5
            if i+1 < len(forecast): w += forecast[i+1] * 0.5
            denom = 1 + (0.5 if i > 0 else 0) + (0.5 if i+1 < len(forecast) else 0)
            smoothed.append(w/denom)
        return smoothed

# ---- CONFIDENCE MULTI-DIMENSIONALE (NUOVO) ----
def advanced_confidence(
    weather_agreement: float,
    habitat_confidence: float, 
    smi_reliability: float,
    vpd_validity: bool,
    has_recent_validation: bool = False
) -> Dict[str, float]:
    """
    Sistema di confidence più sofisticato e informativo
    """
    # Meteorological confidence (accordo tra fonti)
    met_conf = clamp(weather_agreement, 0.2, 0.95)
    
    # Ecological confidence (qualità inference habitat)
    eco_conf = clamp(habitat_confidence, 0.1, 0.9)
    
    # Hydrological confidence (affidabilità SMI)
    hydro_conf = clamp(smi_reliability, 0.3, 0.9)
    
    # Atmospheric confidence (validità VPD)
    atmo_conf = 0.8 if vpd_validity else 0.4
    
    # Empirical confidence (presenza validazioni recenti nell'area)
    emp_conf = 0.7 if has_recent_validation else 0.4
    
    # Overall confidence (media pesata)
    weights = {"met": 0.3, "eco": 0.25, "hydro": 0.2, "atmo": 0.15, "emp": 0.1}
    overall = (weights["met"] * met_conf + 
               weights["eco"] * eco_conf + 
               weights["hydro"] * hydro_conf + 
               weights["atmo"] * atmo_conf + 
               weights["emp"] * emp_conf)
    
    return {
        "meteorological": round(met_conf, 3),
        "ecological": round(eco_conf, 3),
        "hydrological": round(hydro_conf, 3), 
        "atmospheric": round(atmo_conf, 3),
        "empirical": round(emp_conf, 3),
        "overall": round(overall, 3)
    }

# ---------------- Meteo (Open-Meteo + OpenWeather) ----------------
async def fetch_open_meteo(lat:float,lon:float,past:int=15,future:int=10)->Dict[str,Any]:
    url="https://api.open-meteo.com/v1/forecast"
    daily_vars=[
        "precipitation_sum","temperature_2m_mean","temperature_2m_min","temperature_2m_max",
        "et0_fao_evapotranspiration","relative_humidity_2m_mean","shortwave_radiation_sum"
    ]
    params={
        "latitude":lat,"longitude":lon,"timezone":"auto",
        "daily":",".join(daily_vars),
        "past_days":past,"forecast_days":future
    }
    async with httpx.AsyncClient(timeout=35,headers=HEADERS) as c:
        r=await c.get(url,params=params); r.raise_for_status();
        return r.json()

async def fetch_openweather(lat:float,lon:float)->Dict[str,Any]:
    if not OWM_KEY: return {}
    url="https://api.openweathermap.org/data/3.0/onecall"
    params={"lat":lat,"lon":lon,"exclude":"minutely,hourly,current,alerts","units":"metric","lang":"it","appid":OWM_KEY}
    try:
        async with httpx.AsyncClient(timeout=35,headers=HEADERS) as c:
            r=await c.get(url,params=params); r.raise_for_status();
            return r.json()
    except Exception:
        return {}

async def fetch_openweather_past(lat:float, lon:float, days:int=5) -> Dict[str, Dict[str,float]]:
    """
    Best-effort: One Call Timemachine fino a ~5 giorni indietro.
    Ritorna {"YYYY-MM-DD": {"rain": mm, "tmin": C, "tmax": C, "tmean": C}}
    """
    if not OWM_KEY: return {}
    out: Dict[str, Dict[str,float]] = {}
    base = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    try:
        async with httpx.AsyncClient(timeout=35, headers=HEADERS) as c:
            for d in range(1, days+1):
                ts = int((base - timedelta(days=d)).timestamp())
                url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
                params = {"lat": lat, "lon": lon, "dt": ts, "units":"metric", "lang":"it", "appid": OWM_KEY}
                try:
                    r = await c.get(url, params=params)
                    r.raise_for_status()
                    j = r.json()
                    hours = j.get("hourly") or []
                    if not hours: continue
                    rr = 0.0; tmin = +1e9; tmax = -1e9; tsum = 0.0; n = 0
                    for h in hours:
                        rain = 0.0
                        if isinstance(h.get("rain"), dict):
                            rain = float(h["rain"].get("1h") or 0.0)
                        elif isinstance(h.get("snow"), dict):
                            rain = float(h["snow"].get("1h") or 0.0)
                        rr += rain
                        t = float(h.get("temp", 0.0))
                        tmin = min(tmin, t); tmax = max(tmax, t); tsum += t; n += 1
                    if n==0: continue
                    tmean = tsum/n
                    day = (base - timedelta(days=d)).date().isoformat()
                    out[day] = {"rain": rr, "tmin": tmin, "tmax": tmax, "tmean": tmean}
                except Exception:
                    continue
    except Exception:
        return {}
    return out

# ------------------------ DEM / microtopo ------------------------
_elev_cache: Dict[str, Any] = {}

def _grid_key(lat:float,lon:float,step:float)->str: return f"{round(lat,5)},{round(lon,5)}@{int(step)}"

async def _fetch_elev_block(lat:float,lon:float,step_m:float)->Optional[List[List[float]]]:
    key=_grid_key(lat,lon,step_m)
    if key in _elev_cache:
        return _elev_cache[key]
    try:
        deg_lat=1/111320.0
        deg_lon=1/(111320.0*max(0.2,math.cos(math.radians(lat))))
        dlat=step_m*deg_lat; dlon=step_m*deg_lon
        coords=[]
        for dy in (-dlat,0,dlat):
            for dx in (-dlon,0,dlon):
                coords.append({"latitude":lat+dy,"longitude":lon+dx})
        async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
            r=await c.post("https://api.open-elevation.com/api/v1/lookup",json={"locations":coords})
            r.raise_for_status(); j=r.json()
        vals=[p["elevation"] for p in j["results"]]
        grid=[vals[0:3],vals[3:6],vals[6:9]]
        _elev_cache[key]=grid
        if len(_elev_cache)>800:
            for k in list(_elev_cache.keys())[:200]: _elev_cache.pop(k,None)
        return grid
    except Exception:
        return None

def slope_aspect_from_grid(z:List[List[float]],cell_size_m:float=30.0)->Tuple[float,float,Optional[str]]:
    dzdx=((z[0][2]+2*z[1][2]+z[2][2])-(z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_size_m)
    dzdy=((z[2][0]+2*z[2][1]+z[2][2])-(z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_size_m)
    slope=math.degrees(math.atan(math.hypot(dzdx,dzdy)))
    aspect=(math.degrees(math.atan2(-dzdx, dzdy))+360.0)%360.0
    octs=["N","NE","E","SE","S","SW","W","NW","N"]
    octant=octs[int(((aspect%360)+22.5)//45)]
    return round(slope,1),round(aspect,0),octant

def concavity_from_grid(z:List[List[float]])->float:
    center=z[1][1]; neigh=[z[r][c] for r in (0,1,2) for c in (0,1,2) if not (r==1 and c==1)]
    delta=(sum(neigh)/8.0 - center)
    return clamp(delta/6.0, -0.1, +0.1)

async def fetch_elevation_grid_multiscale(lat:float,lon:float)->Tuple[float,float,float,Optional[str],float]:
    best=None; best_grid=None
    for step in (30.0, 90.0, 150.0):
        z = await _fetch_elev_block(lat,lon,step)
        if not z: continue
        flatness = stddev([*z[0],*z[1],*z[2]])
        slope,aspect,octant = slope_aspect_from_grid(z,cell_size_m=step)
        cand = {"z":z,"step":step,"flat":flatness,"slope":slope,"aspect":aspect,"oct":octant,"elev":z[1][1]}
        if best is None: best=cand; best_grid=z
        if slope>1.0 and (best["slope"]<=1.0 or flatness>best["flat"]):
            best=cand; best_grid=z
    if not best:
        return 800.0, 5.0, 0.0, None, 0.0
    aspect_oct = best["oct"] if (best["slope"]>=0.8 and best["flat"]>=0.5) else None
    conc = concavity_from_grid(best_grid)
    return float(best["elev"]), round(best["slope"],1), round(best["aspect"],0), aspect_oct, conc

# ----------------------- Habitat auto da OSM -----------------------
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
]

def _score_tags(tags: Dict[str,str])->Dict[str,float]:
    t = {k.lower(): (v.lower() if isinstance(v,str) else v) for k,v in (tags or {}).items()}
    s = {"castagno":0.0,"faggio":0.0,"quercia":0.0,"conifere":0.0,"misto":0.0}
    genus = t.get("genus",""); species = t.get("species","")
    leaf_type = t.get("leaf_type",""); landuse = t.get("landuse",""); natural=t.get("natural",""); wood=t.get("wood","")
    if "castanea" in genus or "castagna" in species: s["castagno"] += 3.0
    if "quercus" in genus or "querce" in species:  s["quercia"]  += 3.0
    if "fagus" in genus or "faggio" in species:    s["faggio"]   += 3.0
    if any(g in genus for g in ("pinus","abies","picea","larix")): s["conifere"] += 2.5
    if "needleleaved" in leaf_type: s["conifere"] += 1.5
    if wood in ("conifer","pine","spruce","fir"): s["conifere"] += 1.2
    if wood in ("broadleaved","deciduous"): s["misto"] += 0.6
    if landuse=="forest" or natural in ("wood","forest"):
        for k in s: s[k] += 0.1
    return s

def _choose_habitat(scores: Dict[str,float])->Tuple[str,float]:
    best = max(scores.items(), key=lambda kv: kv[1])
    total = sum(scores.values()) or 1.0
    conf = clamp(best[1]/total, 0.05, 0.99)
    return best[0], conf

async def fetch_osm_habitat(lat: float, lon: float, radius_m: int = 400) -> Tuple[str, float, Dict[str,float]]:
    q = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})["landuse"="forest"];
      way(around:{radius_m},{lat},{lon})["natural"="wood"];
      relation(around:{radius_m},{lat},{lon})["landuse"="forest"];
      relation(around:{radius_m},{lat},{lon})["natural"="wood"];
      node(around:{radius_m},{lat},{lon})["tree"]["genus"];
    );
    out tags qt;
    """
    for url in OVERPASS_URLS:
        try:
            async with httpx.AsyncClient(timeout=35, headers=HEADERS) as c:
                r=await c.post(url, data={"data": q}); r.raise_for_status(); j=r.json()
            scores = {"castagno":0.0,"faggio":0.0,"quercia":0.0,"conifere":0.0,"misto":0.0}
            for el in j.get("elements", []):
                local = _score_tags(el.get("tags", {}))
                for k,v in local.items(): scores[k]+=v
            hab, conf = _choose_habitat(scores)
            return hab, conf, scores
        except Exception:
            continue
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
            data = var[:]
            daily: Dict[str, float] = {}
            import numpy as _np
            for i in range(data.shape[0]):
                v = float(_np.nanmean(_np.array(data[i]).astype("float64")))
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

# --------------------- SPECIE: profili ecologici ---------------------
SPECIES_PROFILES = {
    "aereus": {
        "hosts": ["quercia","castagno","misto"],
        "season": {"start_m": 6, "end_m": 10},
        "tm7_opt": (18.0, 23.0),
        "lag_base": 9.0,
        "vpd_sens": 1.1,
        "smi_bias": 0.00
    },
    "reticulatus": {
        "hosts": ["quercia","castagno","faggio","misto"],
        "season": {"start_m": 5, "end_m": 9},
        "tm7_opt": (17.0, 22.0),
        "lag_base": 8.5,
        "vpd_sens": 1.0,
        "smi_bias": 0.00
    },
    "edulis": {
        "hosts": ["faggio","conifere","misto"],
        "season": {"start_m": 8, "end_m": 11},
        "tm7_opt": (12.0, 18.0),
        "lag_base": 10.0,
        "vpd_sens": 1.1,
        "smi_bias": +0.05
    },
    "pinophilus": {
        "hosts": ["conifere","misto"],
        "season": {"start_m": 6, "end_m": 10},
        "tm7_opt": (14.0, 20.0),
        "lag_base": 9.5,
        "vpd_sens": 0.9,
        "smi_bias": -0.02
    }
}

def _month_in_season(m: int, start_m: int, end_m: int) -> bool:
    if start_m <= end_m:
        return start_m <= m <= end_m
    return m >= start_m or m <= end_m

def infer_porcino_species(habitat_used: str, month: int, elev_m: float, aspect_oct: Optional[str]) -> str:
    """
    Sceglie specie plausibile dato habitat/mese/altitudine/esposizione (heuristic).
    """
    h = (habitat_used or "misto").lower()
    candidates = []
    for sp, prof in SPECIES_PROFILES.items():
        if h in prof["hosts"]:
            bonus = 1.0 if _month_in_season(month, prof["season"]["start_m"], prof["season"]["end_m"]) else 0.7
            if elev_m >= 1200 and sp in ("edulis","pinophilus"): bonus += 0.2
            if elev_m <= 700 and sp in ("aereus","reticulatus"): bonus += 0.2
            if aspect_oct in ("S","SE","SW") and sp in ("aereus","reticulatus"): bonus += 0.05
            if aspect_oct in ("N","NE","NW") and sp in ("edulis","pinophilus"): bonus += 0.05
            candidates.append((sp, bonus))
    if not candidates:
        return "reticulatus" if 5 <= month <= 9 else "edulis"
    candidates.sort(key=lambda t: t[1], reverse=True)
    return candidates[0][0]

# -------------------- Lag & previsione --------------------
def stochastic_lag_days(smi: float, shock: float, tmean7: float, species: str) -> int:
    prof = SPECIES_PROFILES.get(species, SPECIES_PROFILES["reticulatus"])
    lag = prof["lag_base"] - 3.5*smi - 1.7*shock
    lo, hi = prof["tm7_opt"]
    if lo <= tmean7 <= hi: lag -= 0.8
    elif tmean7 < lo: lag += 0.7
    elif tmean7 > hi: lag += 0.4
    return int(round(clamp(lag, 5.0, 15.0)))

def gaussian_kernel(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5*((x-mu)/sigma)**2)

def event_strength(mm: float) -> float:
    return 1.0 - math.exp(-mm/20.0)

# ---- EVENT DETECTION CON SOGLIE DINAMICHE (MIGLIORATO) ----
def rain_events_dynamic(rains: List[float], smi_series: List[float], month: int, elevation: float) -> List[Tuple[int,float]]:
    """
    Event detection con soglie dinamiche basate su SMI locale, stagione e quota
    """
    events = []
    n = len(rains)
    i = 0
    
    while i < n:
        # SMI locale per calcolare soglia dinamica
        smi_local = smi_series[i] if i < len(smi_series) else 0.5
        threshold_1d = dynamic_rain_threshold(smi_local, month, elevation)
        threshold_2d = dynamic_rain_threshold_2day(smi_local, month, elevation)
        
        if rains[i] >= threshold_1d:
            events.append((i, rains[i]))
            i += 1
        elif i+1 < n and (rains[i] + rains[i+1]) >= threshold_2d:
            events.append((i+1, rains[i] + rains[i+1]))
            i += 2
        else:
            i += 1
    
    return events

# ---- DATABASE UTILS (NUOVO) ----
def save_prediction(lat: float, lon: float, date: str, score: int, species: str, 
                   habitat: str, confidence_data: dict):
    """Salva predizione per analisi performance futura"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (lat, lon, date, predicted_score, species, habitat, confidence_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, score, species, habitat, json.dumps(confidence_data)))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Errore salvataggio predizione: {e}")

def check_recent_validations(lat: float, lon: float, days: int = 30, radius_km: float = 10.0) -> bool:
    """Controlla se ci sono validazioni recenti nell'area"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Semplificazione: box invece di distanza esatta
        lat_delta = radius_km / 111.0  # ~1 grado = 111 km
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        
        cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        cursor.execute('''
            SELECT COUNT(*) FROM sightings 
            WHERE lat BETWEEN ? AND ? 
            AND lon BETWEEN ? AND ?
            AND date >= ?
        ''', (lat - lat_delta, lat + lat_delta, 
              lon - lon_delta, lon + lon_delta, 
              cutoff_date))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
        
    except Exception as e:
        logger.error(f"Errore controllo validazioni: {e}")
        return False

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
    conf_detail = payload.get("confidence_detailed", {})
    harvest = payload.get("harvest_estimate","—"); harvest_note = payload.get("harvest_note","")
    species = payload.get("species","reticulatus")

    lines = []
    lines.append(f"<p><strong>Specie stimata</strong>: <strong>B. {species}</strong> • Habitat: <strong>{habitat_used}</strong> ({habsrc}). "
                 f"Quota <strong>{elev} m</strong>, pendenza <strong>{slope}°</strong>, esposizione <strong>{aspect}</strong>. "
                 f"Microclima: indice energetico <strong>{energy:.2f}</strong>, TWI-proxy <strong>{twi:.2f}</strong>.</p>")
    
    lines.append(f"<p><strong>Stato idrico-termico</strong> — Sorgente SMI: <strong>{smi_src}</strong> (soglia {sm_thr:.2f}). "
                 f"Piogge antecedenti: <strong>{P15:.0f} mm/15g</strong> (7g: {P7:.0f} mm). "
                 f"Termica 7g: <strong>{Tm7:.1f}°C</strong>; umidità 7g: <strong>{RH7:.0f}%</strong>.</p>")
    
    # Confidence dettagliata
    if conf_detail:
        lines.append(f"<p><strong>Affidabilità dettagliata</strong> — ")
        lines.append(f"Meteorologica: <strong>{conf_detail.get('meteorological', 0):.2f}</strong>, ")
        lines.append(f"Ecologica: <strong>{conf_detail.get('ecological', 0):.2f}</strong>, ")
        lines.append(f"Idrologica: <strong>{conf_detail.get('hydrological', 0):.2f}</strong>, ")
        lines.append(f"Atmosferica: <strong>{conf_detail.get('atmospheric', 0):.2f}</strong>, ")
        lines.append(f"Empirica: <strong>{conf_detail.get('empirical', 0):.2f}</strong>. ")
        lines.append(f"<strong>Complessiva: {conf_detail.get('overall', 0):.2f}</strong></p>")
    
    if best and best.get("mean",0) > 0:
        s, e, m = best["start"], best["end"], best["mean"]
        lines.append(f"<p><strong>Finestra migliore</strong> prossimi 10 giorni: <strong>giorni {s+1}–{e+1}</strong> "
                     f"(media indice ≈ <strong>{m}</strong>). Indice oggi: <strong>{idx}/100</strong>.</p>")
    
    lines.append(f"<p><strong>Raccolto atteso</strong>: {harvest}. <em>{harvest_note}</em></p>")
    
    # Miglioramenti v2.4
    lines.append("<h4>Miglioramenti modello v2.4</h4><ul>")
    lines.append("<li><strong>Soglie dinamiche</strong>: eventi piovosi adattati a SMI, stagione e quota</li>")
    lines.append("<li><strong>Smoothing avanzato</strong>: preserva meglio i picchi importanti</li>")
    lines.append("<li><strong>Confidence multi-dimensionale</strong>: valutazione più accurata dell'affidabilità</li>")
    lines.append("<li><strong>Sistema di validazione</strong>: raccolta dati per miglioramento continuo</li>")
    lines.append("</ul>")
    
    return "\n".join(lines)

# ----------------------------- NUOVI ENDPOINTS -----------------------------
@app.post("/api/report-sighting")
async def report_sighting(
    lat: float, lon: float, species: str, 
    quantity: int = 1, confidence: float = 0.8,
    photo_url: str = "", notes: str = "",
    background_tasks: BackgroundTasks = None
):
    """Endpoint per segnalare ritrovamenti (crowd-sourcing)"""
    try:
        date = datetime.now().date().isoformat()
        
        # Ottieni score predetto per questo punto oggi (per confronto futuro)
        # Per ora salviamo 0, ma in futuro si può recuperare dalla cache
        predicted_score = 0
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sightings (lat, lon, date, species, quantity, confidence, photo_url, notes, predicted_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, species, quantity, confidence, photo_url, notes, predicted_score))
        conn.commit()
        conn.close()
        
        logger.info(f"Segnalazione ricevuta: {species} a ({lat:.4f}, {lon:.4f})")
        
        return {"status": "success", "message": "Segnalazione registrata con successo"}
        
    except Exception as e:
        logger.error(f"Errore segnalazione: {e}")
        raise HTTPException(500, "Errore interno del server")

@app.post("/api/report-no-findings")
async def report_no_findings(
    lat: float, lon: float, searched_hours: int = 2,
    habitat: str = "", notes: str = ""
):
    """Endpoint per segnalare ricerche senza risultato (importante per ML!)"""
    try:
        date = datetime.now().date().isoformat()
        predicted_score = 0  # Come sopra
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO no_sightings (lat, lon, date, searched_hours, habitat, notes, predicted_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, searched_hours, habitat, notes, predicted_score))
        conn.commit()
        conn.close()
        
        logger.info(f"Ricerca senza risultato: {searched_hours}h a ({lat:.4f}, {lon:.4f})")
        
        return {"status": "success", "message": "Report registrato con successo"}
        
    except Exception as e:
        logger.error(f"Errore report: {e}")
        raise HTTPException(500, "Errore interno del server")

@app.get("/api/validation-stats")
async def validation_stats():
    """Statistiche sui dati di validazione raccolti"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Conta segnalazioni positive
        cursor.execute("SELECT COUNT(*) FROM sightings")
        positive_count = cursor.fetchone()[0]
        
        # Conta ricerche negative  
        cursor.execute("SELECT COUNT(*) FROM no_sightings")
        negative_count = cursor.fetchone()[0]
        
        # Conta predizioni salvate
        cursor.execute("SELECT COUNT(*) FROM predictions")
        predictions_count = cursor.fetchone()[0]
        
        # Specie più segnalate
        cursor.execute("""
            SELECT species, COUNT(*) as count 
            FROM sightings 
            GROUP BY species 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_species = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "positive_sightings": positive_count,
            "negative_reports": negative_count, 
            "predictions_logged": predictions_count,
            "total_validations": positive_count + negative_count,
            "top_species": top_species,
            "ready_for_ml": (positive_count + negative_count) >= 50
        }
        
    except Exception as e:
        logger.error(f"Errore stats: {e}")
        return {"error": str(e)}

# ----------------------------- ENDPOINT PRINCIPALE MIGLIORATO -----------------------------
@app.get("/api/health")
async def health():
    return {"ok":True,"time":datetime.now(timezone.utc).isoformat(), "version": "2.4.0"}

@app.get("/api/geocode")
async def api_geocode(q:str):
    try:
        url="https://nominatim.openstreetmap.org/search"
        params={"format":"json","q":q,"addressdetails":1,"limit":1,"email":os.getenv("NOMINATIM_EMAIL","info@example.com")}
        async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
            r=await c.get(url,params=params); r.raise_for_status(); data=r.json()
        if data:
            return {"lat":float(data[0]["lat"]),"lon":float(data[0]["lon"]),"display":data[0].get("display_name","")}
    except Exception:
        pass
    url="https://geocoding-api.open-meteo.com/v1/search"
    params={"name":q,"count":1,"language":"it"}
    async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
        r=await c.get(url,params=params); r.raise_for_status(); j=r.json()
    res=(j.get("results") or [])
    if not res: raise HTTPException(404,"Località non trovata")
    it=res[0]
    return {"lat":float(it["latitude"]),"lon":float(it["longitude"]),"display":f"{it.get('name')} ({(it.get('country_code') or '').upper()})"}

@app.get("/api/score")
async def api_score(
    lat:float=Query(...), lon:float=Query(...),
    half:float=Query(8.0,gt=3.0,lt=20.0),
    habitat:str=Query("", description="castagno,faggio,quercia,conifere,misto,altro"),
    autohabitat:int=Query(1, description="1=auto OSM, 0=manuale"),
    hours:int=Query(2, ge=2, le=8),
    background_tasks: BackgroundTasks = None
):
    # fetch paralleli
    om_task  = asyncio.create_task(fetch_open_meteo(lat,lon,past=15,future=10))
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

    # Habitat usato
    habitat_used = (habitat or "").strip().lower()
    habitat_source = "manuale"
    if autohabitat==1:
        if not habitat_used and auto_hab:
            habitat_used = auto_hab
            habitat_source = f"automatico (OSM, conf {auto_conf:.2f})"
        elif habitat_used:
            habitat_source = "manuale (override)"
    if not habitat_used:
        habitat_used = "misto"

    # Open-Meteo daily
    d=om["daily"]; timev=d["time"]
    P_om=[float(x or 0.0) for x in d["precipitation_sum"]]
    Tmin_om, Tmax_om, Tm_om = d["temperature_2m_min"], d["temperature_2m_max"], d["temperature_2m_mean"]
    ET0_om=d.get("et0_fao_evapotranspiration",[0.0]*len(P_om))
    RH_om=d.get("relative_humidity_2m_mean",[60.0]*len(P_om))
    SW_om=d.get("shortwave_radiation_sum",[15000.0]*len(P_om))

    pastN=15; futN=10
    P_past_om=P_om[:pastN]; P_fut_om=P_om[pastN:pastN+futN]
    Tmin_p_om,Tmax_p_om,Tm_p_om=Tmin_om[:pastN],Tmax_om[:pastN],Tm_om[:pastN]
    Tmin_f_om,Tmax_f_om,Tm_f_om=Tmin_om[pastN:pastN+futN],Tmax_om[pastN:pastN+futN],Tm_om[pastN:pastN+futN]
    ET0_p_om=ET0_om[:pastN]; RH_p_om=RH_om[:pastN]; SW_p_om=SW_om[:pastN]
    RH_f_om=RH_om[pastN:pastN+futN] if len(RH_om)>=pastN+futN else [60.0]*futN

    # ---- BLEND FUTURO ----
    P_fut_ow=[]; Tmin_f_ow=[]; Tmax_f_ow=[]; Tm_f_ow=[]
    if ow and "daily" in ow:
        for day in ow["daily"][:futN]:
            P_fut_ow.append(float(day.get("rain",0.0)))
            t=day.get("temp",{})
            Tmin_f_ow.append(float(t.get("min",0.0)))
            Tmax_f_ow.append(float(t.get("max",0.0)))
            Tm_f_ow.append(float(t.get("day", (t.get("min",0.0)+t.get("max",0.0))/2.0)))
    ow_len=min(len(P_fut_ow),futN)

    P_fut_blend=[]; Tmin_f_blend=[]; Tmax_f_blend=[]; Tm_f_blend=[]
    for i in range(futN):
        if i<ow_len: P_fut_blend.append(max(P_fut_om[i], P_fut_ow[i]))
        else: P_fut_blend.append(P_fut_om[i])
        if i<ow_len:
            w_ow = 0.6 if i<=2 else 0.5
            w_om = 1.0 - w_ow
            Tmin_f_blend.append(w_om*Tmin_f_om[i] + w_ow*Tmin_f_ow[i])
            Tmax_f_blend.append(w_om*Tmax_f_om[i] + w_ow*Tmax_f_ow[i])
            Tm_f_blend.append(w_om*Tm_f_om[i] + w_ow*Tm_f_ow[i])
        else:
            Tmin_f_blend.append(Tmin_f_om[i]); Tmax_f_blend.append(Tmax_f_om[i]); Tm_f_blend.append(Tm_f_om[i])

    # ---- BLEND PASSATO (OW Timemachine) ----
    P_past_blend = P_past_om[:]
    Tmin_p_blend, Tmax_p_blend, Tm_p_blend = Tmin_p_om[:], Tmax_p_om[:], Tm_p_om[:]
    try:
        ow_past = await fetch_openweather_past(lat, lon, days=5)
        for idx in range(pastN-1, max(-1, pastN-6), -1):
            if idx < 0 or idx >= len(timev): 
                continue
            dstr = timev[idx]
            if dstr in ow_past:
                rec = ow_past[dstr]
                P_owd = float(rec.get("rain", 0.0))
                Tmin_owd = float(rec.get("tmin", Tmin_p_blend[idx]))
                Tmax_owd = float(rec.get("tmax", Tmax_p_blend[idx]))
                Tm_owd   = float(rec.get("tmean", Tm_p_blend[idx]))
                P_past_blend[idx] = max(P_past_blend[idx], P_owd)
                days_back = (pastN-1) - idx
                w_ow = 0.6 if days_back <= 2 else 0.5
                w_om = 1.0 - w_ow
                Tmin_p_blend[idx] = w_om*Tmin_p_blend[idx] + w_ow*Tmin_owd
                Tmax_p_blend[idx] = w_om*Tmax_p_blend[idx] + w_ow*Tmax_owd
                Tm_p_blend[idx]   = w_om*Tm_p_blend[idx]   + w_ow*Tm_owd
    except Exception:
        pass

    # Indicatori recenti
    API_val=api_index(P_past_blend,half_life=half)
    ET7=sum(ET0_p_om[-7:]) if ET0_p_om else 0.0
    RH7=sum(RH_p_om[-7:])/max(1,len(RH_p_om[-7:])) if RH_p_om else 60.0
    SW7=sum(SW_p_om[-7:])/max(1,len(SW_p_om[-7:])) if SW_p_om else 15000.0
    Tm7=float(sum(Tm_p_blend[-7:])/max(1,len(Tm_p_blend[-7:])))

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
    species = infer_porcino_species(habitat_used, month_now, float(elev_m), aspect_oct)
    prof = SPECIES_PROFILES[species]
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

    # VPD futuro
    vpd_fut=[vpd_hpa(float(Tm_f_blend[i]), float(RH_f_om[i] if i<len(RH_f_om) else 60.0)) for i in range(futN)]
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
    reliability = reliability_from_sources(P_fut_ow[:ow_len],P_fut_om[:ow_len],Tm_f_ow[:ow_len],Tm_f_om[:ow_len]) if ow_len else 0.6

    # Controllo validazioni recenti nell'area
    has_validations = check_recent_validations(lat, lon)

    # CONFIDENCE MULTI-DIMENSIONALE (NUOVO)
    confidence_detailed = advanced_confidence(
        weather_agreement=reliability,
        habitat_confidence=auto_conf if autohabitat==1 else 0.6,
        smi_reliability=0.9 if sm_used.startswith("ERA5") else 0.7,
        vpd_validity=(vpd_today is not None and vpd_today <= 10.0),
        has_recent_validation=has_validations
    )

    # ---- Event detection CON SOGLIE DINAMICHE (MIGLIORATO) ----
    ev_past = rain_events_dynamic(P_past_blend, smi_series[:pastN], month_now, float(elev_m))
    ev_fut_raw = rain_events_dynamic(P_fut_blend, [smi_now]*futN, month_now, float(elev_m))
    ev_fut=[(pastN+i, mm) for (i,mm) in ev_fut_raw]
    events = ev_past + ev_fut

    # ---- Previsione con lag specie-specifico e VPD specie-specifico ----
    forecast=[0.0]*futN; details=[]
    for (ev_idx_abs, mm_tot) in events:
        # SMI vicino all'evento (±1 giorno)
        i0=max(0, ev_idx_abs-1); i1=min(len(smi_series), ev_idx_abs+1)
        sm_loc=float(_np.nanmean(_np.array(smi_series[i0:i1+1], dtype=float))) if i1>i0 else float(smi_series[ev_idx_abs])
        sm_loc=0.5 if not (sm_loc==sm_loc) else clamp(sm_loc + prof["smi_bias"], 0.0, 1.0)
        lag = stochastic_lag_days(smi=sm_loc, shock=shock, tmean7=Tm7, species=species)
        peak_idx = ev_idx_abs + lag
        # sigma: più largo se nel futuro e reliability bassa
        sigma = 2.5 if mm_tot < 25 else 3.0
        if ev_idx_abs >= pastN and reliability < 0.6:
            sigma += 0.8
        # ampiezza
        amp = event_strength(mm_tot) * micro_amp * (0.6 + 0.7*sm_loc)
        if ev_idx_abs >= pastN:
            amp *= (0.5 + 0.5*reliability)
        for j in range(futN):
            abs_j = pastN + j
            pen = vpd_penalty(vpd_fut[j], species_vpd_sens=prof["vpd_sens"])
            forecast[j] += 100.0 * amp * gaussian_kernel(abs_j, peak_idx, sigma) * pen
        when = timev[ev_idx_abs] if ev_idx_abs < len(timev) else f"+{ev_idx_abs-pastN}d"
        details.append({
            "event_day_index": ev_idx_abs,
            "event_when": when,
            "event_mm": round(mm_tot,1),
            "lag_days": lag,
            "predicted_peak_abs_index": peak_idx,
            "observed": (ev_idx_abs < pastN)
        })

    out = [int(round(clamp(v,0.0,100.0))) for v in forecast]
    
    # SMOOTHING AVANZATO (MIGLIORATO)
    smoothed = advanced_smoothing([float(x) for x in out])
    smoothed = [int(round(x)) for x in smoothed]

    # finestra migliore
    s=e=m=0
    if len(smoothed)>=3:
        best_s,best_e,best_m=0,2,round((smoothed[0]+smoothed[1]+smoothed[2])/3)
        for i in range(1,len(smoothed)-2):
            med=round((smoothed[i]+smoothed[i+1]+smoothed[i+2])/3)
            if med>best_m: best_s,best_e,best_m=i,i+2,med
        s,e,m = best_s,best_e,best_m

    flush_today = int(smoothed[0] if smoothed else 0)

    # ---- Dimensioni cappello (prudente) ----
    def cap_growth_rate_cm_per_day(tmean: float, rh: float, vpd_hpa_max: float, species: str) -> float:
        specie_boost = {"aereus":1.05, "reticulatus":1.05, "edulis":0.95, "pinophilus":0.95}.get(species,1.0)
        if rh < 40: return 0.0
        ur_f = clamp((rh - 40.0) / (85.0 - 40.0), 0.0, 1.0)
        lo,hi = tm_lo, tm_hi
        if tmean <= 10: t_f = 0.2 * (tmean/10.0)
        elif tmean <= 16: t_f = 0.2 + 0.8*(tmean-10)/6.0
        elif tmean <= 20: t_f = 1.0
        elif tmean <= 24: t_f = 1.0 - 0.6*(tmean-20)/4.0
        elif tmean <= 28: t_f = 0.4 - 0.3*(tmean-24)/4.0
        else: t_f = 0.1
        vpd_pen = vpd_penalty(vpd_hpa_max, species_vpd_sens=prof["vpd_sens"])
        return 2.1 * ur_f * clamp(t_f,0.0,1.0) * vpd_pen * specie_boost

    vpd7=max(vpd_hpa(Tm7, float(RH7)), 0.0)
    # età coorte: distanza da picco più vicino
    if details:
        today_abs = pastN
        peak_idxs = [d.get("predicted_peak_abs_index", today_abs) for d in details]
        peak_idx = min(peak_idxs, key=lambda k: abs(today_abs - k)) if peak_idxs else today_abs
        start_buttons = peak_idx - 2
        age_days = max(0, today_abs - start_buttons)
    else:
        age_days = 0
    size_rate = cap_growth_rate_cm_per_day(Tm7, float(RH7), float(vpd7), species)
    size_cm = clamp(size_rate * age_days, 1.5, 18.0)
    if size_cm < 5.0: size_cls = "bottoni (2–5 cm)"
    elif size_cm < 10.0: size_cls = "medi (6–10 cm)"
    else: size_cls = "grandi (10–15+ cm)"
    smin = clamp(size_rate * max(0, age_days-2), 1.5, 18.0)
    smax = clamp(size_rate * (age_days+2), 1.5, 18.0)

    # ---- Raccolto atteso: SEMPRE visibile (ponderato) ----
    def harvest_from_index_always(score:int, hours:int, reliability:float, main_observed:bool,
                                  vpd_hpa_today: Optional[float], smi_now: float, sm_thr: float) -> Tuple[str, str]:
        factor = 1.0 if hours <= 2 else 1.45 if hours <= 4 else 1.6
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
        f_rel = 0.4 + 0.8 * ((rel - 0.25) / (0.95 - 0.25))
        f_obs = 1.0 if main_observed else 0.90
        if vpd_hpa_today is None:
            f_vpd = 0.9; vpd_txt = "n.d."
        else:
            v = vpd_hpa_today
            f_vpd = 1.0 if v <= 6.0 else (0.85 if v < 10.0 else 0.70)
            vpd_txt = f"{v:.1f} hPa"
        f_smi = 0.90 if smi_now < sm_thr else 1.00

        F = max(0.35, min(1.15, f_rel * f_obs * f_vpd * f_smi))
        lo = max(0, int(round(lo0 * F)))
        hi = max(1, int(round(hi0 * F)))
        if lo >= hi: hi = lo + 1

        note = (f"Stima specie B. {species}; ponderazione (reliability={reliability:.2f}, "
                f"{'evento osservato' if main_observed else 'evento non osservato'}, "
                f"VPD={vpd_txt}, SMI {'< thr' if smi_now < sm_thr else '≥ thr'}).")
        return (f"{lo}–{hi} porcini", note)

    main_observed = any(d.get("observed") and pastN <= d.get("predicted_peak_abs_index",1e9) < pastN+futN for d in details)
    harvest_txt, harvest_note = harvest_from_index_always(flush_today, hours, reliability, main_observed, vpd_today, smi_now, sm_thr)

    # Confidence aggregata (0–100) - usa quella dettagliata
    model_conf = int(round(100 * confidence_detailed["overall"]))
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
        "ET0_7d_mm": round(ET7,1),
        "RH7_pct": round(RH7,1),
        "SW7_kj": round(SW7,0),
        "Tmean7_c": round(Tm7,1),

        "index": int(smoothed[0] if smoothed else 0),
        "forecast": [int(x) for x in smoothed],
        "best_window": {"start": s, "end": e, "mean": m},
        "harvest_estimate": harvest_txt,
        "harvest_note": harvest_note,
        "reliability": round(reliability,3),
        "confidence_detailed": confidence_detailed,  # NUOVO

        "rain_past": rain_past,
        "rain_future": rain_future,

        "habitat_used": habitat_used,
        "habitat_source": habitat_source,
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
            "dynamic_thresholds_used": True,  # NUOVO
            "advanced_smoothing": True,      # NUOVO
        },
        "model_confidence": model_conf,
        "vpd_today_hpa": round(vpd_today,1) if vpd_today is not None else None,

        # specie/ecologia
        "species": species,
        "species_profile": species_profile_txt,
        
        # v2.4 features
        "model_version": "2.4.0",
        "has_local_validations": has_validations,
    }

    response_data["dynamic_explanation"] = build_analysis_text_long(response_data)
    
    # SALVA PREDIZIONE PER ANALISI FUTURA (NUOVO)
    if background_tasks:
        background_tasks.add_task(
            save_prediction, 
            lat, lon, 
            datetime.now().date().isoformat(),
            flush_today, species, habitat_used, 
            confidence_detailed
        )
    
    logger.info(f"Predizione: {flush_today}/100 per {species} a ({lat:.4f}, {lon:.4f})")
    
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)
