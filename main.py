# main.py — Trova Porcini API (v2.0.0)
# Compatibile al 100% con la tua UI (endpoint e JSON identici alla v1.8.2)
# Novità (implementa quanto richiesto):
#  • VPD (deficit di pressione di vapore) → penalità/gating su flush e crescita
#  • ΔTmin_3d (cold‑shock) → accorcia il lag se c'è raffreddamento brusco
#  • TWI‑proxy + energy index (esposizione/pendenza) → moltiplicatore dell’ampiezza
#  • SMI (indice umidità suolo) stimato con P‑ET0 (subito) + prefetch ERA5‑Land via CDS (in background, cache)
#  • Blend Open‑Meteo + OpenWeather invariato (T pesata, P = max)
#  • Geocoding robusto (Nominatim → fallback Open‑Meteo)
#  • DEM multiscala con cache, habitat auto OSM invariati
#
# Nota su ERA5‑Land (CDS): i download possono richiedere minuti (coda).
# Per non bloccare l’API, qui effettuiamo un *prefetch asincrono* in background (se presenti `CDS_API_KEY`/`cdsapi`).
# Nell’immediato l’SMI usa P‑ET0; appena il dato ERA5‑Land è disponibile, verrà usato in richieste successive.
#
# Requisiti minimi: fastapi, httpx, uvicorn
# Facoltativi per ERA5‑Land: cdsapi, netCDF4 (se assenti → fallback automatico)

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio, tempfile, time
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta

app = FastAPI(title="Trova Porcini API (v2.0.0)", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

HEADERS = {"User-Agent":"Trovaporcini/2.0.0 (+site)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")
CDS_API_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")  # formato: UID:KEY

# ----------------------------- UTIL -----------------------------
def clamp(v,a,b): return a if v<a else b if v>b else v

def half_life_coeff(days: float) -> float: return 0.5**(1.0/max(1.0,days))

def api_index(precip: List[float], half_life: float=8.0) -> float:
    k=half_life_coeff(half_life); api=0.0
    for p in precip: api=k*api+(p or 0.0)
    return api

def stddev(xs: List[float]) -> float:
    if not xs: return 0.0
    m=sum(xs)/len(xs)
    return (sum((x-m)**2 for x in xs)/len(xs))**0.5

# ---- VPD & fisica semplificata ----
def saturation_vapor_pressure_hpa(Tc: float) -> float:
    return 6.112 * math.exp((17.67 * Tc) / (Tc + 243.5))

def vpd_hpa(Tc: float, RH: float) -> float:
    RHc = clamp(RH, 0.0, 100.0)
    return saturation_vapor_pressure_hpa(Tc) * (1.0 - RHc/100.0)

def vpd_penalty(vpd_max_hpa: float) -> float:
    # <6 hPa nessuna penalità; >12 hPa forte malus
    if vpd_max_hpa <= 6.0: return 1.0
    if vpd_max_hpa >= 12.0: return 0.4
    return 1.0 - 0.6 * (vpd_max_hpa - 6.0) / 6.0

# ---- Shock termico ΔTmin(3d) ----
def cold_shock_from_tmin_series(tmin: List[float]) -> float:
    if len(tmin) < 7: return 0.0
    last3 = sum(tmin[-3:]) / 3.0
    prev3 = sum(tmin[-6:-3]) / 3.0
    drop = last3 - prev3
    if drop >= -1.0: return 0.0
    return clamp((-drop - 1.0) / 3.0, 0.0, 1.0)  # 0…1

# ---- TWI proxy + energy index ----
def twi_proxy_from_slope_concavity(slope_deg: float, concavity: float) -> float:
    # concavity >0 → aree d'accumulo; slope piccolo → TWI grande
    beta = max(0.1, math.radians(max(0.1, slope_deg)))
    tanb = max(0.05, math.tan(beta))
    conc = max(0.0, concavity + 0.02)  # shift leggero
    twi = math.log(1.0 + 6.0 * conc) - math.log(tanb)
    # normalizza ~0..1
    return clamp((twi + 2.2) / 4.0, 0.0, 1.0)

# microclima stagionale da esposizione/pendenza

def microclimate_energy(aspect_oct: Optional[str], slope_deg: float, month: int) -> float:
    if not aspect_oct or slope_deg < 0.8: return 0.5
    summer = 1.0 if month in (7,8,9) else 0.6
    base = 0.5
    if aspect_oct in ("N","NE","NW"): base += 0.15
    if aspect_oct in ("S","SE","SW"): base -= 0.12 * summer
    base *= (1.0 + min(0.15, slope_deg/90.0))
    return clamp(base, 0.25, 0.9)

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

# ----------------------- Geocoding robusto -----------------------
async def geocode_nominatim(q:str)->Optional[Dict[str,Any]]:
    url="https://nominatim.openstreetmap.org/search"
    params={"format":"json","q":q,"addressdetails":1,"limit":1,"email":os.getenv("NOMINATIM_EMAIL","info@example.com")}
    async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
        r=await c.get(url,params=params); r.raise_for_status(); data=r.json()
    if not data: return None
    return {"lat":float(data[0]["lat"]),"lon":float(data[0]["lon"]),"display":data[0].get("display_name","")}

async def geocode_openmeteo(q:str)->Optional[Dict[str,Any]]:
    url="https://geocoding-api.open-meteo.com/v1/search"
    params={"name":q,"count":1,"language":"it"}
    async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
        r=await c.get(url,params=params); r.raise_for_status(); j=r.json()
    res=(j.get("results") or [])
    if not res: return None
    it=res[0]
    return {"lat":float(it["latitude"]),"lon":float(it["longitude"]),"display":f"{it.get('name')} ({(it.get('country_code') or '').upper()})"}

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
        dlat,dlon=step_m*deg_lat,step_m*deg_lon
        coords=[{"latitude":lat+dr*dlat,"longitude":lon+dc*dlon} for dr in(-1,0,1) for dc in(-1,0,1)]
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
    dzdx=((z[0][2]+2*z[1][2]+z[2][2])-(z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_size_m)  # +x = EST
    dzdy=((z[2][0]+2*z[2][1]+z[2][2])-(z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_size_m)  # +y = SUD
    slope=math.degrees(math.atan(math.hypot(dzdx,dzdy)))
    aspect=(math.degrees(math.atan2(-dzdx, dzdy))+360.0)%360.0  # 0°=N, orario
    octs=["N","NE","E","SE","S","SW","W","NW","N"]
    octant=octs[int(((aspect%360)+22.5)//45)]
    return round(slope,1),round(aspect,0),octant

def concavity_from_grid(z:List[List[float]])->float:
    center=z[1][1]; neigh=[z[r][c] for r in (0,1,2) for c in (0,1,2) if not (r==1 and c==1)]
    delta=(sum(neigh)/8.0 - center)  # m; positivo = concavo (accumulo)
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

# -------------------- Lag & previsione (v2) --------------------

def stochastic_lag_days(smi: float, shock: float, tmean7: float) -> int:
    # base 10 giorni, accorciato con SMI alto e shock; modulato da T7
    lag = 10.0 - 4.0*smi - 2.0*shock
    if 16 <= tmean7 <= 20: lag -= 1.0
    elif tmean7 < 12: lag += 1.0
    elif tmean7 > 22: lag += 0.5
    return int(round(clamp(lag, 5.0, 15.0)))

def gaussian_kernel(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5*((x-mu)/sigma)**2)

def event_strength(mm: float) -> float:
    return 1.0 - math.exp(-mm/20.0)

# ----------------------- Specie & safety -----------------------
from typing import TypedDict
class FlushDetail(TypedDict, total=False):
    event_day_index: int
    event_when: str
    event_mm: float
    lag_days: int
    predicted_peak_abs_index: int

# --------------- Modello dimensionale (diametro) ---------------
def cap_growth_rate_cm_per_day(tmean: float, rh: float, vpd_hpa_max: float) -> float:
    if rh < 40: return 0.0
    ur_f = clamp((rh - 40.0) / (85.0 - 40.0), 0.0, 1.0)
    if tmean <= 10: t_f = 0.2 * (tmean/10.0)
    elif tmean <= 16: t_f = 0.2 + 0.8*(tmean-10)/6.0
    elif tmean <= 20: t_f = 1.0
    elif tmean <= 24: t_f = 1.0 - 0.6*(tmean-20)/4.0
    elif tmean <= 28: t_f = 0.4 - 0.3*(tmean-24)/4.0
    else: t_f = 0.1
    vpd_pen = vpd_penalty(vpd_hpa_max)
    return 2.1 * ur_f * clamp(t_f,0.0,1.0) * vpd_pen

# ---------------- Spiegazione (flush + dimensioni) ----------------
def build_analysis_text(payload: Dict[str,Any]) -> str:
    idx = payload["index"]
    best = payload["best_window"]
    p15 = payload["P15_mm"]; p7 = payload["P7_mm"]
    rh7 = payload.get("RH7_pct", None); sw7 = payload.get("SW7_kj", None)
    tm7 = payload["Tmean7_c"]
    aspect = payload.get("aspect_octant") or "NA"
    slope = payload.get("slope_deg"); elev = payload.get("elevation_m")
    habitat_used = (payload.get("habitat_used") or "").capitalize() or "Altro"
    habitat_source = payload.get("habitat_source","manuale")
    size_cm = payload.get("size_cm", 0.0)
    size_class = payload.get("size_class", "—")
    rng = payload.get("size_range_cm",[0.0,0.0])

    out=[]
    out.append(f"<p><strong>Indice attuale (flush): {idx}/100</strong> • Habitat: <strong>{habitat_used}</strong> ({habitat_source}). "
               f"Quota <strong>{elev} m</strong>, pendenza <strong>{slope}°</strong>, esposizione <strong>{aspect or 'NA'}</strong>.</p>")

    if size_cm > 0:
        note_range = ""
        if rng and len(rng)==2 and rng[1] > 0:
            note_range = f" Range atteso oggi: <strong>{rng[0]}–{rng[1]} cm</strong> (coorti di età diverse durante il flush)."
        out.append(f"<p>Per oggi, la <strong>taglia media stimata</strong> dei cappelli è ~<strong>{size_cm} cm</strong> "
                   f"({size_class}).{note_range}</p>")

    out.append("<h4>Come stimiamo i giorni di uscita</h4>")
    out.append("<p>Il modello usa <strong>SMI</strong> (umidità del suolo), <strong>ΔTmin</strong> (shock termico) e <strong>VPD</strong> "
               "per modulare l'innesco e il ritardo; gli <strong>eventi di pioggia</strong> attivano un picco che viene spostato "
               "in avanti di un <strong>lag stocastico</strong> (5–15 giorni) e scalato con indice energetico da esposizione/pendenza e TWI‑proxy.</p>")

    evs = payload.get("flush_events", [])
    if evs:
        out.append("<h4>Eventi e finestre stimate</h4><ul>")
        for e in evs:
            when = e.get("event_when"); mm = e.get("event_mm"); lag = e.get("lag_days")
            out.append(f"<li>Evento ~<strong>{mm} mm</strong> il <strong>{when}</strong> → "
                       f"flush atteso ~<strong>{lag} giorni</strong> dopo.</li>")
        out.append("</ul>")

    if best and best.get("mean",0)>0:
        s,e,m = best["start"], best["end"], best["mean"]
        out.append(f"<p>Nei prossimi 10 giorni la finestra con <strong>maggiore probabilità</strong> è tra "
                   f"<strong>giorno {s+1}</strong> e <strong>giorno {e+1}</strong> (media ≈ <strong>{m}</strong>).</p>")

    cause=[]
    cause.append(f"Antecedente: 15 gg = <strong>{p15:.0f} mm</strong> (7 gg: {p7:.0f} mm).")
    cause.append(f"Termica media 7 gg: <strong>{tm7:.1f}°C</strong>.")
    out.append("<h4>Contesto meteo‑microclimatico</h4><ul>" + "".join(f"<li>{c}</li>" for c in cause) + "</ul>")

    return "\n".join(out)

# --------------- Affidabilità ----------------

def reliability_from_sources(P_ow:List[float], P_om:List[float], T_ow:List[float], T_om:List[float]) -> float:
    n=min(len(P_ow),len(P_om),len(T_ow),len(T_om))
    if n==0: return 0.6
    dp=[abs((P_ow[i] or 0.0)-(P_om[i] or 0.0)) for i in range(n)]
    dt=[abs((T_ow[i] or 0.0)-(T_om[i] or 0.0)) for i in range(n)]
    avg_dp=sum(dp)/n; avg_dt=sum(dt)/n
    sP = 0.95/(1.0+avg_dp/10.0)
    sT = 0.95/(1.0+avg_dt/6.0)
    return clamp(0.25 + 0.5*((sP+sT)/2.0), 0.25, 0.95)

# ----------------------- SMI (P‑ET0 + CDS prefetch) -----------------------
SM_CACHE: Dict[str, Dict[str, Any]] = {}

async def _prefetch_era5l_sm(lat: float, lon: float, days: int = 40) -> None:
    if not CDS_API_KEY:
        return
    key = f"{round(lat,3)},{round(lon,3)}"
    if key in SM_CACHE and (time.time() - SM_CACHE[key].get("ts", 0)) < 12*3600:
        return
    try:
        import cdsapi  # type: ignore
        from netCDF4 import Dataset, num2date  # type: ignore
    except Exception:
        return  # librerie opzionali non presenti → skip
    def _blocking_download() -> Optional[Dict[str, Any]]:
        try:
            c = cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY, quiet=True, verify=1)
            end = datetime.utcnow().date()
            start = end - timedelta(days=days-1)
            years = sorted({start.year, end.year})
            months = [f"{m:02d}" for m in range(1,13)] if len(years)>1 else [f"{m:02d}" for m in range(start.month, end.month+1)]
            days_list = [f"{d:02d}" for d in range(1,31)]  # il server ignora i giorni non validi
            bbox = [lat+0.05, lon-0.05, lat-0.05, lon+0.05]  # N,W,S,E
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
            swvl1 = ds.variables.get("swvl1")
            if swvl1 is None:
                ds.close(); os.remove(target); return None
            vals = swvl1[:]
            # collassa lat/lon prendendo il primo pixel
            if vals.ndim == 3:
                vals = vals[:,0,0]
            import numpy as _np
            out: Dict[str, float] = {}
            for tt, vv in zip(times, vals):
                if isinstance(vv, _np.ma.MaskedArray):
                    v = float(vv.filled(_np.nan))
                else:
                    v = float(vv)
                d = datetime(tt.year, tt.month, tt.day).date().isoformat()
                out.setdefault(d, [])
                out[d].append(v)
            daily = {d: float(_np.nanmean(vs)) for d, vs in out.items()}
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
    # normalizza 5–95° percentile
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

# ----------------------------- ENDPOINTS -----------------------------
@app.get("/api/health")
async def health():
    return {"ok":True,"time":datetime.now(timezone.utc).isoformat()}

@app.get("/api/geocode")
async def api_geocode(q:str):
    g = await geocode_nominatim(q)
    if not g:
        g = await geocode_openmeteo(q)
    if not g: raise HTTPException(404,"Località non trovata")
    return g

@app.get("/api/score")
async def api_score(
    lat:float=Query(...), lon:float=Query(...),
    half:float=Query(8.0,gt=3.0,lt=20.0),
    habitat:str=Query("", description="castagno,faggio,quercia,conifere,misto,altro"),
    autohabitat:int=Query(1, description="1=auto OSM, 0=manuale"),
    hours:int=Query(2, ge=2, le=8)
):
    # fetch paralleli
    om_task=asyncio.create_task(fetch_open_meteo(lat,lon,past=15,future=10))
    ow_task=asyncio.create_task(fetch_openweather(lat,lon))
    dem_task=asyncio.create_task(fetch_elevation_grid_multiscale(lat,lon))
    osm_task=asyncio.create_task(fetch_osm_habitat(lat,lon)) if autohabitat==1 else None
    # prefetch ERA5‑Land (non blocca la risposta)
    _ = asyncio.create_task(_prefetch_era5l_sm(lat,lon))

    om,ow,(elev_m,slope_deg,aspect_deg,aspect_oct,concavity)=await asyncio.gather(om_task,ow_task,dem_task)
    auto_hab, auto_conf, auto_scores = ("",0.0,{})
    if osm_task:
        try:
            auto_hab, auto_conf, auto_scores = await osm_task
        except Exception:
            auto_hab, auto_conf, auto_scores = ("",0.0,{})

    # Habitat finale usato
    habitat_used = habitat.strip().lower()
    habitat_source = "manuale"
    if autohabitat==1:
        if not habitat_used and auto_hab:
            habitat_used = auto_hab
            habitat_source = f"automatico (OSM, conf {auto_conf:.2f})"
        elif habitat_used:
            habitat_source = "manuale (override)"
    if not habitat_used:
        habitat_used = "altro"

    # Dati Open-Meteo
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

    # Blend con OpenWeather (se disponibile)
    P_fut_ow:List[float]=[]; Tmin_f_ow:List[float]=[]; Tmax_f_ow:List[float]=[]; Tm_f_ow:List[float]=[]
    if ow and "daily" in ow:
        for day in ow["daily"][:futN]:
            P_fut_ow.append(float(day.get("rain",0.0)))
            t=day.get("temp",{})
            Tmin_f_ow.append(float(t.get("min",0.0)))
            Tmax_f_ow.append(float(t.get("max",0.0)))
            Tm_f_ow.append(float(t.get("day", (t.get("min",0.0)+t.get("max",0.0))/2.0)))
    ow_len=min(len(P_fut_ow),futN)

    # Strategia blend: P = max(OM,OW); T = media pesata (OW 0.6 nei primi 3gg, poi 0.5)
    P_fut_blend=[]; Tmin_f_blend=[]; Tmax_f_blend=[]; Tm_f_blend=[]
    for i in range(futN):
        # precip
        if i<ow_len: P_fut_blend.append(max(P_fut_om[i], P_fut_ow[i]))
        else: P_fut_blend.append(P_fut_om[i])
        # temperature
        if i<ow_len:
            w_ow = 0.6 if i<=2 else 0.5
            w_om = 1.0 - w_ow
            Tmin_f_blend.append(w_om*Tmin_f_om[i] + w_ow*Tmin_f_ow[i])
            Tmax_f_blend.append(w_om*Tmax_f_om[i] + w_ow*Tmax_f_ow[i])
            Tm_f_blend.append(w_om*Tm_f_om[i] + w_ow*Tm_f_ow[i])
        else:
            Tmin_f_blend.append(Tmin_f_om[i]); Tmax_f_blend.append(Tmax_f_om[i]); Tm_f_blend.append(Tm_f_om[i])

    # Indicatori recenti per lag/microclima
    API_val=api_index(P_past_om,half_life=half)
    ET7=sum(ET0_p_om[-7:]) if ET0_p_om else 0.0
    RH7=sum(RH_p_om[-7:])/max(1,len(RH_p_om[-7:])) if RH_p_om else 60.0
    SW7=sum(SW_p_om[-7:])/max(1,len(SW_p_om[-7:])) if SW_p_om else 15000.0

    # SMI (primario: P‑ET0; se cache ERA5‑Land esiste per questo punto, la usiamo)
    smi_series = smi_from_p_et0(P_om, ET0_om)
    cache_key = f"{round(lat,3)},{round(lon,3)}"
    sm_used = "P-ET0"
    if cache_key in SM_CACHE:
        # prova ad allineare alle date OM
        daily_sm = SM_CACHE[cache_key].get("daily", {})
        tmp=[]
        for dstr in timev:
            tmp.append(float(daily_sm.get(dstr, float('nan'))))
        # normalizza 5–95
        import numpy as _np
        arr=_np.array(tmp, dtype=float)
        if _np.any(_np.isfinite(arr)):
            valid=arr[_np.isfinite(arr)]
            p5,p95=_np.percentile(valid,[5,95])
            arr=(arr-p5)/max(1e-6,(p95-p5))
            arr=_np.clip(arr,0.0,1.0)
            smi_series=arr.tolist()
            sm_used = "ERA5-Land (CDS cache)"

    # soglia dinamica (ultimi 15 gg)
    import numpy as _np
    sm_last=_np.array(smi_series[-15:], dtype=float)
    sm_last=sm_last[_np.isfinite(sm_last)]
    sm_thr=float(_np.percentile(sm_last,55)) if sm_last.size>0 else 0.6

    # Shock termico
    shock = cold_shock_from_tmin_series(Tmin_p_om)

    # VPD serie future (usa RH_om futuro + Tm_f_blend)
    vpd_fut=[vpd_hpa(float(Tm_f_blend[i]), float(RH_f_om[i] if i<len(RH_f_om) else 60.0)) for i in range(futN)]

    # Energia microclimatica e TWI‑proxy
    month_now = datetime.now(timezone.utc).month
    energy = microclimate_energy(aspect_oct, float(slope_deg), month_now)
    twi = twi_proxy_from_slope_concavity(float(slope_deg), float(concavity))
    micro_amp = clamp(energy * (0.8 + 0.4*(twi-0.5)*2.0), 0.6, 1.2)

    # Previsione di flush (nuova logica)
    reliability = reliability_from_sources(P_fut_ow[:ow_len],P_fut_om[:ow_len],Tm_f_ow[:ow_len],Tm_f_om[:ow_len]) if ow_len else 0.6

    # 1) Rileva eventi pioggia base (come prima)
    def rain_events(times: List[str], rains: List[float]) -> List[Tuple[int,float]]:
        events=[]; n=min(len(times),len(rains)); i=0
        while i<n:
            if rains[i] >= 8.0: events.append((i, rains[i])); i += 1
            elif i+1<n and (rains[i]+rains[i+1]) >= 12.0: events.append((i+1, rains[i]+rains[i+1])); i += 2
            else: i += 1
        return events

    ev_past = rain_events(timev[:pastN], P_past_om)
    ev_fut_raw = rain_events(timev[pastN:pastN+futN], P_fut_blend)
    # rende indici assoluti
    ev_fut=[(pastN+i, mm) for (i,mm) in ev_fut_raw]

    # 2) Applica gating da SMI: se SMI vicino all'evento < soglia → indebolisci o scarta
    def smi_amp(ev_abs_idx:int)->float:
        i0=max(0, ev_abs_idx-1); i1=min(len(smi_series), ev_abs_idx+1)
        loc=float(_np.nanmean(_np.array(smi_series[i0:i1+1], dtype=float))) if i1>i0 else float(smi_series[ev_abs_idx])
        if not (loc==loc):  # NaN
            loc=0.5
        # moltiplicatore 0.4..1.2 in base a SMI e soglia
        base = 0.6 + 0.7*loc
        if loc < sm_thr: base *= 0.6
        return clamp(base, 0.3, 1.2)

    events = ev_past + ev_fut

    forecast=[0.0]*futN; details: List[FlushDetail]=[]
    for (ev_idx_abs, mm_tot) in events:
        # lag stocastico guidato da SMI+shock+T7
        sm_loc = smi_amp(ev_idx_abs)
        lag = stochastic_lag_days(smi=min(1.0, max(0.0, sm_loc)), shock=shock, tmean7=sum(Tm_p_om[-7:])/max(1,len(Tm_p_om[-7:])))
        peak_idx = ev_idx_abs + lag
        sigma = 2.5 if mm_tot < 25 else 3.0
        amp = event_strength(mm_tot) * micro_amp * sm_loc
        if ev_idx_abs >= pastN: amp *= (0.5 + 0.5*reliability)
        for j in range(futN):
            abs_j = pastN + j
            pen = vpd_penalty(vpd_fut[j])
            forecast[j] += 100.0 * amp * gaussian_kernel(abs_j, peak_idx, sigma) * pen
        when = timev[ev_idx_abs] if ev_idx_abs < len(timev) else f"+{ev_idx_abs-pastN}d"
        details.append({
            "event_day_index": ev_idx_abs,
            "event_when": when,
            "event_mm": round(mm_tot,1),
            "lag_days": lag,
            "predicted_peak_abs_index": peak_idx,
        })

    out = [int(round(clamp(v,0.0,100.0))) for v in forecast]
    # smoothing 1-2-1
    smoothed=[]
    for i in range(len(out)):
        w = out[i]
        if i>0: w += out[i-1]
        if i+1<len(out): w += out[i+1]
        denom = 1 + (1 if i>0 else 0) + (1 if i+1<len(out) else 0)
        smoothed.append(int(round(w/denom)))

    s=e=m=0
    if len(smoothed)>=3:
        # best finestra 3gg
        best_s,best_e,best_m=0,2,round((smoothed[0]+smoothed[1]+smoothed[2])/3)
        for i in range(1,len(smoothed)-2):
            med=round((smoothed[i]+smoothed[i+1]+smoothed[i+2])/3)
            if med>best_m: best_s,best_e,best_m=i,i+2,med
        s,e,m = best_s,best_e,best_m

    flush_today = int(smoothed[0] if smoothed else 0)

    # Dimensioni con VPD
    vpd7=max(vpd_hpa(float(sum(Tm_p_om[-7:])/max(1,len(Tm_p_om[-7:]))), float(RH7)), 0.0)
    size_rate = cap_growth_rate_cm_per_day(float(sum(Tm_p_om[-7:])/max(1,len(Tm_p_om[-7:]))), float(RH7), float(vpd7))
    # stima età coorte principale ~ distanza da picco più vicino
    if details:
        today_abs = pastN
        peak_idxs = [d.get("predicted_peak_abs_index", today_abs) for d in details]
        peak_idx = min(peak_idxs, key=lambda k: abs(today_abs - k)) if peak_idxs else today_abs
        start_buttons = peak_idx - 2
        age_days = max(0, today_abs - start_buttons)
    else:
        age_days = 0
    size_cm = clamp(size_rate * age_days, 1.5, 18.0)
    if size_cm < 5.0: size_cls = "bottoni (2–5 cm)"
    elif size_cm < 10.0: size_cls = "medi (6–10 cm)"
    else: size_cls = "grandi (10–15+ cm)"
    smin = clamp(size_rate * max(0, age_days-2), 1.5, 18.0)
    smax = clamp(size_rate * (age_days+2), 1.5, 18.0)

    # Raccolto atteso coerente con l'indice
    def harvest_text_from_index(score:int, hours:int) -> str:
        factor = 1.0 if hours <= 2 else 1.45
        if score < 15: base = (0, 1)
        elif score < 35: base = (1, 2)
        elif score < 55: base = (2, 4)
        elif score < 70: base = (4, 8)
        elif score < 85: base = (6, 12)
        else: base = (10, 20)
        lo = max(0, int(round(base[0]*factor)))
        hi = int(round(base[1]*factor))
        if lo == hi: hi = lo+1
        if lo <= 1 and hi <= 1: return "0–1 porcini"
        return f"{lo}–{hi} porcini"

    harvest_txt = harvest_text_from_index(flush_today, hours)

    # Tabelle piogge
    rain_past={timev[i]: round(P_past_om[i],1) for i in range(min(pastN,len(timev)))}
    rain_future={timev[pastN+i] if pastN+i<len(timev) else f"+{i+1}d":round(P_fut_blend[i],1) for i in range(futN)}

    response_data = {
        "lat": lat, "lon": lon,
        "elevation_m": round(elev_m),
        "slope_deg": slope_deg, "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct if aspect_oct else "NA",
        "concavity": round(concavity,3),

        "API_star_mm": round(api_index(P_past_om,half_life=half),1),
        "P7_mm": round(sum(P_past_om[-7:]),1),
        "P15_mm": round(sum(P_past_om),1),
        "ET0_7d_mm": round(ET7,1),
        "RH7_pct": round(RH7,1),
        "SW7_kj": round(SW7,0),
        "Tmean7_c": round(sum(Tm_p_om[-7:])/max(1,len(Tm_p_om[-7:])),1),

        "index": flush_today,
        "forecast": [int(x) for x in smoothed],
        "best_window": {"start": s, "end": e, "mean": m},
        "harvest_estimate": harvest_txt,
        "reliability": round(reliability,3),

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

        # diagnostica nuova
        "diagnostics": {
            "smi_source": sm_used,
            "smi_threshold": round(sm_thr,2),
            "shock_deltaTmin3d": round(cold_shock_from_tmin_series(Tmin_p_om),2),
            "twi_proxy": round(twi,2),
            "energy_index": round(energy,2),
        }
    }
    response_data["dynamic_explanation"] = build_analysis_text(response_data)
    return response_data

# ----
# index.html: nessuna modifica necessaria rispetto alla tua versione (v1.8.2). Mantieni quello che hai caricato.
# ----
