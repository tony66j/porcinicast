# main.py — Trova Porcini API (v2.0.0)
# Requisiti: Python 3.10+, FastAPI, httpx, uvicorn
#
# Novità v2.0.0
# - Esposizione del versante rigorosa (8 settori: N, NE, E, SE, S, SW, W, NW):
#   media circolare multi-scala (30–90–150 m) pesata per pendenza; output:
#   aspect_octant, aspect_confidence (0–1), northness = sin(slope)*cos(aspect).
#   Se pendenza < 1° → "pianeggiante (nessuna esposizione)".
# - Filtro biogeografico/ecologico (areale porcini):
#   penalizza fortemente fascia equatoriale di bassa quota e habitat senza
#   potenziali ospiti ECM; considera anche finestra climatica T/UR.
#   => impedisce falsi positivi (es. Amazzonia).
# - Analisi del Modello molto più estesa, con spiegazione di:
#   perché la sola pioggia non basta, ruolo di ospiti ECM, stagione, suolo,
#   siccità antecedente, microclima, incertezze e riferimenti sintetici.
# - Resta invariato: indice = flush odierno (0–100), lag dinamico 5–15 gg,
#   modello dimensionale (diametro medio + range coorti), concavità/impluvi.
#
# NOTE scientifiche (richiamate nel testo esplicativo):
# - I porcini (complesso Boletus edulis s.l.) sono ECM di conifere e latifoglie
#   temperate; distribuiti nell'emisfero nord (introdotti localmente altrove).
# - "Northness" = sin(slope)*cos(aspect); l'aspect è 0°=N, senso orario.
# - La fruttificazione dipende da pioggia + termica/UR + suolo + ospiti + stagione;
#   la pioggia da sola spesso non produce flush.

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, date

app = FastAPI(title="Trova Porcini API (v2.0.0)", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

HEADERS = {"User-Agent":"Trovaporcini/2.0.0 (+site)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
]

# ----------------------------- UTIL -----------------------------
def clamp(v,a,b): return a if v<a else b if v>b else v
def half_life_coeff(days: float) -> float: return 0.5**(1.0/max(1.0,days))
def deg2rad(d: float) -> float: return d*math.pi/180.0
def rad2deg(r: float) -> float: return r*180.0/math.pi

def api_index(precip: List[float], half_life: float=8.0) -> float:
    k=half_life_coeff(half_life); api=0.0
    for p in precip: api=k*api+(p or 0.0)
    return api

def temperature_fit(tmin: float, tmax: float, tmean: float) -> float:
    # idoneità termica 0..1 (finestra temperata)
    if tmin < -2 or tmax > 33:
        return 0.0
    if tmean<=6: base=max(0.0,(tmean)/6.0*0.35)
    elif tmean<=10: base=0.35+0.25*((tmean-6)/4.0)
    elif tmean<=18: base=0.60+0.30*((tmean-10)/8.0)
    elif tmean<=22: base=0.90-0.20*((tmean-18)/4.0)
    else: base=max(0.0,0.70-0.70*((tmean-22)/10.0))
    if tmin<6:  base*=max(0.35, 1-(6-tmin)/8.0)
    if tmax>26: base*=max(0.35, 1-(tmax-26)/8.0)
    return clamp(base,0.0,1.0)

def reliability_from_sources(ow_vals:List[float],om_vals:List[float]) -> float:
    n=min(len(ow_vals),len(om_vals))
    if n==0: return 0.6
    diffs=[abs((ow_vals[i] or 0.0)-(om_vals[i] or 0.0)) for i in range(n)]
    avg_diff=sum(diffs)/n
    return clamp(0.95/(1.0+avg_diff/6.0),0.25,0.95)

def stddev(xs: List[float]) -> float:
    if not xs: return 0.0
    m=sum(xs)/len(xs)
    return (sum((x-m)**2 for x in xs)/len(xs))**0.5

# ---------------- DEM multi-scala, aspect e concavità ----------------
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
    # Horn kernel, +x=E, +y=S
    dzdx=((z[0][2]+2*z[1][2]+z[2][2])-(z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_size_m)
    dzdy=((z[2][0]+2*z[2][1]+z[2][2])-(z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_size_m)
    slope=math.degrees(math.atan(math.hypot(dzdx,dzdy)))
    # 0°=N (CW), direzione di massima pendenza in discesa:
    aspect=(math.degrees(math.atan2(-dzdx, dzdy))+360.0)%360.0
    octs=["N","NE","E","SE","S","SW","W","NW","N"]
    octant=octs[int(((aspect%360)+22.5)//45)]
    return round(slope,1),round(aspect,0),octant

def concavity_from_grid(z:List[List[float]])->float:
    center=z[1][1]; neigh=[z[r][c] for r in (0,1,2) for c in (0,1,2) if not (r==1 and c==1)]
    delta=(sum(neigh)/8.0 - center)  # m; positivo = concavo (accumulo)
    return clamp(delta/6.0, -0.1, +0.1)

async def fetch_elevation_multiscale_metrics(lat:float,lon:float)->Tuple[float,float,float,Optional[str],float,float,float]:
    """
    Ritorna: elev_m, slope_mean_deg, aspect_mean_deg, aspect_octant, concavity, aspect_confidence (0..1), northness
    """
    steps=(30.0,90.0,150.0)
    records=[]
    elev_center=None
    best_grid=None; best_flat=0.0
    for s in steps:
        z = await _fetch_elev_block(lat,lon,s)
        if not z: continue
        if elev_center is None: elev_center=z[1][1]
        slope,aspect,octant = slope_aspect_from_grid(z,cell_size_m=s)
        flatness = stddev([*z[0],*z[1],*z[2]])
        records.append({"step":s,"slope":slope,"aspect":aspect,"oct":octant,"flat":flatness,"z":z})
        # scegli griglia "best" per concavità: più strutturata e/o più ripida
        if best_grid is None or slope>1.5 and (flatness>best_flat or records[-2]["slope"]<=1.5 if len(records)>1 else True):
            best_grid=z; best_flat=flatness
    if not records:
        return 800.0, 0.0, 0.0, None, 0.0, 0.0, 0.0

    # media circolare pesata per pendenza e scala
    scale_w = {30.0:0.5, 90.0:0.3, 150.0:0.2}
    X=Y=0.0; wsum=0.0; slope_wsum=0.0
    for r in records:
        slope_rad=deg2rad(r["slope"])
        w = math.sin(slope_rad) * scale_w.get(r["step"],0.2)
        a_rad=deg2rad(r["aspect"])
        # componenti: x = sin(a), y = cos(a) con a=0° N, CW
        X += w*math.sin(a_rad)
        Y += w*math.cos(a_rad)
        wsum += w
        slope_wsum += r["slope"]*w
    if wsum <= 1e-6:
        # pianeggiante
        conc = concavity_from_grid(best_grid) if best_grid else 0.0
        return float(elev_center), 0.0, 0.0, None, conc, 0.0, 0.0

    aspect_mean_rad = math.atan2(X, Y)  # rad, 0=N
    aspect_mean_deg = (rad2deg(aspect_mean_rad)+360.0)%360.0
    slope_mean_deg = clamp(slope_wsum/wsum,0.0,90.0)
    conc = concavity_from_grid(best_grid) if best_grid else 0.0

    # confidenza: forza del vettore * fattore pendenza
    R = math.hypot(X,Y)/wsum  # 0..1
    slope_factor = slope_mean_deg / (slope_mean_deg + 5.0)  # saturazione dolce
    aspect_conf = clamp(R * slope_factor, 0.0, 1.0)

    # northness
    northness = math.sin(deg2rad(slope_mean_deg)) * math.cos(aspect_mean_rad)

    # etichetta 8 settori, o pianeggiante
    if slope_mean_deg < 1.0:
        octant = None
    else:
        octs=["N","NE","E","SE","S","SW","W","NW","N"]
        octant=octs[int(((aspect_mean_deg%360)+22.5)//45)]

    return float(elev_center), round(slope_mean_deg,1), round(aspect_mean_deg,0), octant, conc, round(aspect_conf,3), round(northness,3)

# ---------------- Habitat auto da OSM ----------------
def _score_tags(tags: Dict[str,str])->Dict[str,float]:
    t = {k.lower(): (v.lower() if isinstance(v,str) else v) for k,v in (tags or {}).items()}
    s = {"castagno":0.0,"faggio":0.0,"quercia":0.0,"conifere":0.0,"misto":0.0,"altro":0.0}
    genus = t.get("genus",""); species = t.get("species","")
    leaf_type = t.get("leaf_type",""); landuse = t.get("landuse",""); natural=t.get("natural",""); wood=t.get("wood","")
    if "castanea" in genus or "castagna" in species: s["castagno"] += 3.0
    if "quercus" in genus or "querce" in species:  s["quercia"]  += 3.0
    if "fagus" in genus or "faggio" in species:    s["faggio"]   += 3.0
    if any(g in genus for g in ("pinus","abies","picea","larix","cedrus")): s["conifere"] += 2.5
    if "needleleaved" in leaf_type: s["conifere"] += 1.5
    if wood in ("conifer","pine","spruce","fir","larch","cedar"): s["conifere"] += 1.2
    if wood in ("broadleaved","deciduous"): s["misto"] += 0.6
    if landuse=="forest" or natural in ("wood","forest"):
        for k in s: s[k] += 0.1
    if not any(s.values()): s["altro"] = 1.0
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
                r = await c.post(url, data={"data": q})
                r.raise_for_status()
                j = r.json()
            scores = {"castagno":0.0,"faggio":0.0,"quercia":0.0,"conifere":0.0,"misto":0.0,"altro":0.0}
            for el in j.get("elements", []):
                local = _score_tags(el.get("tags", {}))
                for k,v in local.items(): scores[k]+=v
            hab, conf = _choose_habitat(scores)
            return hab, conf, scores
        except Exception:
            continue
    return "altro", 0.10, {"castagno":0,"faggio":0,"quercia":0,"conifere":0,"misto":0,"altro":1}

# ---------------- Microclima / esposizione ----------------
def microclimate_from_aspect(aspect_oct: Optional[str], slope_deg: float, rh7: float, sw7: float, tmean7: float) -> float:
    if not aspect_oct or slope_deg < 1.0:
        return 0.5
    bonus = 0.0
    if aspect_oct in ("N","NE","NW"):
        if tmean7 >= 16 and rh7 < 65: bonus += 0.10
        if sw7 > 18000: bonus += 0.05
    elif aspect_oct in ("S","SE","SW"):
        if tmean7 >= 18 and sw7 > 18000 and rh7 < 60: bonus -= 0.10
    k = min(1.0, max(0.5, slope_deg/12.0))
    bonus *= k
    return clamp(0.5 + bonus,0.0,1.0)

# ---------------- Specie & consigli ----------------
def species_for_habitat(hab: str) -> List[Dict[str,str]]:
    h = (hab or "").lower()
    if "querc" in h or "oak" in h:
        return [{"latin":"Boletus reticulatus (B. aestivalis)","common":"porcino estivo/quercino"},
                {"latin":"Boletus aereus","common":"porcino nero/bronzo"}]
    if "castag" in h:
        return [{"latin":"Boletus aereus","common":"porcino nero/bronzo"},
                {"latin":"Boletus reticulatus","common":"porcino estivo/quercino"}]
    if "fagg" in h or "beech" in h:
        return [{"latin":"Boletus edulis","common":"porcino (faggete)"},
                {"latin":"Boletus aereus","common":"sporadico in latifoglie calde"}]
    if "conifer" in h or "abete" in h or "pino" in h:
        return [{"latin":"Boletus pinophilus","common":"porcino dei pini"},
                {"latin":"Boletus edulis","common":"porcino (abete/pecceta)"}]
    return [{"latin":"Boletus edulis agg.","common":"complesso dei porcini"},
            {"latin":"Boletus reticulatus / B. aereus","common":"latifoglie"}]

def safety_notes_short()->List[str]:
    return [
        "Attenzione a Tylopilus felleus (amaro): reticolo scuro sul gambo, pori presto rosati.",
        "Evita Rubroboletus (es. R. satanas): colori vivi e viraggio marcato; molti sono tossici.",
        "Raccogli solo specie che riconosci con certezza e rispetta le normative locali."
    ]

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

# ---------------- Lag, eventi pioggia e previsione flush ----------------
def dynamic_lag_days(tmean: float, elev: float, rh: float, sw: float, concavity: float, antecedent_api: float, habitat: str) -> int:
    lag = 9
    if 16 <= tmean <= 20: lag -= 2
    elif tmean < 12: lag += 2
    elif tmean > 22: lag += 1
    if elev >= 1200: lag += 1
    elif elev <= 400 and tmean >= 18: lag -= 1
    if rh >= 70 and sw < 15000: lag -= 1
    if rh < 50 and sw > 19000: lag += 1
    if concavity > 0.05: lag -= 1
    if concavity < -0.05: lag += 1
    if antecedent_api < 8: lag += 1
    return int(clamp(lag, 5, 15))

def rain_events(times: List[str], rains: List[float]) -> List[Tuple[int,float]]:
    events=[]; n=min(len(times),len(rains)); i=0
    while i<n:
        if rains[i] >= 8.0: events.append((i, rains[i])); i += 1
        elif i+1<n and (rains[i]+rains[i+1]) >= 12.0: events.append((i+1, rains[i]+rains[i+1])); i += 2
        else: i += 1
    return events

def gaussian_kernel(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5*((x-mu)/sigma)**2)

def event_strength(mm: float) -> float:
    return 1.0 - math.exp(-mm/20.0)

def climate_suitability_factor(tmin: float, tmax: float, tmean7: float, rh7: float) -> float:
    # riduce il flush quando fuori finestra termica/igrometrica
    tf = temperature_fit(tmin, tmax, tmean7)  # 0..1
    rh_pen = 1.0
    if rh7 < 40: rh_pen = 0.5
    elif rh7 < 55: rh_pen = 0.8
    elif rh7 > 95: rh_pen = 0.8
    return clamp(tf * rh_pen, 0.0, 1.0)

def biogeographic_suitability(lat: float, elev: float, habitat_used: str, auto_scores: Dict[str,float], tmean7: float, rh7: float) -> Tuple[float,str]:
    """
    Idoneità 0..1 vs areale/ospiti/clima di fondo. Penalizza fascia equatoriale di bassa quota e habitat senza ospiti ECM.
    """
    score = 1.0
    reason = []
    abslat = abs(lat)
    # Fascia equatoriale bassa quota: quasi zero
    if abslat < 12.0 and elev < 900:
        score *= 0.05; reason.append("fascia equatoriale di bassa quota (fuori areale tipico)")
    elif abslat < 12.0 and elev >= 900:
        score *= 0.4; reason.append("fascia tropicale montana: possibile solo con boschi ECM idonei")
    else:
        reason.append("fascia temperata/subtrop. idonea")
    # Ospiti ECM: conifere, faggio, quercia, castagno, misto → ok; altro → penalità
    h = (habitat_used or "").lower()
    if any(k in h for k in ("conifer","abete","pino","fagg","querc","castag","misto")):
        reason.append("presenza potenziale di ospiti ECM")
    else:
        score *= 0.7; reason.append("ospiti ECM incerti/assenti")
    # Clima di fondo (temperatura/UR medie 7 gg)
    clim = climate_suitability_factor(tmin=10, tmax=24, tmean7=tmean7, rh7=rh7)  # grossolana: usiamo Tmean7
    score *= (0.6 + 0.4*clim)
    return clamp(score,0.0,1.0), "; ".join(reason)

def build_flush_forecast(
    past_api: float,
    tmin7: float, tmax7: float, tmean7: float, elev: float, rh7: float, sw7: float,
    concavity: float, micro: float, habitat: str,
    timev: List[str], pastN: int, futN: int,
    P_past: List[float], P_fut_blend: List[float],
    reliability: float,
    suitability: float, host_factor: float
) -> Tuple[List[int], List[Dict[str,Any]]]:
    ev_past = rain_events(timev[:pastN], P_past)
    ev_fut = []
    for i in range(futN):
        if P_fut_blend[i] >= 8.0 or (i+1<futN and (P_fut_blend[i]+P_fut_blend[i+1])>=12.0):
            if i+1<futN and P_fut_blend[i+1] >= P_fut_blend[i]:
                ev_fut.append((pastN+i+1, P_fut_blend[i]+(P_fut_blend[i+1] if P_fut_blend[i+1]>0 else 0)))
            else:
                ev_fut.append((pastN+i, P_fut_blend[i]))
    # Unisci, evitando duplicati
    events=[]; seen=set()
    for e in ev_past + ev_fut:
        if e[0] in seen:
            idx = [k for k,(ii,_) in enumerate(events) if ii==e[0]]
            if idx: events[idx[0]] = (events[idx[0]][0], events[idx[0]][1] + e[1])
            else: events.append(e); seen.add(e[0])
        else:
            events.append(e); seen.add(e[0])

    clim_factor = climate_suitability_factor(tmin7, tmax7, tmean7, rh7)
    forecast=[0.0]*futN; details=[]
    for (ev_idx, mm_tot) in events:
        lag = dynamic_lag_days(tmean7, elev, rh7, sw7, concavity, past_api, habitat)
        peak_idx = ev_idx + lag
        sigma = 2.5 if mm_tot < 25 else 3.0
        amp = event_strength(mm_tot)
        # Modulatori: microclima, affidabilità, clima di fondo, areale/ospiti
        amp *= clamp(0.85 + 0.3*(micro-0.5), 0.75, 1.15)
        if ev_idx >= pastN: amp *= (0.5 + 0.5*reliability)
        amp *= clamp(0.3 + 0.7*clim_factor, 0.0, 1.0)
        amp *= clamp(suitability, 0.0, 1.0) * clamp(host_factor, 0.2, 1.0)
        for j in range(futN):
            abs_j = pastN + j
            forecast[j] += 100.0 * amp * gaussian_kernel(abs_j, peak_idx, sigma)
        when = timev[ev_idx] if ev_idx < len(timev) else f"+{ev_idx-pastN}d"
        details.append({
            "event_day_index": ev_idx,
            "event_when": when,
            "event_mm": round(mm_tot,1),
            "lag_days": lag,
            "predicted_peak_abs_index": peak_idx,
        })
    out = [int(round(clamp(v,0.0,100.0))) for v in forecast]
    # smoothing leggero (1-2-1)
    smoothed=[]
    for i in range(len(out)):
        w = out[i]
        if i>0: w += out[i-1]
        if i+1<len(out): w += out[i+1]
        denom = 1 + (1 if i>0 else 0) + (1 if i+1<len(out) else 0)
        smoothed.append(int(round(w/denom)))
    return smoothed, details

def best_window(values: List[int]) -> Tuple[int,int,int]:
    if len(values)<3: return (0,0,0)
    best_s,best_e,best_m=0,2,round((values[0]+values[1]+values[2])/3)
    for i in range(1,len(values)-2):
        m=round((values[i]+values[i+1]+values[i+2])/3)
        if m>best_m: best_s,best_e,best_m=i,i+2,m
    return best_s,best_e,best_m

# ---------------- Modello dimensionale (diametro cappello) ----------------
def cap_growth_rate_cm_per_day(tmean: float, rh: float) -> float:
    if rh < 40: return 0.0
    ur_f = clamp((rh - 40.0) / (85.0 - 40.0), 0.0, 1.0)
    if tmean <= 10: t_f = 0.2 * (tmean/10.0)
    elif tmean <= 16: t_f = 0.2 + 0.8*(tmean-10)/6.0
    elif tmean <= 20: t_f = 1.0
    elif tmean <= 24: t_f = 1.0 - 0.6*(tmean-20)/4.0
    elif tmean <= 28: t_f = 0.4 - 0.3*(tmean-24)/4.0
    else: t_f = 0.1
    return 2.1 * ur_f * clamp(t_f,0.0,1.0)  # cm/giorno

def estimate_size_today_cm(events: List[Dict[str,Any]], pastN:int, tmean7:float, rh7:float) -> Tuple[float,str]:
    if not events: return 0.0, "—"
    today_abs = pastN
    peak_idxs = [e["predicted_peak_abs_index"] for e in events if "predicted_peak_abs_index" in e]
    if not peak_idxs: return 0.0, "—"
    peak_idx = min(peak_idxs, key=lambda k: abs(today_abs - k))
    start_buttons = peak_idx - 2
    age_days = max(0, today_abs - start_buttons)
    g = cap_growth_rate_cm_per_day(tmean7, rh7)
    size_cm = clamp(g * age_days, 1.5, 18.0)
    if size_cm < 5.0: cls = "bottoni (2–5 cm)"
    elif size_cm < 10.0: cls = "medi (6–10 cm)"
    else: cls = "grandi (10–15+ cm)"
    return round(size_cm,1), cls

def estimate_size_range_today_cm(events: List[Dict[str,Any]], pastN:int, tmean7:float, rh7:float) -> Tuple[float,float]:
    if not events: return (0.0, 0.0)
    today_abs = pastN
    peak_idxs = [e["predicted_peak_abs_index"] for e in events if "predicted_peak_abs_index" in e]
    if not peak_idxs: return (0.0, 0.0)
    peak_idx = min(peak_idxs, key=lambda k: abs(today_abs - k))
    start_buttons = peak_idx - 2
    age_med = max(0, today_abs - start_buttons)
    g = cap_growth_rate_cm_per_day(tmean7, rh7)
    age_min = max(0, age_med - 2)
    age_max = age_med + 2
    smin = clamp(g * age_min, 1.5, 18.0)
    smax = clamp(g * age_max, 1.5, 18.0)
    if smax < smin: smin, smax = smax, smin
    return round(smin,1), round(smax,1)

# ---------------- Spiegazione (estesa) ----------------
def build_analysis_text(payload: Dict[str,Any]) -> str:
    idx = payload["index"]
    best = payload["best_window"]
    p15 = payload["P15_mm"]; p7 = payload["P7_mm"]
    rh7 = payload.get("RH7_pct", None); sw7 = payload.get("SW7_kj", None)
    tm7 = payload["Tmean7_c"]
    aspect = payload.get("aspect_octant")
    slope = payload.get("slope_deg")
    elev = payload.get("elevation_m")
    aspect_conf = payload.get("aspect_confidence",0.0)
    northness = payload.get("northness",0.0)
    habitat_used = (payload.get("habitat_used") or "").capitalize() or "Altro"
    habitat_source = payload.get("habitat_source","manuale")
    size_cm = payload.get("size_cm", 0.0)
    size_class = payload.get("size_class", "—")
    rng = payload.get("size_range_cm",[0.0,0.0])
    suit = payload.get("suitability_score",1.0)
    suit_reason = payload.get("suitability_reason","")
    host_factor = payload.get("host_factor",1.0)

    out=[]
    # Oggi
    exp = aspect if aspect else "pianeggiante (nessuna esposizione)"
    out.append(f"<p><strong>Indice attuale (flush): {idx}/100</strong> • Habitat: <strong>{habitat_used}</strong> ({habitat_source}). "
               f"Quota <strong>{elev} m</strong>, pendenza <strong>{slope}°</strong>, esposizione <strong>{exp}</strong> "
               f"(conf {round(aspect_conf*100):d}%, northness {northness:+.2f}).</p>")

    if size_cm > 0:
        note_range = ""
        if rng and len(rng)==2 and rng[1] > 0:
            note_range = f" Range atteso oggi: <strong>{rng[0]}–{rng[1]} cm</strong> (coorti di età diverse durante il flush)."
        out.append(f"<p>Per oggi, la <strong>taglia media stimata</strong> dei cappelli è ~<strong>{size_cm} cm</strong> "
                   f"({size_class}).{note_range}</p>")

    # Perché pioggia ≠ porcini garantiti
    out.append("<h4>Perché dopo la pioggia non sempre nascono porcini</h4><ul>"
               "<li><strong>Ospiti ECM</strong>: i porcini fruttificano in simbiosi con conifere e latifoglie idonee; senza ospite la probabilità è quasi nulla.</li>"
               "<li><strong>Finestra climatica</strong>: serve termica temperata e UR non troppo bassa/alta; ondate di caldo o siccità antecedente possono ritardare/azzerare il flush.</li>"
               "<li><strong>Suolo e concavità</strong>: serve umidità nel profilo; impluvi/ombreggi aumentano la riuscita, convessità/irraggiamento la riducono.</li>"
               "<li><strong>Fenologia stagionale</strong>: gli apici stagionali dipendono da stagione/quota/habitat; fuori fase stagionale il segnale è debole.</li>"
               "</ul>")

    # Filtro areale/ecologico
    out.append(f"<h4>Idoneità ecologica/areale</h4>"
               f"<p>Fattore di idoneità: <strong>{round(suit*100):d}%</strong> "
               f"(motivo: {suit_reason}; ospiti/habitat: fattore {host_factor:.2f}). "
               f"In aree tropicali di bassa quota o senza ospiti ECM l'indice viene <strong>ridotto quasi a zero</strong>.</p>")

    # Eventi e finestre
    evs = payload.get("flush_events", [])
    if evs:
        out.append("<h4>Eventi di pioggia e finestre stimate</h4><ul>")
        for e in evs:
            when = e.get("event_when"); mm = e.get("event_mm"); lag = e.get("lag_days")
            out.append(f"<li>Evento ~<strong>{mm} mm</strong> il <strong>{when}</strong> → "
                       f"flush atteso ~<strong>{lag} giorni</strong> dopo (dipende da T, UR, suolo, microclima e habitat).</li>")
        out.append("</ul>")

    if best and best.get("mean",0)>0:
        s,e,m = best["start"], best["end"], best["mean"]
        out.append(f"<p>Nei prossimi 10 giorni la finestra con <strong>maggiore probabilità</strong> è tra "
                   f"<strong>giorno {s+1}</strong> e <strong>giorno {e+1}</strong> (media ≈ <strong>{m}</strong>).</p>")

    # Contesto meteo/microclima
    cause=[]
    cause.append(f"Antecedente: 15 gg = <strong>{p15:.0f} mm</strong> (7 gg: {p7:.0f} mm).")
    cause.append(f"Termica media 7 gg: <strong>{tm7:.1f}°C</strong>; UR7 ≈ <strong>{rh7:.0f}%</strong>; radiazione ≈ <strong>{sw7:.0f} kJ m⁻²</strong>.")
    if aspect:
        if aspect in ("N","NE","NW"): cause.append("Versante fresco/ombroso → conserva umidità.")
        elif aspect in ("S","SE","SW"): cause.append("Versante caldo/irradiato → asciuga più in fretta.")
        else: cause.append("Esposizione intermedia (E/W).")
    else:
        cause.append("Pendenza bassa → esposizione trascurabile.")
    out.append("<h4>Contesto meteo-microclimatico</h4><ul>" + "".join(f"<li>{c}</li>" for c in cause) + "</ul>")

    # Consigli
    out.append("<h4>Consigli operativi</h4><ul>"
               "<li>Cerca <strong>impluvi e margini ombrosi</strong> quando UR è bassa/radiazione alta; evita dorsali esposte.</li>"
               "<li>Se l'indice è alto ma taglia piccola, prova aree più calde/esposte; se indice basso ma stagione idonea, aspetta il <em>lag</em> del prossimo evento.</li>"
               "<li>Ricorda i limiti: previsioni probabilistiche, fortemente dipendenti dall’ospite e dal micro-sito.</li>"
               "</ul>")

    # Riferimenti sintetici (nominali)
    out.append("<details><summary>Riferimenti essenziali (sintesi)</summary>"
               "<ul>"
               "<li>Distribuzione/ECM dei porcini (emisfero nord; conifere/latifoglie).</li>"
               "<li>Northness = sin(pendenza)·cos(aspect) per stimare l'esposizione termica/idrica.</li>"
               "<li>Rendimento e timing guidati da meteo/umidità del suolo; la pioggia da sola non basta.</li>"
               "</ul></details>")
    return "\n".join(out)

# ---------------- Mappa indice → raccolto atteso ----------------
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

# ---------------- ENDPOINTS ----------------
@app.get("/api/health")
async def health():
    return {"ok":True,"time":datetime.now(timezone.utc).isoformat()}

@app.get("/api/geocode")
async def api_geocode(q:str):
    url="https://nominatim.openstreetmap.org/search"
    params={"format":"json","q":q,"addressdetails":1,"limit":1,"email":"info@example.com"}
    async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
        r=await c.get(url,params=params); r.raise_for_status(); data=r.json()
    if not data: raise HTTPException(404,"Località non trovata")
    return {"lat":float(data[0]["lat"]),"lon":float(data[0]["lon"]),"display":data[0].get("display_name","")}

@app.get("/api/score")
async def api_score(
    lat:float=Query(...), lon:float=Query(...),
    half:float=Query(8.0,gt=3.0,lt=20.0),
    habitat:str=Query("", description="castagno,faggio,quercia,conifere,misto,altro"),
    autohabitat:int=Query(1, description="1=auto OSM, 0=manuale"),
    hours:int=Query(2, ge=2, le=8)
):
    om_task=asyncio.create_task(fetch_open_meteo(lat,lon,past=15,future=10))
    ow_task=asyncio.create_task(fetch_openweather(lat,lon))
    dem_task=asyncio.create_task(fetch_elevation_multiscale_metrics(lat,lon))
    osm_task=asyncio.create_task(fetch_osm_habitat(lat,lon)) if autohabitat==1 else None

    om,ow,(elev_m,slope_deg,aspect_deg,aspect_oct,concavity,aspect_conf,northness)=await asyncio.gather(om_task,ow_task,dem_task)
    auto_hab, auto_conf, auto_scores = ("",0.0,{})
    if osm_task:
        try:
            auto_hab, auto_conf, auto_scores = await osm_task
        except Exception:
            auto_hab, auto_conf, auto_scores = ("",0.0,{})

    # Habitat finale
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

    # Open-Meteo
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
    ET0_p_om=ET0_om[:pastN]; ET0_f_om=ET0_om[pastN:pastN+futN]
    RH_p_om=RH_om[:pastN]; SW_p_om=SW_om[:pastN]

    # Blend con OpenWeather (se disponibile)
    P_fut_ow:List[float]=[]; Tmin_f_ow:List[float]=[]; Tmax_f_ow:List[float]=[]; Tm_f_ow:List[float]=[]
    if ow and "daily" in ow:
        for day in ow["daily"]:
            P_fut_ow.append(float(day.get("rain",0.0)))
            t=day.get("temp",{})
            Tmin_f_ow.append(float(t.get("min",0.0)))
            Tmax_f_ow.append(float(t.get("max",0.0)))
            Tm_f_ow.append(float(t.get("day",(t.get("min",0.0)+t.get("max",0.0))/2.0)))
    ow_len=min(len(P_fut_ow),futN)
    w_ow,w_om=0.60,0.40
    def blend(arr_ow,arr_om,i):
        return w_ow*arr_ow[i]+w_om*arr_om[i] if i<ow_len else arr_om[i]

    P_fut_blend,Tmin_f_blend,Tmax_f_blend,Tm_f_blend=[],[],[],[]
    for i in range(futN):
        P_fut_blend.append(blend(P_fut_ow,P_fut_om,i))
        Tmin_f_blend.append(blend(Tmin_f_ow,Tmin_f_om,i) if ow_len else Tmin_f_om[i])
        Tmax_f_blend.append(blend(Tmax_f_ow,Tmax_f_om,i) if ow_len else Tmax_f_om[i])
        Tm_f_blend.append(blend(Tm_f_ow,Tm_f_om,i) if ow_len else Tm_om[i+pastN])

    API_val=api_index(P_past_om,half_life=half)
    ET7=sum(ET0_p_om[-7:]) if ET0_p_om else 0.0
    RH7=sum(RH_p_om[-7:])/max(1,len(RH_p_om[-7:])) if RH_p_om else 60.0
    SW7=sum(SW_p_om[-7:])/max(1,len(SW_p_om[-7:])) if SW_p_om else 15000.0
    Tm7=sum(map(float,Tm_p_om[-7:]))/max(1,len(Tm_p_om[-7:]))

    # Microclima da esposizione
    micro_today = microclimate_from_aspect(aspect_oct or None, float(slope_deg), float(RH7), float(SW7), float(Tm7))

    # Filtro areale/ecologico e fattore ospite
    suitability_score, suitability_reason = biogeographic_suitability(lat, elev_m, habitat_used, auto_scores, Tm7, RH7)
    host_factor = 1.0 if any(k in habitat_used for k in ("castagno","faggio","quercia","conifere","misto")) else 0.3

    # Previsione di flush (con modulatori areali/climatici)
    Tmin7 = float(Tmin_p_om[-1]) if Tmin_p_om else Tm7-2.0
    Tmax7 = float(Tmax_p_om[-1]) if Tmax_p_om else Tm7+2.0
    reliability = reliability_from_sources(P_fut_ow[:ow_len],P_fut_om[:ow_len]) if ow_len else 0.6
    flush_forecast, events_details = build_flush_forecast(
        past_api=API_val,
        tmin7=Tmin7, tmax7=Tmax7, tmean7=Tm7, elev=elev_m, rh7=RH7, sw7=SW7,
        concavity=concavity, micro=micro_today, habitat=habitat_used,
        timev=om["daily"]["time"], pastN=pastN, futN=futN,
        P_past=P_past_om, P_fut_blend=P_fut_blend,
        reliability=reliability,
        suitability=suitability_score, host_factor=host_factor
    )
    s,e,m=best_window(flush_forecast)
    flush_today = int(flush_forecast[0] if flush_forecast else 0)

    # Dimensioni
    size_cm, size_cls = estimate_size_today_cm(events_details, pastN=pastN, tmean7=Tm7, rh7=RH7)
    size_min_cm, size_max_cm = estimate_size_range_today_cm(events_details, pastN=pastN, tmean7=Tm7, rh7=RH7)

    # Raccolto atteso coerente
    harvest_txt = harvest_text_from_index(flush_today, hours)

    # Tabelle piogge
    rain_past={om["daily"]["time"][i]: round(P_past_om[i],1) for i in range(min(pastN,len(om["daily"]["time"])))}
    rain_future={om["daily"]["time"][pastN+i] if pastN+i<len(om["daily"]["time"]) else f"+{i+1}d":round(P_fut_blend[i],1) for i in range(futN)}

    response_data = {
        "lat": lat, "lon": lon,
        "elevation_m": round(elev_m),
        "slope_deg": slope_deg, "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct if aspect_oct else None,
        "aspect_confidence": aspect_conf,
        "northness": northness,
        "concavity": round(concavity,3),

        "API_star_mm": round(API_val,1),
        "P7_mm": round(sum(P_past_om[-7:]),1),
        "P15_mm": round(sum(P_past_om),1),
        "ET0_7d_mm": round(ET7,1),
        "RH7_pct": round(RH7,1),
        "SW7_kj": round(SW7,0),
        "Tmean7_c": round(Tm7,1),

        "index": flush_today,
        "forecast": [int(x) for x in flush_forecast],
        "best_window": {"start": s, "end": e, "mean": m},
        "harvest_estimate": harvest_txt,
        "reliability": round(reliability,3),

        "rain_past": rain_past,
        "rain_future": rain_future,

        "habitat_used": habitat_used,
        "habitat_source": habitat_source,
        "auto_habitat_scores": auto_scores,
        "auto_habitat_confidence": round(auto_conf,3),

        "flush_events": events_details,

        "size_cm": size_cm,
        "size_class": size_cls,
        "size_range_cm": [size_min_cm, size_max_cm],

        "suitability_score": round(suitability_score,3),
        "suitability_reason": suitability_reason,
        "host_factor": round(host_factor,2),
    }
    response_data["dynamic_explanation"] = build_analysis_text(response_data)
    return response_data


