# main.py — Trova Porcini API (v1.7.1 — aspect fix + dynamic harvest)
# Requisiti: Python 3.10+, FastAPI, httpx, uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, date

app = FastAPI(title="Trova Porcini API (v1.7.1)", version="1.7.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

HEADERS = {"User-Agent":"Trovaporcini/1.7 (+site)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
]

# =============== UTIL ===============
def clamp(v,a,b): return a if v<a else b if v>b else v
def half_life_coeff(days: float) -> float: return 0.5**(1.0/max(1.0,days))

def api_index(precip: List[float], half_life: float=8.0) -> float:
    """Antecedent Precipitation Index con decadimento esponenziale (emivita in giorni)."""
    k=half_life_coeff(half_life); api=0.0
    for p in precip: api=k*api+(p or 0.0)
    return api

def temperature_fit(tmin: float, tmax: float, tmean: float) -> float:
    """Fitness termica morbida (0–1) per Boletus spp."""
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

# =============== DEM multi-scala + concavità (open-elevation) ===============
_elev_cache: Dict[str, Any] = {}
def _grid_key(lat:float,lon:float,step:float)->str: return f"{round(lat,5)},{round(lon,5)}@{int(step)}"
async def _fetch_elev_block(lat:float,lon:float,step_m:float)->Optional[List[List[float]]]:
    key=_grid_key(lat,lon,step_m)
    if key in _elev_cache:
        return _elev_cache[key]
    try:
        deg_lat=1/111320.0; deg_lon=1/(111320.0*max(0.2,math.cos(math.radians(lat))))
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
    """
    Calcolo Horn 3×3. Convenzione aspect corretta:
    0° = N, 90° = E, 180° = S, 270° = W.
    """
    # derivata verso EST (x) e verso NORD (y)
    dzdx=((z[0][2]+2*z[1][2]+z[2][2])-(z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_size_m)
    dzdy=((z[2][0]+2*z[2][1]+z[2][2])-(z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_size_m)
    slope=math.degrees(math.atan(math.hypot(dzdx,dzdy)))
    # aspetta: flusso massimo di pendenza verso l'angolo azimutale
    # formula standard per 0=N: atan2(dzdy, -dzdx)
    aspect=(math.degrees(math.atan2(dzdy, -dzdx))+360.0)%360.0
    octs=["N","NE","E","SE","S","SW","W","NW"]
    octant = octs[int(((aspect+22.5)//45) % 8)]
    return round(slope,1), round(aspect,0), octant

def concavity_from_grid(z:List[List[float]])->float:
    center=z[1][1]; neigh=[z[r][c] for r in (0,1,2) for c in (0,1,2) if not (r==1 and c==1)]
    delta=(sum(neigh)/8.0 - center)  # m
    return clamp(delta/6.0, -0.1, +0.1)  # concavo positivo (accumulo)

async def fetch_elevation_grid_multiscale(lat:float,lon:float)->Tuple[float,float,float,Optional[str],float]:
    best=None; best_grid=None
    for step in (30.0, 90.0, 150.0):
        z = await _fetch_elev_block(lat,lon,step)
        if not z: continue
        flatness = stddev([*z[0],*z[1],*z[2]])
        slope,aspect,octant = slope_aspect_from_grid(z,cell_size_m=step)
        cand = {"z":z,"step":step,"flat":flatness,"slope":slope,"aspect":aspect,"oct":octant,"elev":z[1][1]}
        if best is None: best=cand; best_grid=z
        # preferisci patch con pendenza non nulla e maggiore struttura
        if slope>1.0 and (best["slope"]<=1.0 or flatness>best["flat"]):
            best=cand; best_grid=z
    if not best:
        return 800.0, 5.0, 0.0, None, 0.0
    # fornisci ottante solo se la pendenza è significativa e la patch non è "piatta"
    aspect_oct = best["oct"] if (best["slope"]>=1.0 and best["flat"]>=0.3) else None
    conc = concavity_from_grid(best_grid)
    return float(best["elev"]), round(best["slope"],1), round(best["aspect"],0), aspect_oct, conc

# =============== Habitat auto da OSM ===============
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
                r = await c.post(url, data={"data": q})
                r.raise_for_status()
                j = r.json()
            scores = {"castagno":0.0,"faggio":0.0,"quercia":0.0,"conifere":0.0,"misto":0.0}
            for el in j.get("elements", []):
                local = _score_tags(el.get("tags", {}))
                for k,v in local.items(): scores[k]+=v
            hab, conf = _choose_habitat(scores)
            return hab, conf, scores
        except Exception:
            continue
    return "misto", 0.15, {"castagno":0,"faggio":0,"quercia":0,"conifere":0,"misto":1}

# =============== Microclima/esposizione ===============
def microclimate_from_aspect(aspect_oct: Optional[str], slope_deg: float, rh7: float, sw7: float, tmean7: float) -> float:
    if not aspect_oct or slope_deg < 0.8:
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

# =============== Specie & tips ===============
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

# =============== Meteo (Open-Meteo + OpenWeather) ===============
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

# =============== Lag biologico & flush forecast ===============
def habitat_factor(habitat: str, elev_m: float, today: date) -> float:
    """Moltiplicatore lieve dell'indice base (stabilità)."""
    hab = (habitat or "").lower()
    m = today.month
    is_autumn = m in (8,9,10,11)
    is_summer = m in (6,7,8)
    f = 1.0
    if "castag" in hab:
        if is_autumn: f *= 1.08
    elif "fagg" in hab:
        if is_autumn and elev_m>=700: f *= 1.08
    elif "querc" in hab:
        if is_autumn: f *= 1.05
    elif any(k in hab for k in ("conifer","abete","pino")):
        if (is_autumn and elev_m>=800) or (is_summer and elev_m>=1200): f *= 1.07
    elif "misto" in hab:
        if is_autumn: f *= 1.05
    return f

def dynamic_lag_days(tmean: float, elev: float, rh: float, sw: float, concavity: float, antecedent_api: float, habitat: str) -> int:
    """
    Lag biologico (giorni dal picco piovoso al picco flush).
    Base = 9 giorni; aggiustamenti +-4 in funzione di termica, quota, umidità/irraggiamento, concavità, pre-asciutto e habitat.
    """
    lag = 9
    # termica
    if 16 <= tmean <= 20: lag -= 2
    elif tmean < 12: lag += 2
    elif tmean > 22: lag += 1
    # quota
    if elev >= 1200: lag += 1
    elif elev <= 400 and tmean >= 18: lag -= 1
    # umidità/irraggiamento
    if rh >= 70 and sw < 15000: lag -= 1
    if rh < 50 and sw > 19000: lag += 1
    # concavità (impluvi accelerano)
    if concavity > 0.05: lag -= 1
    if concavity < -0.05: lag += 1
    # antecedente secco/umido
    if antecedent_api < 8: lag += 1  # molto secco → un filo più lento
    if "fagg" in (habitat or "").lower() and elev >= 700:
        lag = lag  # neutro (già incluso con quota)
    lag = int(clamp(lag, 5, 15))
    return lag

def rain_events(times: List[str], rains: List[float]) -> List[Tuple[int,float]]:
    """
    Trova eventi di pioggia (index giorno, mm totali evento).
    Regola: giornate con >=8 mm o coppie consecutive con somma >=12 mm.
    """
    events=[]
    n=min(len(times),len(rains))
    i=0
    while i<n:
        if rains[i] >= 8.0:
            events.append((i, rains[i]))
            i += 1
        elif i+1<n and (rains[i]+rains[i+1]) >= 12.0:
            events.append((i+1, rains[i]+rains[i+1]))  # picco sul secondo
            i += 2
        else:
            i += 1
    return events

def gaussian_kernel(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5*((x-mu)/sigma)**2)

def event_strength(mm: float) -> float:
    """0..1 saturante: 20 mm ~0.63, 40 mm ~0.86, 60 mm ~0.95"""
    return 1.0 - math.exp(-mm/20.0)

def build_flush_forecast(
    past_api: float,              # API calcolato sui 15 gg passati
    tmean7: float, elev: float, rh7: float, sw7: float,
    concavity: float, micro: float, habitat: str,
    timev: List[str], pastN: int, futN: int,
    P_past: List[float], P_fut_blend: List[float],
    reliability: float
) -> Tuple[List[int], List[Dict[str,Any]]]:
    """
    Restituisce: (forecast[10], dettagli_eventi)
    forecast[j] è la probabilità/forza di uscita (0..100) nel giorno j (oggi+j).
    """
    # 1) Eventi passati (ultimi 15 gg)
    ev_past = rain_events(timev[:pastN], P_past)
    # 2) Eventi futuri (prossimi 10 gg)
    ev_fut = []
    for i in range(futN):
        if P_fut_blend[i] >= 8.0 or (i+1<futN and (P_fut_blend[i]+P_fut_blend[i+1])>=12.0):
            if i+1<futN and P_fut_blend[i+1] >= P_fut_blend[i]:
                ev_fut.append((pastN+i+1, P_fut_blend[i]+(P_fut_blend[i+1] if P_fut_blend[i+1]>0 else 0)))
            else:
                ev_fut.append((pastN+i, P_fut_blend[i]))
    # 3) Fuse events (indice assoluto sulla sequenza totale)
    events = []
    seen = set()
    for e in ev_past + ev_fut:
        if e[0] in seen:
            idx = [k for k,(i,_) in enumerate(events) if i==e[0]]
            if idx:
                i0 = idx[0]
                events[i0] = (events[i0][0], events[i0][1] + e[1])
            else:
                events.append(e); seen.add(e[0])
        else:
            events.append(e); seen.add(e[0])
    # 4) Prepara forecast 10 gg
    forecast = [0.0]*futN
    details=[]
    # 5) Per ogni evento, calcola il lag e una gaussiana centrata al giorno del flush
    for (ev_idx, mm_tot) in events:
        lag = dynamic_lag_days(tmean7, elev, rh7, sw7, concavity, past_api, habitat)
        peak_idx = ev_idx + lag  # indice assoluto sulla sequenza totale
        sigma = 2.5 if mm_tot < 25 else 3.0
        amp = event_strength(mm_tot)  # 0..1
        amp *= clamp(0.85 + 0.3*(micro-0.5), 0.75, 1.15)
        if ev_idx >= pastN:
            amp *= (0.5 + 0.5*reliability)
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
    # smoothing 1-2-1
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

# =============== Harvest dinamico (nuovo) ===============
def harvest_expected_string(forecast: List[int], idx_today: int, micro_today: float,
                            habitat_used: str, reliability: float, hours: int) -> Tuple[str, float]:
    """
    Converte lo stato (forecast 10 gg) in un numero atteso per la sessione (2–8h) e in una stringa compatibile UI.
    """
    if not forecast:
        return "0–1 porcini", 0.4

    peak = max(forecast)
    s,e,m = best_window(forecast)
    mean3 = m or 0
    # score di raccolta 0..100 (peso su picco e media finestra)
    score = 0.6*mean3 + 0.4*peak

    # moltiplicatori lievi
    hab = (habitat_used or "").lower()
    hab_k = 1.0
    if "fagg" in hab: hab_k *= 1.05
    elif "castag" in hab: hab_k *= 1.05
    elif "conifer" in hab or "abete" in hab or "pino" in hab: hab_k *= 1.03

    micro_k = clamp(0.85 + 0.3*(micro_today-0.5), 0.75, 1.15)
    rel_k   = clamp(0.85 + 0.15*reliability/0.95, 0.85, 1.0)  # previsioni più affidabili → meno incertezza verso l'alto

    # base: curva convessa in modo che salga davvero con score
    base_2h = (clamp(score,0,100)/25.0)**1.4  # 0.. ~ (4)**1.4 ≈ 6.96
    exp_count = base_2h * (hours/2.0) * hab_k * micro_k * rel_k

    # mappo in classi leggibili
    if exp_count < 0.6:
        s = "0–1 porcini"
    elif exp_count < 1.5:
        s = "1–2 porcini"
    elif exp_count < 3.5:
        s = "2–5 porcini"
    elif exp_count < 6.5:
        s = "4–8 porcini"
    else:
        s = "6–12+ porcini"
    return s, float(round(exp_count,2))

# =============== Spiegazione dinamica (centrata sul flush) ===============
def build_analysis_text(payload: Dict[str,Any]) -> str:
    idx = payload["index"]
    best = payload["best_window"]
    p15 = payload["P15_mm"]; p7 = payload["P7_mm"]
    rh7 = payload.get("RH7_pct", None); sw7 = payload.get("SW7_kj", None)
    tm7 = payload["Tmean7_c"]
    aspect = payload.get("aspect_octant") or "NA"
    slope = payload.get("slope_deg")
    elev = payload.get("elevation_m")
    habitat_used = (payload.get("habitat_used") or "").capitalize() or "Altro"
    habitat_source = payload.get("habitat_source","manuale")
    rain_future = payload["rain_future"]
    future_rain_total = sum(rain_future.values()) if rain_future else 0.0

    out=[]
    out.append(f"<p><strong>Indice attuale: {idx}/100</strong> • Habitat: <strong>{habitat_used}</strong> ({habitat_source}). "
               f"Quota <strong>{elev} m</strong>, pendenza <strong>{slope}°</strong>, esposizione <strong>{aspect or 'NA'}</strong>.</p>")

    out.append("<h4>Come stimiamo i giorni di uscita</h4>")
    out.append("<p>Il modello individua gli <strong>eventi di pioggia</strong> (passati e previsti) e applica un "
               "<strong>ritardo biologico</strong> dipendente da temperatura media, quota, umidità/irraggiamento, "
               "concavità del versante e habitat. Il picco di probabilità di flush viene quindi proiettato nei prossimi 10 giorni.</p>")

    evs = payload.get("flush_events", [])
    if evs:
        out.append("<h4>Eventi di pioggia e finestre stimate</h4><ul>")
        for e in evs:
            when = e.get("event_when")
            mm = e.get("event_mm")
            lag = e.get("lag_days")
            out.append(f"<li>Evento ~<strong>{mm} mm</strong> il <strong>{when}</strong> → "
                       f"flush atteso ~<strong>{lag} giorni</strong> dopo (condizionato a T, quota, microclima).</li>")
        out.append("</ul>")

    if best and best.get("mean",0)>0:
        s,e,m = best["start"], best["end"], best["mean"]
        out.append(f"<p>Nei prossimi 10 giorni la finestra con <strong>maggiore probabilità di uscita</strong> è tra "
                   f"<strong>giorno {s+1}</strong> e <strong>giorno {e+1}</strong> (media ≈ <strong>{m}</strong>).</p>")

    cause=[]
    cause.append(f"Antecedente: 15 gg = <strong>{p15:.0f} mm</strong> (7 gg: {p7:.0f} mm).")
    cause.append(f"Termica media 7 gg: <strong>{tm7:.1f}°C</strong> (ritardo biologico variabile 5–15 gg).")
    if aspect != "NA":
        if aspect in ("N","NE","NW"): cause.append("Versante fresco/ombroso, conserva umidità.")
        elif aspect in ("S","SE","SW"): cause.append("Versante caldo/irradiato: asciuga più in fretta.")
        else: cause.append("Esposizione intermedia (E/W).")
    if rh7 is not None and sw7 is not None:
        if rh7 < 55 and sw7 > 18000: cause.append("UR% bassa e radiazione alta: flush più lento e concentrato in impluvi/ombra.")
        elif rh7 >= 70 and sw7 < 15000: cause.append("UR% alta e radiazione moderata: ritardo minore e flush più ampio.")
    out.append("<h4>Contesto meteo-microclimatico</h4><ul>" + "".join(f"<li>{c}</li>" for c in cause) + "</ul>")

    spp = species_for_habitat(habitat_used)
    out.append("<h4>Specie di porcini probabili (in base all’habitat)</h4><ul>" + "".join(
        f"<li><em>{s['latin']}</em> — {s['common']}</li>" for s in spp
    ) + "</ul>")

    notes = safety_notes_short()
    out.append("<details><summary>Note rapide di sicurezza</summary><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul></details>")
    return "\n".join(out)

# =============== ENDPOINTS ===============
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
    # fetch paralleli
    om_task=asyncio.create_task(fetch_open_meteo(lat,lon,past=15,future=10))
    ow_task=asyncio.create_task(fetch_openweather(lat,lon))
    dem_task=asyncio.create_task(fetch_elevation_grid_multiscale(lat,lon))
    osm_task=asyncio.create_task(fetch_osm_habitat(lat,lon)) if autohabitat==1 else None

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

    # Indicatori attuali (contesto)
    API_val=api_index(P_past_om,half_life=half); ET7=sum(ET0_p_om[-7:]) if ET0_p_om else 0.0
    RH7=sum(RH_p_om[-7:])/max(1,len(RH_p_om[-7:])) if RH_p_om else 60.0
    SW7=sum(SW_p_om[-7:])/max(1,len(SW_p_om[-7:])) if SW_p_om else 15000.0
    Tfit_today=temperature_fit(float(Tmin_p_om[-1]),float(Tmax_p_om[-1]),float(Tm_p_om[-1]))
    micro_today = microclimate_from_aspect(aspect_oct or None, float(slope_deg), float(RH7), float(SW7), float(sum(map(float,Tm_p_om[-7:]))/7.0))
    moisture=max(0.0,min(1.0,(API_val-0.6*ET7)/40.0+0.6))
    dryness_penalty = 0.0
    if SW7>18000 and RH7<55: dryness_penalty = min(0.25, (SW7-18000)/12000 * (55-RH7)/55)
    idx_today = int(round(100.0*(0.54*moisture + 0.36*Tfit_today + 0.10*micro_today) * (1.0 - dryness_penalty)))
    h_factor = habitat_factor(habitat_used, elev_m, datetime.utcnow().date())
    idx_today = int(round(max(0, min(100, idx_today*h_factor))))

    # Affidabilità delle previsioni
    reliability = reliability_from_sources(P_fut_ow[:ow_len],P_fut_om[:ow_len]) if ow_len else 0.6

    # Tabelle pioggia
    rain_past={d["time"][i] if isinstance(d,dict) else om["daily"]["time"][i]: round(P_past_om[i],1) for i,d in [(i,om["daily"]) for i in range(min(pastN,len(om["daily"]["time"])))]}
    rain_future={om["daily"]["time"][pastN+i] if pastN+i<len(om["daily"]["time"]) else f"+{i+1}d":round(P_fut_blend[i],1) for i in range(futN)}

    # Flush forecast
    Tm7=sum(map(float,Tm_p_om[-7:]))/max(1,len(Tm_p_om[-7:]))
    flush_forecast, events_details = build_flush_forecast(
        past_api=API_val, tmean7=Tm7, elev=elev_m, rh7=RH7, sw7=SW7,
        concavity=concavity, micro=micro_today, habitat=habitat_used,
        timev=om["daily"]["time"], pastN=pastN, futN=futN,
        P_past=P_past_om, P_fut_blend=P_fut_blend,
        reliability=reliability
    )
    s,e,m=best_window(flush_forecast)

    # Harvest atteso (nuovo, dipende da hours)
    harvest_str, harvest_mean = harvest_expected_string(
        forecast=flush_forecast, idx_today=idx_today, micro_today=micro_today,
        habitat_used=habitat_used, reliability=reliability, hours=hours
    )

    # Response
    P7=sum(P_past_om[-7:])
    response_data = {
        "lat": lat, "lon": lon,
        "elevation_m": round(elev_m),
        "slope_deg": slope_deg, "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct if aspect_oct else "NA",
        "concavity": round(concavity,3),
        "API_star_mm": round(API_val,1),
        "P7_mm": round(P7,1), "P15_mm": round(sum(P_past_om),1),
        "ET0_7d_mm": round(ET7,1),
        "RH7_pct": round(RH7,1),
        "SW7_kj": round(SW7,0),
        "Tmean7_c": round(Tm7,1),
        "index": idx_today,
        "forecast": [int(x) for x in flush_forecast],
        "best_window": {"start": s, "end": e, "mean": m},
        "harvest_estimate": harvest_str,
        "harvest_detail": {"expected_mean": harvest_mean, "hours": hours},
        "reliability": round(reliability,3),
        "rain_past": rain_past,
        "rain_future": rain_future,
        "habitat_used": habitat_used,
        "habitat_source": habitat_source,
        "auto_habitat_scores": auto_scores,
        "auto_habitat_confidence": round(auto_conf,3),
        "flush_events": events_details,
    }
    response_data["dynamic_explanation"] = build_analysis_text(response_data)
    return response_data

