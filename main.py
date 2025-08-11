# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio, json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta, date

app = FastAPI(title="Trova Porcini API (v1.5.0)", version="1.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

HEADERS = {"User-Agent":"Trovaporcini/1.5 (+site)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
]

# ----------------- Utilità -----------------
def half_life_coeff(days: float) -> float:
    return 0.5**(1.0/max(1.0,days))

def api_index(precip: List[float], half_life: float=8.0) -> float:
    k=half_life_coeff(half_life); api=0.0
    for p in precip: api=k*api+(p or 0.0)
    return api

def temperature_fit(tmin: float, tmax: float, tmean: float) -> float:
    if tmin < -2 or tmax > 33: 
        return 0.0
    if tmean<=6: base=max(0.0,(tmean)/6.0*0.35)
    elif tmean<=10: base=0.35+0.25*((tmean-6)/4.0)
    elif tmean<=18: base=0.60+0.30*((tmean-10)/8.0)
    elif tmean<=22: base=0.90-0.20*((tmean-18)/4.0)
    else: base=max(0.0,0.70-0.70*((tmean-22)/10.0))
    if tmin<6:  base*=max(0.35, 1-(6-tmin)/8.0)
    if tmax>26: base*=max(0.35, 1-(tmax-26)/8.0)
    return max(0.0, min(1.0, base))

def final_index(api_val:float, et0_7:float, t_fit:float, rh7:float, sw7:float, micro:float) -> int:
    moisture=max(0.0,min(1.0,(api_val-0.6*et0_7)/40.0+0.6))
    dryness_penalty = 0.0
    if sw7>18000 and rh7<55: dryness_penalty = min(0.25, (sw7-18000)/12000 * (55-rh7)/55)
    score = 100.0*(0.54*moisture + 0.36*t_fit + 0.10*micro)
    score = score*(1.0 - dryness_penalty)
    return int(round(max(0.0,min(100.0,score))))

def best_window(values: List[int]) -> Tuple[int,int,int]:
    if len(values)<3: return (0,0,0)
    best=(0,2,round((values[0]+values[1]+values[2])/3))
    for i in range(1,len(values)-2):
        m=round((values[i]+values[i+1]+values[i+2])/3)
        if m>best[2]: best=(i,i+2,m)
    return best

def reliability_from_sources(ow_vals:List[float],om_vals:List[float]) -> float:
    n=min(len(ow_vals),len(om_vals))
    if n==0: return 0.6
    diffs=[abs((ow_vals[i] or 0.0)-(om_vals[i] or 0.0)) for i in range(n)]
    avg_diff=sum(diffs)/n
    return max(0.25, min(0.95, 0.95/(1.0+avg_diff/6.0)))

def yield_estimate(idx: int, hours:int=2) -> str:
    base = "0–1 porcini"
    if idx>=80: base="6–10+ porcini"
    elif idx>=60: base="2–5 porcini"
    elif idx>=40: base="1–2 porcini"
    if hours>=4:
        if base=="0–1 porcini": base="1–2 porcini"
        elif base=="1–2 porcini": base="2–4 porcini"
        elif base=="2–5 porcini": base="3–7 porcini"
        else: base="8–12+ porcini"
    return base

def slope_aspect_from_grid(z:List[List[float]],cell_size_m:float=30.0)->Tuple[float,float,Optional[str]]:
    dzdx=((z[0][2]+2*z[1][2]+z[2][2])-(z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_size_m)
    dzdy=((z[2][0]+2*z[2][1]+z[2][2])-(z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_size_m)
    slope=math.degrees(math.atan(math.hypot(dzdx,dzdy)))
    aspect=(math.degrees(math.atan2(dzdx, dzdy))+360.0)%360.0  # 0=N
    octs=["N","NE","E","SE","S","SW","W","NW","N"]
    octant=octs[int(((aspect%360)+22.5)//45)]
    return round(slope,1),round(aspect,0),octant

def stddev(xs: List[float]) -> float:
    if not xs: return 0.0
    m=sum(xs)/len(xs)
    return (sum((x-m)**2 for x in xs)/len(xs))**0.5

# ----------------- Providers Meteo/DEM -----------------
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
    async with httpx.AsyncClient(timeout=30,headers=HEADERS) as c:
        r=await c.get(url,params=params); r.raise_for_status(); 
        return r.json()

async def fetch_openweather(lat:float,lon:float)->Dict[str,Any]:
    if not OWM_KEY: return {}
    url="https://api.openweathermap.org/data/3.0/onecall"
    params={"lat":lat,"lon":lon,"exclude":"minutely,hourly,current,alerts","units":"metric","lang":"it","appid":OWM_KEY}
    try:
        async with httpx.AsyncClient(timeout=30,headers=HEADERS) as c:
            r=await c.get(url,params=params); r.raise_for_status(); 
            return r.json()
    except Exception:
        return {}

async def _fetch_elev_block(lat:float,lon:float,step_m:float)->Optional[List[List[float]]]:
    try:
        deg_lat=1/111320.0; deg_lon=1/(111320.0*max(0.2,math.cos(math.radians(lat))))
        dlat,dlon=step_m*deg_lat,step_m*deg_lon
        coords=[{"latitude":lat+dr*dlat,"longitude":lon+dc*dlon} for dr in(-1,0,1) for dc in(-1,0,1)]
        async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
            r=await c.post("https://api.open-elevation.com/api/v1/lookup",json={"locations":coords})
            r.raise_for_status(); j=r.json()
        vals=[p["elevation"] for p in j["results"]]
        return [vals[0:3],vals[3:6],vals[6:9]]
    except Exception:
        return None

async def fetch_elevation_grid_multiscale(lat:float,lon:float)->Tuple[float,float,float,Optional[str]]:
    best = None
    for step in (30.0, 90.0, 150.0):
        z = await _fetch_elev_block(lat,lon,step)
        if not z: 
            continue
        flatness = stddev([*z[0],*z[1],*z[2]])
        slope,aspect,octant = slope_aspect_from_grid(z,cell_size_m=step)
        cand = {"z":z,"step":step,"flat":flatness,"slope":slope,"aspect":aspect,"oct":octant,"elev":z[1][1]}
        if best is None: best=cand
        if slope>1.0 and (best["slope"]<=1.0 or flatness>best["flat"]):
            best=cand
    if not best:
        return 800.0, 5.0, 0.0, None
    if best["slope"]<0.8 or best["flat"]<0.5:
        return float(best["elev"]), round(best["slope"],1), round(best["aspect"],0), None
    return float(best["elev"]), round(best["slope"],1), round(best["aspect"],0), best["oct"]

# ----------------- Rilevamento Habitat (OSM) -----------------
def _score_tags(tags: Dict[str,str])->Dict[str,float]:
    """Heuristica: ritorna punteggi per categorie habitat"""
    t = {k.lower(): (v.lower() if isinstance(v,str) else v) for k,v in (tags or {}).items()}
    s = {"castagno":0.0,"faggio":0.0,"quercia":0.0,"conifere":0.0,"misto":0.0}
    genus = t.get("genus","")
    species = t.get("species","")
    leaf_type = t.get("leaf_type","")
    leaf_cycle = t.get("leaf_cycle","")
    landuse = t.get("landuse","")
    natural = t.get("natural","")
    wood = t.get("wood","")
    # Genere specifico
    if "castanea" in genus or "castagna" in species:
        s["castagno"] += 3.0
    if "quercus" in genus or "querce" in species:
        s["quercia"] += 3.0
    if "fagus" in genus or "faggio" in species:
        s["faggio"] += 3.0
    if any(g in genus for g in ("pinus","abies","picea","larix")):
        s["conifere"] += 2.5
    # Tipo foglia
    if "needleleaved" in leaf_type:
        s["conifere"] += 1.5
    if "broadleaved" in leaf_type:
        s["misto"] += 0.5  # generico latifoglie
    # Ciclo e legni
    if wood in ("conifer","pine","spruce","fir"):
        s["conifere"] += 1.2
    if wood in ("broadleaved","deciduous"):
        s["misto"] += 0.6
    # Forest generic
    if landuse=="forest" or natural in ("wood","forest"):
        for k in s: s[k] += 0.1
    return s

def _choose_habitat(scores: Dict[str,float])->Tuple[str,float]:
    best = max(scores.items(), key=lambda kv: kv[1])
    total = sum(scores.values()) or 1.0
    conf = min(0.99, max(0.05, best[1]/total))
    return best[0], conf

async def fetch_osm_habitat(lat: float, lon: float, radius_m: int = 400) -> Tuple[str, float, Dict[str,float]]:
    # Overpass query: forest/wood + isolated trees with genus around the point
    around = radius_m
    q = f"""
    [out:json][timeout:25];
    (
      way(around:{around},{lat},{lon})["landuse"="forest"];
      way(around:{around},{lat},{lon})["natural"="wood"];
      relation(around:{around},{lat},{lon})["landuse"="forest"];
      relation(around:{around},{lat},{lon})["natural"="wood"];
      node(around:{around},{lat},{lon})["tree"]["genus"];
    );
    out tags qt;
    """
    errors = []
    for url in OVERPASS_URLS:
        try:
            async with httpx.AsyncClient(timeout=35, headers=HEADERS) as c:
                r = await c.post(url, data={"data": q})
                r.raise_for_status()
                j = r.json()
            scores = {"castagno":0.0,"faggio":0.0,"quercia":0.0,"conifere":0.0,"misto":0.0}
            for el in j.get("elements", []):
                tags = el.get("tags", {})
                local = _score_tags(tags)
                for k,v in local.items(): scores[k]+=v
            hab, conf = _choose_habitat(scores)
            return hab, conf, scores
        except Exception as e:
            errors.append(str(e))
            continue
    # fallback
    return "misto", 0.15, {"castagno":0,"faggio":0,"quercia":0,"conifere":0,"misto":1}

# ----------------- Microclima/esposizione -----------------
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
    return max(0.0, min(1.0, 0.5 + bonus))

# ----------------- Specie & Tips -----------------
def species_for_habitat(hab: str) -> List[Dict[str,str]]:
    h = (hab or "").lower()
    if "querc" in h or "oak" in h:
        return [
            {"latin":"Boletus reticulatus (syn. B. aestivalis)","common":"porcino estivo/quercino"},
            {"latin":"Boletus aereus","common":"porcino nero/bronzo"},
        ]
    if "castag" in h:
        return [
            {"latin":"Boletus aereus","common":"porcino nero/bronzo"},
            {"latin":"Boletus reticulatus","common":"porcino estivo/quercino"},
        ]
    if "fagg" in h or "beech" in h:
        return [
            {"latin":"Boletus edulis","common":"porcino (faggete)"},
            {"latin":"Boletus aereus","common":"(sporadico in latifoglie calde)"},
        ]
    if "conifer" in h or "abete" in h or "pino" in h:
        return [
            {"latin":"Boletus pinophilus","common":"porcino dei pini"},
            {"latin":"Boletus edulis","common":"porcino (abete/pecceta)"},
        ]
    return [
        {"latin":"Boletus edulis agg.","common":"complesso dei porcini"},
        {"latin":"Boletus reticulatus / B. aereus","common":"latifoglie"},
    ]

def tips_for_conditions(hab:str, aspect:str, slope:float, rh7:float, sw7:float, idx:int, p15:float, future_rain_total:float)->List[str]:
    L=[]
    a = aspect or "NA"
    # Micro-siti in base a secco/caldo
    if rh7 < 55 and sw7 > 18000:
        if a in ("N","NE","NW"):
            L.append("Batti versanti ombrosi (N/NE/NW), impluvi e fossi con lettiera spessa; cerca sotto coperture fitte.")
        else:
            L.append("Evita ore centrali sui versanti S/SE/SW; preferisci ombra, impluvi e margini di radure.")
    else:
        L.append("Esplora margini di radure e cambi di pendenza; dopo piogge i bordi dei sentieri drenanti sono produttivi.")

    h=hab.lower()
    if "querc" in h:
        L.append("In querceti individua esemplari maturi isolati e le zone con erba rada su suoli ben drenati.")
    if "castag" in h:
        L.append("Nel castagneto cerca nelle zone con lettiera abbondante ma non fradicia; ottimi i crinali ombreggiati.")
    if "fagg" in h:
        L.append("In faggeta preferisci impluvi freschi e margini di schiarite sopra ~700–900 m in autunno.")
    if any(k in h for k in ("conifer","abete","pino")):
        L.append("In conifera cerca nei cuscini di aghi ben drenanti vicino a piante vetuste o radure luminose.")
    # Dinamica pioggia
    if future_rain_total>15:
        L.append("Se sono attese nuove piogge rilevanti, pianifica un’uscita 7–14 giorni dopo il picco principale.")
    if p15<20 and idx<50:
        L.append("Con poche piogge recenti concentrare la ricerca in zone umide residue: forre, sorgenti, esposizioni nord.")
    return L

def safety_notes_short()->List[str]:
    return [
        "Attenzione a Tylopilus felleus (amaro): reticolo scuro sul gambo, pori che non sono bianchi a lungo.",
        "Evita Rubroboletus (es. R. satanas): colori vivi, viraggio marcato al taglio; molti sono tossici.",
        "Raccogli solo specie che riconosci con certezza e rispetta le normative locali di raccolta."
    ]

# ----------------- Spiegazione dinamica estesa -----------------
def _days_ago(iso_date: str) -> Optional[int]:
    try:
        d = datetime.strptime(iso_date, "%Y-%m-%d").date()
        return (datetime.utcnow().date() - d).days
    except Exception:
        return None

def build_analysis_text(payload: Dict[str,Any]) -> str:
    idx = payload["index"]
    best = payload["best_window"]
    p15 = payload["P15_mm"]; p7 = payload["P7_mm"]
    rh7 = payload.get("RH7_pct", None); sw7 = payload.get("SW7_kj", None)
    tm7 = payload["Tmean7_c"]
    aspect = payload.get("aspect_octant") or "NA"
    slope = payload.get("slope_deg")
    rain_past = payload["rain_past"]; rain_future = payload["rain_future"]
    future_rain_total = sum(rain_future.values()) if rain_future else 0.0
    elev = payload.get("elevation_m")
    habitat_used = (payload.get("habitat_used") or "").capitalize() or "Altro"
    habitat_source = payload.get("habitat_source","manuale")
    # Pioggia significativa più recente
    last_sig_days_ago = None; last_sig_mm = 0
    for date_iso, mm in sorted(rain_past.items(), key=lambda kv: kv[0], reverse=True):
        if mm > 8.0:
            last_sig_days_ago = _days_ago(date_iso); last_sig_mm = mm; break

    # Titolo e quadro sintetico
    out = []
    out.append(f"<p><strong>Indice attuale: {idx}/100</strong> • Habitat usato: <strong>{habitat_used}</strong> ({habitat_source}). "
               f"Quota <strong>{elev} m</strong>, pendenza <strong>{slope}°</strong>, esposizione <strong>{aspect or 'NA'}</strong>.</p>")

    # Perché l'indice è X
    cause = []
    cause.append(f"Equilibrio idrico: ultime 2 settimane <strong>{p15:.0f} mm</strong> (7 gg: {p7:.0f} mm).")
    if last_sig_days_ago is not None and 0 <= last_sig_days_ago <= 12:
        cause.append(f"Trigger piovoso ~<strong>{last_sig_mm:.0f} mm</strong> caduti <strong>{last_sig_days_ago} giorni fa</strong>.")
    cause.append(f"Termica media 7 gg: <strong>{tm7:.1f}°C</strong> (penalità se Tmin ≪ 6°C o Tmax ≫ 26°C).")
    if aspect != "NA":
        if aspect in ("N","NE","NW"):
            cause.append("Versante fresco/ombroso (N/NE/NW) che mitiga disidratazione in fasi calde.")
        elif aspect in ("S","SE","SW"):
            cause.append("Versante caldo/irradiato (S/SE/SW): possibile essiccamento superficiale.")
        else:
            cause.append("Microclima intermedio dato da esposizione E/W.")
    if rh7 is not None and sw7 is not None:
        if rh7 < 55 and sw7 > 18000:
            cause.append("UR% bassa con irraggiamento elevato: applicata lieve penalità per secco superficiale.")
        elif rh7 >= 70 and sw7 < 15000:
            cause.append("UR% alta e irraggiamento moderato: bonus di persistenza umidità.")
    out.append("<h4>Perché questo punteggio?</h4><ul>" + "".join(f"<li>{c}</li>" for c in cause) + "</ul>")

    # Best window e prospettiva
    if best and best.get("mean",0)>0:
        s,e,m = best["start"], best["end"], best["mean"]
        out.append(f"<p>Nei prossimi 10 giorni la finestra più promettente è tra <strong>giorno {s+1}</strong> e <strong>giorno {e+1}</strong> "
                   f"(media indice ≈ <strong>{m}</strong>).")
    if future_rain_total > 15:
        out.append(f" Sono attesi ~<strong>{future_rain_total:.0f} mm</strong> complessivi: pianifica un’uscita <strong>7–14 giorni</strong> dopo il picco piovoso.</p>")
    else:
        out.append(" Senza piogge rilevanti in vista, l’indice tende a calare gradualmente.</p>")

    # Specie probabili
    spp = species_for_habitat(habitat_used)
    out.append("<h4>Specie di porcini probabili (in base all’habitat)</h4><ul>" + "".join(
        f"<li><em>{s['latin']}</em> — {s['common']}</li>" for s in spp
    ) + "</ul>")

    # Dove cercare (tips)
    tips = tips_for_conditions(habitat_used, aspect, float(slope or 0), float(rh7 or 60), float(sw7 or 15000), idx, p15, future_rain_total)
    out.append("<h4>Dove cercare oggi</h4><ul>" + "".join(f"<li>{t}</li>" for t in tips) + "</ul>")

    # Sicurezza
    notes = safety_notes_short()
    out.append("<details><summary>Note rapide di sicurezza</summary><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul></details>")

    return "\n".join(out)

# ----------------- Endpoint -----------------
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
    autohabitat:int=Query(1, description="1=auto rilevamento OSM, 0=solo manuale"),
    hours:int=Query(2, ge=2, le=8)
):
    om_task=asyncio.create_task(fetch_open_meteo(lat,lon,past=15,future=10))
    ow_task=asyncio.create_task(fetch_openweather(lat,lon))
    dem_task=asyncio.create_task(fetch_elevation_grid_multiscale(lat,lon))
    osm_task=asyncio.create_task(fetch_osm_habitat(lat,lon)) if autohabitat==1 else None

    om,ow,(elev_m,slope_deg,aspect_deg,aspect_oct)=await asyncio.gather(om_task,ow_task,dem_task)
    auto_hab, auto_conf, auto_scores = ("",0.0,{})
    if osm_task:
        try:
            auto_hab, auto_conf, auto_scores = await osm_task
        except Exception:
            auto_hab, auto_conf, auto_scores = ("",0.0,{})

    # Scegli habitat da usare
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
        Tm_f_blend.append(blend(Tm_f_ow,Tm_f_om,i) if ow_len else Tm_f_om[i])

    API_val=api_index(P_past_om,half_life=half); ET7=sum(ET0_p_om[-7:]) if ET0_p_om else 0.0
    RH7=sum(RH_p_om[-7:])/max(1,len(RH_p_om[-7:])) if RH_p_om else 60.0
    SW7=sum(SW_p_om[-7:])/max(1,len(SW_p_om[-7:])) if SW_p_om else 15000.0
    Tfit_today=temperature_fit(float(Tmin_p_om[-1]),float(Tmax_p_om[-1]),float(Tm_p_om[-1]))
    micro_today = microclimate_from_aspect(aspect_oct or None, float(slope_deg), float(RH7), float(SW7), float(sum(map(float,Tm_p_om[-7:]))/7.0))
    idx_today=final_index(API_val,ET7,Tfit_today,RH7,SW7,micro_today)

    scores=[]; k=half_life_coeff(half); rolling=API_val
    for i in range(futN):
        rolling=k*rolling+(P_fut_blend[i] or 0.0)
        et7f=sum(ET0_f_om[max(0,i-6):i+1]) if ET0_f_om else 0.0
        tfit=temperature_fit(float(Tmin_f_blend[i]),float(Tmax_f_blend[i]),float(Tm_f_blend[i]))
        micro_f = microclimate_from_aspect(aspect_oct or None, float(slope_deg), float(RH7), float(SW7), float(Tm_f_blend[i]))
        raw = final_index(rolling,et7f,tfit,RH7,SW7,micro_f)
        scores.append(raw)

    s,e,m=best_window(scores)
    reliability=reliability_from_sources(P_fut_ow[:ow_len],P_fut_om[:ow_len]) if ow_len else 0.6
    adj_scores=[int(round( (0.5+0.5*reliability) * x )) for x in scores]

    rain_past={timev[i]:round(P_past_om[i],1) for i in range(min(pastN,len(timev)))}
    rain_future={timev[pastN+i] if pastN+i<len(timev) else f"+{i+1}d":round(P_fut_blend[i],1) for i in range(futN)}

    P7=sum(P_past_om[-7:]); P15=sum(P_past_om[-15:])
    Tm7=sum(map(float,Tm_p_om[-7:]))/max(1,len(Tm_p_om[-7:]))

    # Habitat factor (leggero) su indice
    def habitat_factor(habitat: str, elev_m: float, today: date) -> float:
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

    h_factor = habitat_factor(habitat_used, elev_m, datetime.utcnow().date())
    idx_today = int(round(max(0, min(100, idx_today*h_factor))))
    adj_scores = [int(round(max(0, min(100, v*h_factor)))) for v in adj_scores]
    m = int(round(m*h_factor))

    response_data = {
        "lat": lat, "lon": lon,
        "elevation_m": round(elev_m),
        "slope_deg": slope_deg, "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct if aspect_oct else "NA",
        "API_star_mm": round(API_val,1),
        "P7_mm": round(P7,1), "P15_mm": round(P15,1),
        "ET0_7d_mm": round(ET7,1),
        "RH7_pct": round(RH7,1),
        "SW7_kj": round(SW7,0),
        "Tmean7_c": round(Tm7,1),
        "index": idx_today,
        "forecast": [int(x) for x in adj_scores],
        "best_window": {"start": s, "end": e, "mean": m},
        "harvest_estimate": yield_estimate(idx_today, hours=hours),
        "reliability": round(reliability,3),
        "rain_past": rain_past,
        "rain_future": rain_future,
        "habitat_used": habitat_used,
        "habitat_source": habitat_source,
        "auto_habitat_scores": auto_scores,
        "auto_habitat_confidence": round(auto_conf,3),
        "tips": []
    }
    response_data["dynamic_explanation"] = build_analysis_text(response_data)
    return response_data

