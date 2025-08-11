# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta, date

app = FastAPI(title="Trova Porcini API (stable+micology+DEM-multiscale)", version="1.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

HEADERS = {"User-Agent":"Trovaporcini/1.4 (+site)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")

# ----------------- Utilità -----------------
def half_life_coeff(days: float) -> float: 
    return 0.5**(1.0/max(1.0,days))

def api_index(precip: List[float], half_life: float=8.0) -> float:
    k=half_life_coeff(half_life); api=0.0
    for p in precip: api=k*api+(p or 0.0)
    return api

def temperature_fit(tmin: float, tmax: float, tmean: float) -> float:
    # Range ottimale morbido per porcini (letteratura pop+accademica)
    # > penalità continue, no "gradini" duri
    if tmin < -2 or tmax > 33: 
        return 0.0
    # base su tmean
    if tmean<=6: base=max(0.0,(tmean)/6.0*0.35)
    elif tmean<=10: base=0.35+0.25*((tmean-6)/4.0)      # 0.35 -> 0.60
    elif tmean<=18: base=0.60+0.30*((tmean-10)/8.0)     # 0.60 -> 0.90
    elif tmean<=22: base=0.90-0.20*((tmean-18)/4.0)     # 0.90 -> 0.70
    else: base=max(0.0,0.70-0.70*((tmean-22)/10.0))     # 22..32 → 0
    # correzioni estreme
    if tmin<6:  base*=max(0.35, 1-(6-tmin)/8.0)
    if tmax>26: base*=max(0.35, 1-(tmax-26)/8.0)
    return max(0.0, min(1.0, base))

def final_index(api_val:float, et0_7:float, t_fit:float, rh7:float, sw7:float, micro:float) -> int:
    # Umidità e irraggiamento modulano l'umidità del suolo percepita
    moisture=max(0.0,min(1.0,(api_val-0.6*et0_7)/40.0+0.6))
    # penalità se radiazione alta e RH bassa
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
        # scala dolce verso l'alto
        if base=="0–1 porcini": base="1–2 porcini"
        elif base=="1–2 porcini": base="2–4 porcini"
        elif base=="2–5 porcini": base="3–7 porcini"
        else: base="8–12+ porcini"
    return base

def slope_aspect_from_grid(z:List[List[float]],cell_size_m:float=30.0)->Tuple[float,float,Optional[str]]:
    # Horn 3x3
    dzdx=((z[0][2]+2*z[1][2]+z[2][2])-(z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_size_m)
    dzdy=((z[2][0]+2*z[2][1]+z[2][2])-(z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_size_m)
    slope=math.degrees(math.atan(math.hypot(dzdx,dzdy)))
    # azimut (0=N), attenzione ai segni: usiamo convenzione meteo
    aspect=(math.degrees(math.atan2(dzdx, dzdy))+360.0)%360.0
    octs=["N","NE","E","SE","S","SW","W","NW","N"]
    octant=octs[int(((aspect%360)+22.5)//45)]
    return round(slope,1),round(aspect,0),octant

def stddev(xs: List[float]) -> float:
    if not xs: return 0.0
    m=sum(xs)/len(xs)
    return (sum((x-m)**2 for x in xs)/len(xs))**0.5

# ----------------- Providers -----------------
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
    # Prova 3 scale: scegli quella con pendenza > soglia o con varianza maggiore
    best = None
    for step in (30.0, 90.0, 150.0):
        z = await _fetch_elev_block(lat,lon,step)
        if not z: 
            continue
        flatness = stddev([*z[0],*z[1],*z[2]])
        slope,aspect,octant = slope_aspect_from_grid(z,cell_size_m=step)
        cand = {"z":z,"step":step,"flat":flatness,"slope":slope,"aspect":aspect,"oct":octant,"elev":z[1][1]}
        if best is None: best=cand
        # privilegia pendenza >1° e maggiore varianza
        if slope>1.0 and (best["slope"]<=1.0 or flatness>best["flat"]):
            best=cand
    if not best:
        return 800.0, 5.0, 0.0, None
    # se comunque quasi piano, nessuna esposizione affidabile
    if best["slope"]<0.8 or best["flat"]<0.5:
        return float(best["elev"]), round(best["slope"],1), round(best["aspect"],0), None
    return float(best["elev"]), round(best["slope"],1), round(best["aspect"],0), best["oct"]

async def geocode_nominatim(q: str)->Dict[str,Any]:
    url="https://nominatim.openstreetmap.org/search"
    params={"format":"json","q":q,"addressdetails":1,"limit":1,"email":"info@example.com"}
    async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c:
        r=await c.get(url,params=params); r.raise_for_status(); data=r.json()
    if not data: raise HTTPException(404,"Località non trovata")
    return {"lat":float(data[0]["lat"]),"lon":float(data[0]["lon"]),"display":data[0].get("display_name","")}

# ----------------- Spiegazione dinamica -----------------
def _days_ago(iso_date: str) -> Optional[int]:
    try:
        d = datetime.strptime(iso_date, "%Y-%m-%d").date()
        return (datetime.utcnow().date() - d).days
    except Exception:
        return None

def generate_dynamic_explanation(data: Dict[str, Any]) -> str:
    idx = data["index"]
    p15 = data["P15_mm"]
    tm7 = data["Tmean7_c"]
    rh7 = data.get("RH7_pct", None)
    sw7 = data.get("SW7_kj", None)
    aspect = data.get("aspect_octant") or "NA"
    slope = data.get("slope_deg")
    p_fut = data["rain_future"]

    # Pioggia significativa più recente
    last_sig_days_ago = None
    last_sig_mm = 0
    for date_iso, mm in sorted(data["rain_past"].items(), key=lambda kv: kv[0], reverse=True):
        if mm > 8.0:
            last_sig_days_ago = _days_ago(date_iso)
            last_sig_mm = mm
            break

    text = f"Potenziale attuale: **{'buono' if idx > 60 else 'moderato' if idx > 40 else 'basso'} (indice {idx})**. "

    if idx > 40:
        if last_sig_days_ago is not None and 0 <= last_sig_days_ago <= 12:
            text += f"Spinta dovuta a ~**{last_sig_mm:.0f} mm** caduti **{last_sig_days_ago} giorni fa**. "
        else:
            text += f"Equilibrio idrico favorevole nelle ultime due settimane (**{p15:.0f} mm**). "
        text += f"Le T\u2099 medie recenti (**{tm7:.1f}°C**) sono nella finestra utile per la fruttificazione. "
    else:
        if p15 < 20:
            text += f"**Scarsità di piogge**: solo **{p15:.0f} mm** negli ultimi 15 giorni. "
        elif tm7 < 10 or tm7 > 22:
            text += f"Temperature medie (**{tm7:.1f}°C**) attualmente {'troppo basse' if tm7 < 10 else 'troppo alte'}. "

    if aspect != "NA":
        text += f"Il versante **{aspect}** (pendenza {slope}°) modula il microclima locale; "
        if aspect in ("N","NW","NE"):
            text += "in genere più fresco/umido, utile nei periodi caldi. "
        elif aspect in ("S","SE","SW"):
            text += "in genere più caldo/irradiato: evita ore centrali con tempo secco. "
        else:
            text += "microclima intermedio. "
    else:
        text += "Area quasi pianeggiante: esposizione non determinante. "

    if rh7 is not None and sw7 is not None:
        if rh7 < 55 and sw7 > 18000:
            text += "Attenzione a **bassa UR% e forte irraggiamento**: possibile essiccamento superficiale. "
        elif rh7 >= 70 and sw7 < 15000:
            text += "UR elevata e irraggiamento moderato favoriscono la persistenza dell’umidità. "

    future_rain_total = sum(p_fut.values())
    if future_rain_total > 15:
        text += f"Nei prossimi 10 gg attesi ~**{future_rain_total:.0f} mm**: nuova finestra ottimale probabile ~**10–14 giorni** dopo l’evento principale."
    else:
        text += "Previsioni senza piogge rilevanti: potenziale in graduale calo senza nuovi impulsi."

    return text

# ----------------- Fattori habitat -----------------
def habitat_factor(habitat: str, elev_m: float, today: date) -> float:
    """Moltiplicatore lieve (±5–10%)."""
    hab = (habitat or "").lower()
    m = today.month
    is_autumn = m in (8,9,10,11)
    is_summer = m in (6,7,8)
    f = 1.0
    if "castagno" in hab or "chestnut" in hab:
        if is_autumn: f *= 1.08
    elif "faggio" in hab or "beech" in hab:
        if is_autumn and elev_m>=700: f *= 1.08
    elif "querc" in hab or "oak" in hab:
        if is_autumn: f *= 1.05
    elif "conifer" in hab or "abete" in hab or "pino" in hab:
        if (is_autumn and elev_m>=800) or (is_summer and elev_m>=1200):
            f *= 1.07
    elif "misto" in hab:
        if is_autumn: f *= 1.05
    return f

def microclimate_from_aspect(aspect_oct: Optional[str], slope_deg: float, rh7: float, sw7: float, tmean7: float) -> float:
    """-0.15..+0.15 circa, trasforma in 0..1 per final_index."""
    if not aspect_oct or slope_deg < 0.8:
        return 0.5
    bonus = 0.0
    if aspect_oct in ("N","NE","NW"):
        if tmean7 >= 16 and rh7 < 65: bonus += 0.10
        if sw7 > 18000: bonus += 0.05
    elif aspect_oct in ("S","SE","SW"):
        if tmean7 >= 18 and sw7 > 18000 and rh7 < 60: bonus -= 0.10
    # slope attenua/rafforza
    k = min(1.0, max(0.5, slope_deg/12.0))
    bonus *= k
    return max(0.0, min(1.0, 0.5 + bonus))

# ----------------- Endpoint -----------------
@app.get("/api/health")
async def health(): 
    return {"ok":True,"time":datetime.now(timezone.utc).isoformat()}

@app.get("/api/geocode")
async def api_geocode(q:str): 
    return await geocode_nominatim(q)

@app.get("/api/score")
async def api_score(
    lat:float=Query(...), lon:float=Query(...), 
    half:float=Query(8.0,gt=3.0,lt=20.0),
    habitat:str=Query("", description="castagno,faggio,quercia,conifere,misto,altro"),
    hours:int=Query(2, ge=2, le=8)
):
    om_task=asyncio.create_task(fetch_open_meteo(lat,lon,past=15,future=10))
    ow_task=asyncio.create_task(fetch_openweather(lat,lon))
    dem_task=asyncio.create_task(fetch_elevation_grid_multiscale(lat,lon))
    om,ow,(elev_m,slope_deg,aspect_deg,aspect_oct)=await asyncio.gather(om_task,ow_task,dem_task)

    d=om["daily"]; timev=d["time"]
    P_om=[float(x or 0.0) for x in d["precipitation_sum"]]
    Tmin_om, Tmax_om, Tm_om = d["temperature_2m_min"], d["temperature_2m_max"], d["temperature_2m_mean"]
    ET0_om=d.get("et0_fao_evapotranspiration",[0.0]*len(P_om))
    RH_om=d.get("relative_humidity_2m_mean",[60.0]*len(P_om))
    SW_om=d.get("shortwave_radiation_sum",[15000.0]*len(P_om))  # kJ/m2

    pastN=15; futN=10
    # split
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

    # Indici
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
    # attenua/esalta forecast in base all'affidabilità (±25%)
    adj_scores=[int(round( (0.5+0.5*reliability) * x )) for x in scores]

    # Rain tables
    rain_past={timev[i]:round(P_past_om[i],1) for i in range(min(pastN,len(timev)))}
    rain_future={timev[pastN+i] if pastN+i<len(timev) else f"+{i+1}d":round(P_fut_blend[i],1) for i in range(futN)}

    P7=sum(P_past_om[-7:]); P15=sum(P_past_om[-15:])
    Tm7=sum(map(float,Tm_p_om[-7:]))/max(1,len(Tm_p_om[-7:]))

    # Habitat factor
    h_factor = habitat_factor(habitat, elev_m, datetime.utcnow().date())
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
        "habitat_used": habitat or "",
        "tips": []
    }
    response_data["dynamic_explanation"] = generate_dynamic_explanation(response_data)
    return response_data
