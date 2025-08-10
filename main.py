# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta

app = FastAPI(title="Trova Porcini API (Definitive)", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

HEADERS = {"User-Agent":"Trovaporcini/2.1 (+https://github.com/tony66j/porcinicast-2)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")

# ----------------- Utilities -----------------
def half_life_coeff(d: float) -> float: return 0.5**(1.0/max(1.0,d))
def api_index(p: List[float], h: float=8.0) -> float:
    k=half_life_coeff(h); api=0.0
    for val in p: api=k*api+(val or 0.0)
    return api
def temperature_fit(tmin: float, tmax: float, tmean: float) -> float:
    if tmin<1 or tmax>32: return 0.0
    if tmean<=6: b=max(0.0,(tmean)/6.0*0.3)
    elif tmean<=10: b=0.3+0.2*((tmean-6)/4.0)
    elif tmean<=18: b=0.5+0.5*((tmean-10)/8.0)
    elif tmean<=22: b=0.8-0.2*((tmean-18)/4.0)
    else: b=max(0.0,0.6-0.6*((tmean-22)/10.0))
    if tmin<6: b*=max(0.3,1-(6-tmin)/8.0)
    if tmax>24: b*=max(0.3,1-(tmax-24)/10.0)
    return max(0.0, min(1.0, b))
def final_index(api:float, et0:float, t_fit:float, slope:float, alt:float) -> int:
    moisture=max(0.0,min(1.0,(api-0.6*et0)/40.0+0.6))
    alt_factor = 1.0 - abs(alt - 900.0) / 800.0
    slope_factor = 1.0 - (slope / 45.0)
    base_score = 100.0 * (0.55*moisture + 0.45*t_fit)
    final_score = base_score * max(0.5, alt_factor) * max(0.5, slope_factor)
    return int(round(max(0.0,min(100.0,final_score))))
def best_window(v: List[int]) -> Tuple[int,int,int]:
    if len(v)<3: return (0,0,0)
    best=(0,2,round((v[0]+v[1]+v[2])/3))
    for i in range(1,len(v)-2):
        m=round((v[i]+v[i+1]+v[i+2])/3)
        if m>best[2]: best=(i,i+2,m)
    return best
def reliability_from_sources(ow:List[float],om:List[float]) -> float:
    n=min(len(ow),len(om));
    if n==0: return 0.55
    diffs=[abs((ow[i] or 0.0)-(om[i] or 0.0)) for i in range(n)]
    return max(0.25, min(0.95, 0.95/(1.0+sum(diffs)/n/6.0)))

# AGGIORNATO PER 4 ORE DI RACCOLTA
def yield_estimate(idx: int) -> str:
    if idx >= 80: return "10-20+ porcini"
    if idx >= 60: return "5–10 porcini"
    if idx >= 45: return "2–4 porcini"
    return "0–2 porcini"
    
def slope_aspect_from_grid(z:List[List[float]],c:float=30.0)->Tuple[float,float,str]:
    dzdx=((z[0][2]+2*z[1][2]+z[2][2])-(z[0][0]+2*z[1][0]+z[2][0]))/(8*c)
    dzdy=((z[2][0]+2*z[2][1]+z[2][2])-(z[0][0]+2*z[0][1]+z[0][2]))/(8*c)
    slope=math.degrees(math.atan(math.hypot(dzdx,dzdy)))
    aspect=(math.degrees(math.atan2(dzdx,dzdy))+360)%360
    octs=["N","NE","E","SE","S","SW","W","NW","N"]
    return round(slope,1),round(aspect,0),octs[int(((aspect%360)+22.5)//45)]

# ----------------- Providers (unchanged) -----------------
async def fetch_open_meteo(lat:float,lon:float,past:int=15,future:int=10)->Dict[str,Any]:
    url="https://api.open-meteo.com/v1/forecast"; params={"latitude":lat,"longitude":lon,"timezone":"auto","daily":",".join(["precipitation_sum","temperature_2m_mean","temperature_2m_min","temperature_2m_max","et0_fao_evapotranspiration","relative_humidity_2m_mean"]),"past_days":past,"forecast_days":future}
    async with httpx.AsyncClient(timeout=30,headers=HEADERS) as c: r=await c.get(url,params=params); r.raise_for_status(); return r.json()
async def fetch_openweather(lat:float,lon:float)->Dict[str,Any]:
    if not OWM_KEY: return {}
    url="https://api.openweathermap.org/data/3.0/onecall"; params={"lat":lat,"lon":lon,"exclude":"minutely,hourly,current,alerts","units":"metric","lang":"it","appid":OWM_KEY}
    try:
        async with httpx.AsyncClient(timeout=30,headers=HEADERS) as c: r=await c.get(url,params=params); r.raise_for_status(); return r.json()
    except Exception: return {}
async def fetch_elevation_grid(lat:float,lon:float,step_m:float=30.0)->Optional[List[List[float]]]:
    try:
        dlat_m,dlon_m=1/111320.0,1/(111320.0*math.cos(math.radians(lat))); dlat,dlon=step_m*dlat_m,step_m*dlon_m
        coords=[{"latitude":lat+dr*dlat,"longitude":lon+dc*dlon} for dr in(-1,0,1) for dc in(-1,0,1)]
        async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c: r=await c.post("https://api.open-elevation.com/api/v1/lookup",json={"locations":coords}); r.raise_for_status(); j=r.json()
        vals=[p["elevation"] for p in j["results"]]; return [vals[0:3],vals[3:6],vals[6:9]]
    except Exception: return None
async def geocode_nominatim(q: str)->Dict[str,Any]:
    url="https://nominatim.openstreetmap.org/search"; params={"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=20,headers=HEADERS) as c: r=await c.get(url,params=params); r.raise_for_status(); data=r.json()
    if not data: raise HTTPException(404,"Località non trovata")
    return {"lat":float(data[0]["lat"]),"lon":float(data[0]["lon"]),"display":data[0].get("display_name","")}

# ----------------- Field Guide (unchanged) -----------------
def generate_field_guide(data: Dict[str, Any]) -> Dict[str, Any]:
    altitude = data["elevation_m"]; month = datetime.now().month
    forest_type = "Misto Latifoglie/Conifere"
    if altitude < 600: forest_type = "Bosco di Latifoglie (Quercia, Castagno)"
    elif altitude > 1400: forest_type = "Bosco di Conifere (Abete, Pino)"
    else: forest_type = "Faggeta"
    other_porcini = []
    if "Latifoglie" in forest_type and month >= 5 and month <= 9: other_porcini.append({"name": "Porcino Estivo (B. aestivalis)", "desc": "Ama il caldo, cresce con querce e castagni.", "img": "https://i.imgur.com/gGkyqApp.jpg"})
    if "Latifoglie" in forest_type and month >= 8 and month <= 10: other_porcini.append({"name": "Porcino Nero (B. aereus)", "desc": "Termofilo, predilige boschi caldi e asciutti.", "img": "https://i.imgur.com/5wO5hS3p.jpg"})
    if "Conifere" in forest_type and month >= 8 and month <= 10: other_porcini.append({"name": "Porcino dei Pini (B. pinophilus)", "desc": "Tipico delle pinete montane, cappello rossiccio.", "img": "https://i.imgur.com/r3C0123p.jpg"})
    to_avoid = [
        {"name": "Boleto di Satana (R. satanas)", "desc": "TOSSICO. Cappello biancastro, pori rossi, viraggio intenso.", "img": "https://i.imgur.com/xT5e533p.jpg"},
        {"name": "Boleto Amaro (T. felleus)", "desc": "NON COMMESTIBILE. Pori rosati, sapore amarissimo.", "img": "https://i.imgur.com/q7z5g9vp.jpg"}
    ]
    where_to_look = []
    temp_avg_7d=data["Tmean7_c"]; slope=data["slope_deg"]; aspect=data["aspect_octant"]
    if temp_avg_7d>22: where_to_look.append(f"**Cerca al fresco:** Con temperature elevate, concentra la ricerca sui versanti esposti a **Nord e Nord-Ovest ({aspect})**.")
    else: where_to_look.append(f"**Sfrutta il sole:** Con temperature miti o fresche, i versanti esposti a **Sud e Sud-Est** si scaldano maggiormente. L'esposizione attuale è **{aspect}**.")
    if slope>20: where_to_look.append(f"**Attenzione alla pendenza:** In aree ripide ({slope:.0f}°), cerca nelle **piccole terrazze naturali** e negli impluvi.")
    else: where_to_look.append(f"**Pendenza favorevole:** La pendenza moderata del sito ({slope:.0f}°) favorisce un buon drenaggio. Ispeziona le aree leggermente incavate.")
    where_to_look.append(f"**L'habitat ideale:** A {altitude}m, sei in un **{forest_type}**. Cerca vicino agli alberi più maturi.")
    return {"where_to_look_tips":where_to_look,"other_porcini":other_porcini,"to_avoid":to_avoid}

# ----------------- Dynamic Explanation (unchanged) -----------------
def generate_dynamic_explanation(data: Dict[str, Any]) -> str:
    idx=data["index"]; p15=data["P15_mm"]; tm7=data["Tmean7_c"]; aspect=data["aspect_octant"]; p_fut=data["rain_future"]
    last_rain_days_ago,last_rain_mm = -1,0
    past_rain_items=sorted(data["rain_past"].items(),key=lambda i:i[0],reverse=True)
    for i,(date,mm) in enumerate(past_rain_items):
        if mm>8.0: last_rain_days_ago=i; last_rain_mm=mm; break
    text=f"Il potenziale di raccolta attuale è stimato come **{'buono' if idx>60 else 'moderato' if idx>40 else 'basso'} (indice {idx})**. "
    if idx>40:
        if last_rain_days_ago!=-1 and last_rain_days_ago<=12: text+=f"Questa valutazione è guidata principalmente dalle piogge di circa **{last_rain_mm:.0f} mm** cadute {last_rain_days_ago} giorni fa. "
        else: text+=f"La valutazione si basa su un equilibrio idrico favorevole accumulato, con un totale di **{p15:.0f} mm** di precipitazioni nelle ultime due settimane. "
        text+=f"Le temperature medie recenti, intorno ai **{tm7:.1f}°C**, si collocano nella finestra ottimale per l'attività del micelio. "
    else:
        if p15<20: text+=f"La causa principale è la **carenza di precipitazioni significative** (solo **{p15:.0f} mm** negli ultimi 15 giorni). "
        elif tm7<10 or tm7>22: text+=f"Nonostante le piogge, le temperature medie di **{tm7:.1f}°C** sono attualmente {'troppo basse' if tm7<10 else 'troppo elevate'}, inibendo il micelio. "
        else: text+="Manca un vero 'shock' idro-termico recente, essenziale per una fruttificazione abbondante. "
    text+=f"L'esposizione del versante verso **{aspect}** contribuisce a modulare il microclima locale. "
    future_rain_total=sum(p_fut.values())
    if future_rain_total>15: text+=f"Guardando al futuro, le previsioni indicano l'arrivo di circa **{future_rain_total:.0f} mm** di pioggia. Se confermate, un nuovo ciclo potrebbe iniziare, con una finestra di raccolta ottimale prevista tra 10-15 giorni da quel momento."
    else: text+="Le previsioni non mostrano eventi piovosi importanti, quindi il potenziale è destinato a diminuire gradualmente."
    return text

# ----------------- Main Endpoint (unchanged) -----------------
@app.get("/api/health")
async def health(): return {"ok":True}
@app.get("/api/geocode")
async def api_geocode(q:str): return await geocode_nominatim(q)
@app.get("/api/score")
async def api_score(lat:float=Query(...),lon:float=Query(...),half:float=Query(8.0,gt=3.0,lt=20.0)):
    om_task=asyncio.create_task(fetch_open_meteo(lat,lon)); ow_task=asyncio.create_task(fetch_openweather(lat,lon)); dem_task=asyncio.create_task(fetch_elevation_grid(lat,lon))
    om,ow,elev_grid=await asyncio.gather(om_task,ow_task,dem_task)
    d=om["daily"]; timev=d["time"]; P_om=[float(x or 0.0) for x in d["precipitation_sum"]]; Tm_om=d["temperature_2m_mean"]; Tmin_om=d["temperature_2m_min"]; Tmax_om=d["temperature_2m_max"]; ET0_om=d.get("et0_fao_evapotranspiration",[0.0]*len(P_om))
    pastN=15; futN=10; P_past_om=P_om[:pastN]; P_fut_om=P_om[pastN:pastN+futN]; Tmin_p_om,Tmax_p_om,Tm_p_om=Tmin_om[:pastN],Tmax_om[:pastN],Tm_om[:pastN]; Tmin_f_om,Tmax_f_om,Tm_f_om=Tmin_om[pastN:pastN+futN],Tmax_om[pastN:pastN+futN],Tm_om[pastN:pastN+futN]; ET0_p_om=ET0_om[:pastN]
    P_fut_ow:List[float]=[]; Tmin_f_ow:List[float]=[]; Tmax_f_ow:List[float]=[]; Tm_f_ow:List[float]=[]
    if ow and"daily"in ow:
        for day in ow["daily"]: P_fut_ow.append(float(day.get("rain",0.0))); t=day.get("temp",{}); Tmin_f_ow.append(float(t.get("min",0.0))); Tmax_f_ow.append(float(t.get("max",0.0))); Tm_f_ow.append(float(t.get("day",(t.get("min",0.0)+t.get("max",0.0))/2.0)))
    ow_len=min(len(P_fut_ow),futN)
    w_ow,w_om=0.60,0.40
    def blend(a,b,i): return w_ow*a[i]+w_om*b[i] if i<ow_len else b[i]
    P_fut_blend,Tmin_f_blend,Tmax_f_blend,Tm_f_blend=[],[],[],[]
    for i in range(futN): P_fut_blend.append(blend(P_fut_ow,P_fut_om,i)); Tmin_f_blend.append(blend(Tmin_f_ow,Tmin_f_om,i)); Tmax_f_blend.append(blend(Tmax_f_ow,Tmax_f_om,i)); Tm_f_blend.append(blend(Tm_f_ow,Tm_f_om,i))
    if elev_grid: elev_m=float(elev_grid[1][1]); slope_deg,aspect_deg,aspect_oct=slope_aspect_from_grid(elev_grid)
    else: elev_m,slope_deg,aspect_deg,aspect_oct=800.0,5.0,0.0,"N"
    API_val=api_index(P_past_om,half_life=half); ET7=sum(ET0_p_om[-7:]) if ET0_p_om else 0.0
    Tfit_today=temperature_fit(Tmin_p_om[-1],Tmax_p_om[-1],Tm_p_om[-1])
    idx_today=final_index(API_val,ET7,Tfit_today,slope_deg,elev_m)
    scores=[]; k=half_life_coeff(half); rolling=API_val
    for i in range(futN): rolling=k*rolling+(P_fut_blend[i] or 0.0); tfit=temperature_fit(Tmin_f_blend[i],Tmax_f_blend[i],Tm_f_blend[i]); scores.append(final_index(rolling,sum(P_fut_om[max(0,i-6):i+1]),tfit,slope_deg,elev_m))
    s,e,m=best_window(scores)
    reliability=reliability_from_sources(P_fut_ow[:ow_len],P_fut_om[:ow_len]) if ow_len else 0.6
    rain_past={timev[i]:round(P_past_om[i],1) for i in range(min(pastN,len(timev)))}
    rain_future={timev[pastN+i] if pastN+i<len(timev) else f"+{i+1}d":round(P_fut_blend[i],1) for i in range(futN)}
    P15=sum(P_past_om[-15:]); Tm7=sum(Tm_p_om[-7:])/max(1,len(Tm_p_om[-7:]))
    response_data={"lat":lat,"lon":lon,"elevation_m":round(elev_m),"slope_deg":slope_deg,"aspect_deg":aspect_deg,"aspect_octant":aspect_oct,"Tmean7_c":round(Tm7,1),"P15_mm":round(P15,1),"index":idx_today,"forecast":[int(x) for x in scores],"best_window":{"start":s,"end":e,"mean":m},"harvest_estimate":yield_estimate(idx_today),"reliability":round(reliability,3),"rain_past":rain_past,"rain_future":rain_future}
    response_data["dynamic_explanation"] = generate_dynamic_explanation(response_data)
    response_data["field_guide"] = generate_field_guide(response_data)
    return response_data
