from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import math, asyncio, os
import httpx

# ---------------- App ----------------
APP = FastAPI(title="TrovaPorcini API v0.97")
app = APP  # <- uvicorn main:app

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UA = "TrovaPorcini/0.97"
HDR = {"User-Agent": UA}
OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

# -------------- utils ---------------
def clamp(x,a,b): return max(a, min(b, x))

def deg_to_octant(deg: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    i = int((deg % 360) / 45.0 + 0.5) % 8
    return dirs[i]

def slope_aspect_from_grid(e9: List[List[float]], cell_m: float=30.0) -> Tuple[float, Optional[float]]:
    if not e9 or len(e9)!=3 or any(len(r)!=3 for r in e9): return 0.0, None
    z=e9
    dzdx=((z[0][2]+2*z[1][2]+z[2][2])-(z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_m)
    dzdy=((z[2][0]+2*z[2][1]+z[2][2])-(z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_m)
    slope=math.degrees(math.atan(math.hypot(dzdx,dzdy)))
    if dzdx==0 and dzdy==0: aspect=None
    else:
        aspect=math.degrees(math.atan2(dzdx,dzdy))
        if aspect<0: aspect+=360
    return slope, aspect

# -------------- external ------------
async def geocode(q: str) -> Dict[str, Any]:
    url="https://nominatim.openstreetmap.org/search"
    p={"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=15, headers={**HDR,"Accept-Language":"it"}) as c:
        r=await c.get(url,params=p); r.raise_for_status()
        j=r.json()
    if not j: return {}
    return {"lat":float(j[0]["lat"]), "lon":float(j[0]["lon"]), "display":j[0]["display_name"]}

async def open_elev_grid(lat:float, lon:float) -> List[List[float]]:
    step=30.0
    dlat=step/111320.0
    dlon=step/(111320.0*math.cos(math.radians(lat)))
    coords=[]
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            coords.append({"latitude":lat+dr*dlat, "longitude":lon+dc*dlon})
    async with httpx.AsyncClient(timeout=15, headers=HDR) as c:
        r=await c.post("https://api.open-elevation.com/api/v1/lookup", json={"locations":coords})
        r.raise_for_status(); j=r.json()
    v=[p["elevation"] for p in j["results"]]
    return [v[0:3], v[3:6], v[6:9]]

async def overpass_forest(lat:float, lon:float, radius:int=800)->Optional[str]:
    q=f"""
[out:json][timeout:25];
(way(around:{radius},{lat},{lon})[natural=wood];
 relation(around:{radius},{lat},{lon})[natural=wood]);
out tags center 60;
"""
    async with httpx.AsyncClient(timeout=30, headers=HDR) as c:
        r=await c.post("https://overpass-api.de/api/interpreter", data={"data":q})
        r.raise_for_status(); j=r.json()
    lab=[]
    for el in j.get("elements",[]):
        t=el.get("tags",{})
        if "leaf_type" in t:
            lt=t["leaf_type"].lower()
            if "broad" in lt: lab.append("broadleaved")
            elif "conifer" in lt: lab.append("coniferous")
        elif "wood" in t:
            v=t["wood"]
            if v in ("deciduous","broadleaved"): lab.append("broadleaved")
            if v in ("coniferous","needleleaved"): lab.append("coniferous")
    if not lab: return None
    return "broadleaved" if lab.count("broadleaved")>=lab.count("coniferous") else "coniferous"

def forest_label(kind: Optional[str], alt: float) -> str:
    if kind=="coniferous": return "Pinus/Abies/Picea"
    if kind=="broadleaved":
        if alt>800: return "Fagus sylvatica"
        if alt>500: return "Castanea sativa"
        return "Quercus spp."
    if alt>1400: return "Pinus/Abies/Picea"
    if alt>900:  return "Fagus sylvatica"
    if alt>500:  return "Castanea sativa"
    return "Quercus spp."

async def meteo(lat:float, lon:float)->Dict[str,Any]:
    base="https://api.open-meteo.com/v1/forecast"
    p={"latitude":lat,"longitude":lon,
       "daily":",".join(["precipitation_sum","temperature_2m_mean","relative_humidity_2m_mean","et0_fao_evapotranspiration"]),
       "past_days":14,"forecast_days":10,"timezone":"auto"}
    async with httpx.AsyncClient(timeout=25, headers=HDR) as c:
        r=await c.get(base,params=p); r.raise_for_status(); return r.json()

# -------------- model ---------------
def season_band(month:int)->Tuple[int,int]:
    if month in (9,10,11): return 400,1500
    if month in (6,7,8):  return 900,1800
    return 300,1000

def comp_score(P14:float,T7:float,alt:float,aspect:str,forest:str,month:int)->Tuple[float,Dict[str,float]]:
    p=clamp(P14/35.0,0,1)
    if T7<5: t=0
    elif T7>24: t=0.2
    else: t=clamp(1.0-abs((T7-16)/11.0),0,1)
    a0,a1=season_band(month)
    if alt<a0: z=max(0.0,1.0-(a0-alt)/400.0)
    elif alt>a1: z=max(0.0,1.0-(alt-a1)/600.0)
    else: z=1.0
    asp_bonus={"N":0.1,"NE":0.15,"E":0.05,"SE":-0.05,"S":-0.1,"SW":-0.05,"W":0.0,"NW":0.1}
    a=clamp(0.5+asp_bonus.get(aspect,0.0),0,1)
    if "Fagus" in forest: b=1.0
    elif "Castanea" in forest: b=0.9
    elif "Quercus" in forest: b=0.7
    else: b=0.5
    s=100.0*(0.38*p+0.28*t+0.18*z+0.10*b+0.06*a)
    return s,{"p14n":round(p,2),"tn":round(t,2),"zn":round(z,2),"compat":round(b,2),"asp":round(a,2)}

def window3(xs:List[int])->Tuple[int,int,float]:
    best=-1;i0=0
    for i in range(0,len(xs)-2):
        m=(xs[i]+xs[i+1]+xs[i+2])/3.0
        if m>best: best=m; i0=i
    return i0,i0+2,round(best,1)

def tips(T7:float,P14:float,slope:float,asp:str,forest:str)->str:
    out=[]
    out.append("cerca zone con suolo che trattiene umidità" if P14<8 else "buon apporto idrico recente")
    if T7<8: out.append("prediligi esposizioni NE–E più miti")
    elif T7>20: out.append("meglio quote leggermente maggiori e versanti N–NE–NW")
    if slope<5: out.append("evita piani: cerca pendenze 5–20°")
    if "Fagus" in forest: out.append("nei faggeti: chiarie e bordi sentiero")
    if "Castanea" in forest: out.append("castagneti: margini e zone muschiose")
    if "Quercus" in forest: out.append("querceti: tratti più freschi/ombreggiati")
    return "; ".join(out)

def species(forest:str,lat:float,alt:float,month:int)->List[str]:
    out=[]
    if "Fagus" in forest:
        out=["Boletus pinophilus","Boletus edulis"] if month>=8 else ["Boletus pinophilus"]
    elif "Castanea" in forest or "Quercus" in forest:
        if lat<43 and month in (6,7,8,9): out.append("Boletus aereus")
        out+=["Boletus reticulatus"]+(["Boletus edulis"] if month>=9 else [])
    else:
        out=["Boletus edulis"]
    # dedup
    seen=set();res=[]
    for s in out:
        if s not in seen: seen.add(s); res.append(s)
    return res

def est_kg(score:float, compat:float, uncert:float, lat:float, month:int)->float:
    base=max(0.0,(score-40)/60.0)
    seas=1.1 if (month in (6,7,8) and lat<43) else 1.0
    kg=0.8*base*(0.6+0.6*compat-0.3*uncert)*seas
    return round(max(0.0,kg),2)

# -------------- endpoints -----------
@APP.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode(q)

@APP.get("/api/score")
async def api_score(lat: float, lon: float, day:int=Query(0,ge=0,le=9)):
    # terreno
    elev9=await open_elev_grid(lat,lon)
    elev=float(elev9[1][1])
    slope, aspect_deg = slope_aspect_from_grid(elev9,30.0)
    aspect_oct = deg_to_octant(aspect_deg) if aspect_deg is not None else "N/A"
    # bosco
    k = await overpass_forest(lat,lon)
    flbl = forest_label(k, elev)
    # meteo (14 passati + 10 futuri)
    m = await meteo(lat,lon)
    d = m["daily"]
    prec = d["precipitation_sum"]; temp = d["temperature_2m_mean"]
    rh = d.get("relative_humidity_2m_mean",[None]*len(prec))
    et0= d.get("et0_fao_evapotranspiration",[0.0]*len(prec))
    past14 = prec[:14]
    futP   = prec[14:14+10]
    futT   = temp[14:14+10]
    # metriche pioggia
    P14 = sum(past14)
    last_rain=0
    for x in reversed(past14):
        if x>=0.5: break
        last_rain+=1
    next_mm=0.0; next_in=None
    for i,mm in enumerate(futP[1:],start=1):
        next_mm += mm
        if next_in is None and mm>=0.5: next_in=i
    month=int(datetime.now(timezone.utc).month)
    # score per 10 giorni
    scores=[]; tips10=[]; why10=[]; kg10=[]; kgNote10=[]
    roll=P14
    uncert=0.3 + (0.2 if k is None else 0.0) + (0.1 if all(v is None for v in rh) else 0.0)
    uncert=clamp(uncert,0.0,1.0)
    for i in range(10):
        roll = max(0.0, roll*(13/14) + (futP[i] if i<len(futP) else 0.0))
        s, br = comp_score(roll, futT[i], elev, aspect_oct, flbl, month)
        scores.append(int(round(s)))
        tips10.append(tips(futT[i], roll, slope, aspect_oct, flbl))
        why10.append(
            f"P14={round(roll,1)} mm; T7={round(futT[i],1)} °C; quota {int(round(elev))} m; esposizione {aspect_oct}; bosco {flbl}"
        )
        kg10.append(est_kg(s, br["compat"], uncert, lat, month))
        kgNote10.append("prudente" if kg10[-1]<=0.6 else ("realistica" if kg10[-1]<=1.2 else "ottimista"))
    s0,s2,smean=window3(scores)
    return {
        "elevation_m": elev,
        "slope_deg": round(slope,1),
        "aspect_deg": round(aspect_deg,1) if aspect_deg is not None else None,
        "aspect_octant": aspect_oct,
        "forest": flbl,
        "last_rain_days": last_rain,
        "next_rain_mm_5d": round(next_mm,1),
        "next_rain_in_days": next_in,
        "scores_next10": scores,
        "tips10": tips10,
        "why10": why10,
        "kg10": kg10,
        "kg_note10": kgNote10,
        "best_window": {"start":s0,"end":s2,"mean":smean},
        "series": {
            "precip_past14": [round(x,1) for x in past14],
            "precip_next10": [round(x,1) for x in futP],
            "tmean_next10": [round(x,1) for x in futT]
        },
        "species_probable": species(flbl, lat, elev, month)
    }



