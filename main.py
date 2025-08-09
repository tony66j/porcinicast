from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import httpx, math, os

APP = FastAPI(title="TrovaPorcini API v1.0")
app = APP  # <- uvicorn main:app

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UA = "TrovaPorcini/1.0"
HDR = {"User-Agent": UA}

# ----------------- helpers -----------------
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
    out.append("suolo che trattiene umidità" if P14<8 else "apporto idrico recente ok")
    if T7<8: out.append("quote un po’ più basse; esposizioni NE–E")
    elif T7>20: out.append("quote leggermente maggiori e versanti N–NE–NW")
    if slope<5: out.append("evita piani: cerca pendenze 5–20°")
    if "Fagus" in forest: out.append("nel faggio: chiarie e bordi sentiero")
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
    res=[]; seen=set()
    for s in out:
        if s not in seen: seen.add(s); res.append(s)
    return res

def est_caps(score:float, compat:float, uncert:float)->Tuple[float,Tuple[int,int]]:
    # intensità attesa (porcini/ora) in funzione dello score (≥40 inizia a muoversi)
    lam = max(0.0,(score-40)/10.0) * (0.7+0.6*compat) * (1.0-0.5*uncert)
    exp3 = lam*3.0
    lo   = max(0, int(round(exp3*0.6)))
    hi   =        int(round(exp3*1.6))
    return round(exp3,1), (lo, hi)

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

# ---------------- external (fault-tolerant) ----------------
async def safe_get(client, *a, **kw):
    try:
        r=await client.get(*a, **kw); r.raise_for_status(); return r
    except Exception:
        return None

async def safe_post(client, *a, **kw):
    try:
        r=await client.post(*a, **kw); r.raise_for_status(); return r
    except Exception:
        return None

async def geocode(q: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=15, headers={**HDR,"Accept-Language":"it"}) as c:
        r=await safe_get(c,"https://nominatim.openstreetmap.org/search",
                         params={"format":"json","q":q,"addressdetails":1,"limit":1})
    if not r: return {}
    j=r.json()
    if not j: return {}
    return {"lat":float(j[0]["lat"]), "lon":float(j[0]["lon"]), "display":j[0]["display_name"]}

async def open_elev_grid(lat:float, lon:float) -> List[List[float]]:
    # 3x3 griglia Open-Elevation; fallback piatto a quota media 800 m
    step=30.0
    dlat=step/111320.0
    dlon=step/(111320.0*math.cos(math.radians(lat)))
    coords=[{"latitude":lat+dr*dlat, "longitude":lon+dc*dlon}
            for dr in (-1,0,1) for dc in (-1,0,1)]
    async with httpx.AsyncClient(timeout=15, headers=HDR) as c:
        r=await safe_post(c,"https://api.open-elevation.com/api/v1/lookup", json={"locations":coords})
    if not r:
        return [[800,800,800],[800,800,800],[800,800,800]]
    j=r.json(); v=[p.get("elevation",800) for p in j.get("results",[])]
    if len(v)!=9: v=[800]*9
    return [v[0:3], v[3:6], v[6:9]]

async def overpass_forest(lat:float, lon:float, radius:int=800)->Optional[str]:
    q=f"""
[out:json][timeout:25];
(way(around:{radius},{lat},{lon})[natural=wood];
 relation(around:{radius},{lat},{lon})[natural=wood]);
out tags center 60;
"""
    async with httpx.AsyncClient(timeout=30, headers=HDR) as c:
        r=await safe_post(c,"https://overpass-api.de/api/interpreter", data={"data":q})
        if not r:
            r=await safe_post(c,"https://overpass.kumi.systems/api/interpreter", data={"data":q})
    if not r: return None
    j=r.json()
    lab=[]
    for el in j.get("elements",[]):
        t=el.get("tags",{})
        if "leaf_type" in t:
            lt=t["leaf_type"].lower()
            if "broad" in lt: lab.append("broadleaved")
            elif "conifer" in lt: lab.append("coniferous")
        elif "wood" in t:
            w=t["wood"]
            if w in ("deciduous","broadleaved"): lab.append("broadleaved")
            if w in ("coniferous","needleleaved"): lab.append("coniferous")
    if not lab: return None
    return "broadleaved" if lab.count("broadleaved")>=lab.count("coniferous") else "coniferous"

async def meteo(lat:float, lon:float)->Dict[str,Any]:
    base="https://api.open-meteo.com/v1/forecast"
    p={"latitude":lat,"longitude":lon,
       "daily":",".join(["precipitation_sum","temperature_2m_mean","relative_humidity_2m_mean","et0_fao_evapotranspiration"]),
       "past_days":14,"forecast_days":10,"timezone":"auto"}
    async with httpx.AsyncClient(timeout=25, headers=HDR) as c:
        r=await safe_get(c,base,params=p)
    return r.json() if r else {"daily":{"precipitation_sum":[0]*24,"temperature_2m_mean":[16]*24}}

# ---------------- endpoints ----------------
@APP.get("/api/health")
def health(): return {"ok":True}

@APP.get("/api/geocode")
async def api_geocode(q: str): return await geocode(q)

@APP.get("/api/score")
async def api_score(lat: float, lon: float, day:int=Query(0,ge=0,le=9)):
    degraded=False
    # terreno
    elev9=await open_elev_grid(lat,lon)
    elev=float(elev9[1][1])
    slope, aspect_deg = slope_aspect_from_grid(elev9,30.0)
    aspect_oct = deg_to_octant(aspect_deg) if aspect_deg is not None else "N"
    # bosco
    kind = await overpass_forest(lat,lon)
    flbl = forest_label(kind, elev)
    if kind is None: degraded=True
    # meteo (14 passati + 10 futuri)
    m = await meteo(lat,lon)
    d = m.get("daily",{})
    prec = d.get("precipitation_sum",[0.0]*24)
    temp = d.get("temperature_2m_mean",[16.0]*24)
    rh   = d.get("relative_humidity_2m_mean",[None]*len(prec))
    past14 = prec[:14]
    futP   = prec[14:24]
    futT   = temp[14:24]
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
    # uncertainty
    uncert=0.2 + (0.2 if kind is None else 0.0) + (0.1 if all(v is None for v in rh) else 0.0)
    uncert=clamp(uncert,0.0,1.0)
    # score 10 gg + stime cappelli
    scores=[]; tips10=[]; why10=[]; caps10=[]; capsrng10=[]
    roll=P14
    for i in range(10):
        roll = max(0.0, roll*(13/14) + (futP[i] if i<len(futP) else 0.0))
        s, br = comp_score(roll, futT[i], elev, aspect_oct, flbl, month)
        scores.append(int(round(s)))
        tips10.append(tips(futT[i], roll, slope, aspect_oct, flbl))
        why10.append(f"P14={round(roll,1)} mm; T7={round(futT[i],1)} °C; quota {int(round(elev))} m; esposizione {aspect_oct}; bosco {flbl}")
        mean_caps, rng = est_caps(s, br["compat"], uncert)
        caps10.append(mean_caps); capsrng10.append(rng)
    s0,s2,smean=window3(scores)
    return {
        "ok": True, "degraded": degraded,
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
        "caps3_10": caps10,
        "caps3_range10": capsrng10,
        "best_window": {"start":s0,"end":s2,"mean":smean},
        "series": {
            "precip_past14": [round(x,1) for x in past14],
            "precip_next10": [round(x,1) for x in futP],
            "tmean_next10": [round(x,1) for x in futT]
        },
        "species_probable": species(flbl, lat, elev, month),
        "uncertainty": round(uncert,2)
    }



