# main.py — TrovaPorcini v4.3-Pro
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, math, asyncio, os, re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

app = FastAPI(title="TrovaPorcini API v4.3-Pro")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- util ----------
def clamp(x,a,b): return max(a, min(b,x))
def deg_to_octant(a):
    d=(a+22.5)%360; return ["N","NE","E","SE","S","SW","W","NW"][int(d//45)]
def parse_coords_any(s:str)->Tuple[float,float]:
    s2=s.replace(",","."); m=re.findall(r"[-+]?\d+(?:\.\d+)?", s2)
    if len(m)<2: raise ValueError("Coordinate non riconosciute")
    return float(m[0]), float(m[1])

def exp_decay_weights(n:int, half=7.0)->List[float]:
    lam=math.log(2.0)/half
    return [math.exp(-lam*d) for d in range(n)]

# ---------- terreno ----------
async def open_elev_grid(lat,lon, step_m=30.0)->List[List[float]]:
    dlat=step_m/111320.0
    dlon=step_m/(111320.0*math.cos(math.radians(lat)))
    coords=[{"latitude":lat+dr*dlat,"longitude":lon+dc*dlon}
            for dr in(-1,0,1) for dc in(-1,0,1)]
    url="https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=20) as c:
        r=await c.post(url,json={"locations":coords}); r.raise_for_status()
        els=[p["elevation"] for p in r.json()["results"]]
        return [els[0:3],els[3:6],els[6:9]]

def slope_aspect(grid:List[List[float]], cell=30.0)->Tuple[float,float]:
    z=grid
    dzdx=((z[0][2]+2*z[1][2]+z[2][2])-(z[0][0]+2*z[1][0]+z[2][0]))/(8*cell)
    dzdy=((z[2][0]+2*z[2][1]+z[2][2])-(z[0][0]+2*z[0][1]+z[0][2]))/(8*cell)
    slope=math.degrees(math.atan(math.hypot(dzdx,dzdy)))
    aspect=(math.degrees(math.atan2(dzdx,dzdy))+360.0)%360.0
    return slope,aspect

# ---------- geocoding / bosco ----------
async def geocode(q:str)->Dict[str,Any]:
    url="https://nominatim.openstreetmap.org/search"
    params={"format":"json","q":q,"limit":1,"addressdetails":1}
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent":"TrovaPorcini/4.3"}) as c:
        r=await c.get(url,params=params); r.raise_for_status()
        arr=r.json(); if not arr: raise HTTPException(404,"Località non trovata")
        j=arr[0]; return {"lat":float(j["lat"]), "lon":float(j["lon"]), "name":j.get("display_name",q)}

async def overpass_forest(lat,lon,radius=800)->Optional[str]:
    q=f"""[out:json][timeout:25];
      (way(around:{radius},{lat},{lon})[natural=wood];
       relation(around:{radius},{lat},{lon})[natural=wood];); out tags 20;"""
    url="https://overpass-api.de/api/interpreter"
    async with httpx.AsyncClient(timeout=25) as c:
        r=await c.post(url,data={"data":q}); r.raise_for_status()
        labels=[]
        for el in r.json().get("elements",[]):
            t=el.get("tags",{}); lt=(t.get("leaf_type") or "").lower()
            if "broad" in lt or lt=="broadleaved": labels.append("broadleaved")
            elif "conifer" in lt or lt=="needleleaved": labels.append("coniferous")
            elif "wood" in t:
                if t["wood"] in ("deciduous","broadleaved"): labels.append("broadleaved")
                if t["wood"] in ("coniferous","needleleaved"): labels.append("coniferous")
        if labels: return "broadleaved" if labels.count("broadleaved")>=labels.count("coniferous") else "coniferous"
    return None

def forest_label(kind:Optional[str], alt:float)->str:
    if kind=="coniferous": return "Pinus/Abies/Picea"
    if kind=="broadleaved":
        if alt>900: return "Fagus sylvatica"
        if alt>500: return "Castanea sativa"
        return "Quercus spp."
    if alt>1400: return "Pinus/Abies/Picea"
    if alt>900: return "Fagus sylvatica"
    if alt>500: return "Castanea sativa"
    return "Quercus spp."

# ---------- meteo: Open-Meteo (storico+fisica) ----------
async def om_daily(lat,lon)->Dict[str,Any]:
    base="https://api.open-meteo.com/v1/forecast"
    params={
        "latitude":lat,"longitude":lon,
        "daily":",".join(["precipitation_sum","temperature_2m_mean",
                          "et0_fao_evapotranspiration","shortwave_radiation_sum",
                          "relative_humidity_2m_mean"]),
        "past_days":14,"forecast_days":10,"timezone":"auto"
    }
    async with httpx.AsyncClient(timeout=25) as c:
        r=await c.get(base,params=params); r.raise_for_status()
        return r.json()["daily"]

# ---------- meteo: OpenWeather (previsioni) ----------
async def ow_daily(lat,lon,key)->Optional[Dict[str,Any]]:
    if not key: return None
    url="https://api.openweathermap.org/data/3.0/onecall"
    par={"lat":lat,"lon":lon,"units":"metric","exclude":"minutely,alerts","appid":key}
    async with httpx.AsyncClient(timeout=25) as c:
        r=await c.get(url,params=par)
        if r.status_code!=200: return None
        j=r.json()
        return {
            "daily": j.get("daily",[]),      # 7-8 giorni
            "hourly": j.get("hourly",[])     # 48h
        }

# ---------- modello ----------
def thermal_window(T7, month, lat)->float:
    if month in (9,10,11): lo,hi=9.0,14.0
    else: lo,hi=14.0,19.0
    if T7<=lo-3 or T7>=hi+3: return 0.0
    if T7<=lo:  return (T7-(lo-3))/3*0.6
    if T7>=hi:  return (hi+3-T7)/3*0.6
    return 0.6+0.4*((T7-lo)/(hi-lo))

def alt_score(alt,lat): 
    opt=900.0+(lat-45.0)*25.0
    return clamp(1.0-abs(alt-opt)/800.0,0,1)

def asp_slope_bonus(aspect,slope,hot):
    b=0.0
    if hot:
        if deg_to_octant(aspect) in ("N","NE","NW"): b+=0.15
        if deg_to_octant(aspect) in ("S","SE","SW"): b-=0.10
    b+= clamp((slope-5)/15.0,0,0.10)
    return b

def hum_score(rh):
    if rh is None: return 0.5
    if rh>=85: return 1.0
    if rh<=55: return 0.0
    return (rh-55)/30.0

def rad_penalty(sw):
    if sw is None: return 0.0
    return -clamp((sw-18.0)/10.0,0,0.25)

def moist_eff(r14:List[float], et0_7:float)->float:
    w=exp_decay_weights(len(r14),7.0)
    eff=sum(r14[i]*w[i] for i in range(len(r14)))
    eff-=0.6*et0_7*7.0
    return max(0.0, eff)

def build_index_series(lat,lon,alt,slope,aspect,dates,precip,tmean,et0,sw,rh)->Dict[str,Any]:
    today=13  # 0..13 (passati), 14..23 (futuri)
    month=datetime.now(timezone.utc).astimezone().month
    res_idx=[]; caps=[]; whys=[]; tips=[]; rains=[]
    hot = (sum(tmean[max(0,today-6):today+1])/max(1,len(tmean[max(0,today-6):today+1])))>=18.0
    for d in range(10):
        k=today+d
        r14=[precip[k-i] if k-i>=0 else 0.0 for i in range(14)]
        et0_7=sum(x for x in et0[max(0,k-6):k+1] if x is not None)/max(1,len([x for x in et0[max(0,k-6):k+1] if x is not None])) if et0 else 0.0
        eff=moist_eff(r14, et0_7)
        T7=sum(tmean[max(0,k-6):k+1])/max(1,len(tmean[max(0,k-6):k+1]))
        win=thermal_window(T7, month, lat)
        altc=alt_score(alt,lat)
        aspb=asp_slope_bonus(aspect,slope,hot)
        rh7=sum(x for x in rh[max(0,k-6):k+1] if x is not None)/max(1,len([x for x in rh[max(0,k-6):k+1] if x is not None])) if rh else None
        hum=hum_score(rh7)
        sw7=sum(x for x in sw[max(0,k-6):k+1] if x is not None)/max(1,len([x for x in sw[max(0,k-6):k+1] if x is not None])) if sw else None
        rad=rad_penalty(sw7)
        moist=clamp((eff-20.0)/40.0,0,1)
        idx01=0.35*moist + 0.25*win + 0.15*altc + 0.10*clamp(0.5+aspb,0,1) + 0.10*hum + 0.05*clamp(1.0+rad,0,1)
        idx=int(round(100*clamp(idx01,0,1))); res_idx.append(idx)
        # cappelli/3h
        lo,hi = (0,1) if idx<30 else (1,3) if idx<50 else (3,6) if idx<65 else (6,12) if idx<80 else (12,25)
        caps.append((lo,hi))
        whys.append(f"P14_eff≈{eff:.1f} mm; T7={T7:.1f}°C; quota={int(alt)} m; pend={slope:.0f}° asp={deg_to_octant(aspect)}; RH7≈{rh7:.0f}%" if rh7 is not None else
                    f"P14_eff≈{eff:.1f} mm; T7={T7:.1f}°C; quota={int(alt)} m; pend={slope:.0f}° asp={deg_to_octant(aspect)}")
        tip=[]
        if idx>=70: tip.append("Giornata buona: muoviti presto; cerca faggi/castagneti con lettiera umida.")
        elif idx>=50: tip.append("Possibile ma non ovunque: versanti N-NE-NW e suolo che trattiene umidità.")
        else: tip.append("Bassa probabilità: impluvi/fossi ombrosi; attendi piogge.")
        if hot: tip.append("Con caldo: evita esposizioni S; alza leggermente la quota.")
        if slope<5: tip.append("Pendenza lieve: prova 5–15° per drenaggio migliore.")
        tips.append(" ".join(tip))
        rains.append({"p14_mm": round(sum(precip[max(0,k-13):k+1]),1),
                      "next10_total_mm": round(sum(precip[k+1:k+11]),1)})
    return {"idx":res_idx,"caps":caps,"why":whys,"tips":tips,"rain":rains}

# ---------- endpoint ----------
@app.get("/api/geocode")
async def api_geocode(q:str): return await geocode(q)

@app.get("/api/score")
async def api_score(
    q: Optional[str] = Query(None),
    coords: Optional[str] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
):
    # risoluzione coordinate
    if coords:
        try: lat,lon = parse_coords_any(coords)
        except Exception: raise HTTPException(400,"Coordinate non riconosciute")
    elif lat is None or lon is None:
        if not q: raise HTTPException(400,"Inserisci località o coordinate")
        g=await geocode(q); lat,lon=g["lat"],g["lon"]

    # terreno
    try:
        grid=await open_elev_grid(lat,lon); elev=grid[1][1]; slope,aspect=slope_aspect(grid)
    except Exception:
        elev,slope,aspect=700.0,8.0,0.0
    try: kind=await overpass_forest(lat,lon)
    except Exception: kind=None
    forest=forest_label(kind, elev)

    # meteo: OM (storico+fisica) + OW (forecast) fuso
    om = await om_daily(lat,lon)
    ow = await ow_daily(lat,lon, os.getenv("OPENWEATHER_API_KEY",""))

    # serie base OM (24 gg)
    dates=om["time"]; precip=om["precipitation_sum"]; tmean=om["temperature_2m_mean"]
    et0=om.get("et0_fao_evapotranspiration",[None]*len(precip))
    sw =om.get("shortwave_radiation_sum",[None]*len(precip))
    rh =om.get("relative_humidity_2m_mean",[None]*len(precip))

    # sostituisci i 10 futuri con OW quando disponibile (più “vicino al suolo”)
    if ow and ow.get("daily"):
        for i in range(10):
            if i < len(ow["daily"]):
                d = ow["daily"][i]
                # OpenWeather: mm di pioggia nel campo "rain" (può mancare)
                pr = float(d.get("rain",0.0))
                tm = (d["temp"]["day"] + d["temp"]["night"])/2.0
                j = 14 + i  # indice futuro
                if j < len(precip): precip[j]=pr
                if j < len(tmean):  tmean[j]=tm

    comp = build_index_series(lat,lon,elev,slope,aspect,dates,precip,tmean,et0,sw,rh)

    # diagnostica piogge
    last_rain_days = None
    for i in range(13,-1,-1):
        if precip[i] >= 1.0: last_rain_days = 13 - i; break
    next_rain_in, next_rain_mm = None, 0.0
    for d in range(1,11):
        val = precip[13+d] if 13+d < len(precip) else 0.0
        if val >= 1.0 and next_rain_in is None: next_rain_in = d
        next_rain_mm += val

    return {
        "coords":{"lat":lat,"lon":lon},
        "terrain":{"elev_m":float(elev),"slope_deg":round(slope,1),"aspect_deg":round(aspect,1),
                   "aspect_oct":deg_to_octant(aspect),"forest":forest},
        "series":{"dates":dates,"precip":precip,"tmean":tmean},
        "daily":comp,
        "meta":{"p14_mm":round(sum(precip[0:14]),1),
                "tmean7_c": round(sum(tmean[7:14])/7.0,1),
                "last_rain_days": last_rain_days,
                "next_rain_in_days": next_rain_in,
                "next_rain_10d_mm": round(next_rain_mm,1),
                "source": "open-meteo + openweather"}
    }




