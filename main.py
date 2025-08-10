from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, math, asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone

# ---------------- Base utili (come la versione che funzionava) ----------------
def k_from_half_life(days: float)->float: return 0.5**(1.0/days)

def api_decay(precip: List[float], half_life_days: float=8.0)->float:
    k = k_from_half_life(half_life_days); api = 0.0
    for p in precip: api = k*api + (p or 0.0)
    return api

def temperature_suitability(tmin: float, tmax: float, tmean: float) -> float:
    if tmin < 1 or tmax > 32: return 0.0
    if tmean <= 6: base = max(0.0, (tmean-0)/6.0*0.3)
    elif tmean <= 10: base = 0.3 + 0.2*((tmean-6)/4.0)
    elif tmean <= 18: base = 0.5 + 0.5*((tmean-10)/8.0)
    elif tmean <= 22: base = 0.8 - 0.2*((tmean-18)/4.0)
    else: base = max(0.0, 0.6 - 0.6*((tmean-22)/10.0))
    if tmin < 6: base *= max(0.3, 1 - (6 - tmin)/8.0)
    if tmax > 24: base *= max(0.3, 1 - (tmax - 24)/10.0)
    return max(0.0, min(1.0, base))

def final_score(api_val: float, et0_7: float, t_suit: float) -> int:
    moisture = max(0.0, min(1.0, (api_val - 0.6*et0_7)/40.0 + 0.6))
    return int(round(max(0.0, min(100.0, 100.0*(0.6*moisture + 0.4*t_suit)))))

def best_window_3day(scores: List[int]) -> Tuple[int,int,int]:
    if len(scores)<3: return (0,0,0)
    best, m = (0,2,0), -1
    for i in range(len(scores)-2):
        s = round((scores[i]+scores[i+1]+scores[i+2])/3)
        if s>m: m=s; best=(i,i+2,m)
    return best

def deg_to_octant(deg: float)->str:
    octs = ["N","NE","E","SE","S","SW","W","NW","N"]
    return octs[int(((deg%360)+22.5)//45)]

def slope_aspect_from_elev_grid(z: List[List[float]], cell_size_m: float=30.0) -> Tuple[float,float,str]:
    dzdx = ((z[0][2]+2*z[1][2]+z[2][2]) - (z[0][0]+2*z[1][0]+z[2][0]))/(8*cell_size_m)
    dzdy = ((z[2][0]+2*z[2][1]+z[2][2]) - (z[0][0]+2*z[0][1]+z[0][2]))/(8*cell_size_m)
    slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect = (math.degrees(math.atan2(dzdx, dzdy))+360)%360
    return (round(slope,1), round(aspect,0), deg_to_octant(aspect))

# ---------------- FastAPI ----------------
app = FastAPI(title="Trovapolcini API", version="0.7.2")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
HEADERS = {"User-Agent":"Trovapolcini/0.7 (https://netlify.app)", "Accept-Language":"it"}

# ---------------- Providers ----------------
async def nominatim(q: str) -> Dict[str,Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.get(url, params=params); r.raise_for_status()
        j = r.json()
    if not j: raise HTTPException(404, "Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0].get("display_name","")}

async def open_meteo(lat:float, lon:float, past_days:int=30, forecast_days:int=10) -> Dict[str,Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":lat, "longitude":lon, "timezone":"auto",
        "daily": "precipitation_sum,temperature_2m_mean,temperature_2m_min,temperature_2m_max,et0_fao_evapotranspiration,relative_humidity_2m_mean",
        "past_days": past_days, "forecast_days": forecast_days
    }
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
        r = await c.get(url, params=params); r.raise_for_status()
        return r.json()

async def elevation_grid(lat:float, lon:float, step_m:float=30.0)->List[List[float]]:
    deg_per_m_lat = 1/111320.0
    deg_per_m_lon = 1/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m*deg_per_m_lat; dlon = step_m*deg_per_m_lon
    coords=[{"latitude":lat+dr*dlat,"longitude":lon+dc*dlon} for dr in(-1,0,1) for dc in(-1,0,1)]
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.post("https://api.open-elevation.com/api/v1/lookup", json={"locations":coords})
        r.raise_for_status(); j=r.json()
    v=[p["elevation"] for p in j["results"]]
    return [v[0:3],v[3:6],v[6:9]]

# ---------------- Endpoints ----------------
@app.get("/api/health")
async def health(): return {"ok":True,"app":"Trovapolcini","time":datetime.now(timezone.utc).isoformat()}

@app.get("/api/geocode")
async def api_geocode(q: str): return await nominatim(q)

@app.get("/api/score")
async def api_score(lat: float, lon: float, half: float=8.0):
    meteo, elev = await asyncio.gather(open_meteo(lat,lon), elevation_grid(lat,lon))
    elev_m = float(elev[1][1])
    slope_deg, aspect_deg, aspect_oct = slope_aspect_from_elev_grid(elev)

    d = meteo["daily"]; T=d["time"]; P=d["precipitation_sum"]; Tm=d["temperature_2m_mean"]
    Tmin=d["temperature_2m_min"]; Tmax=d["temperature_2m_max"]; ET0=d.get("et0_fao_evapotranspiration",[0.0]*len(P))
    RH=d.get("relative_humidity_2m_mean",[None]*len(P))

    pastN=30
    P_past=P[:pastN]; ET_past=ET0[:pastN]; Tmin_p=Tmin[:pastN]; Tmax_p=Tmax[:pastN]; Tm_p=Tm[:pastN]
    P7=sum(P_past[-7:]); P14=sum(P_past[-14:])
    API=api_decay(P_past,half_life_days=half); ET7=sum(ET_past[-7:])
    Tmin7=min(Tmin_p[-7:]) if Tmin_p else 0.0; Tmax7=max(Tmax_p[-7:]) if Tmax_p else 0.0
    Tmean7=sum(Tm_p[-7:])/max(1,len(Tm_p[-7:]))

    t_s=temperature_suitability(Tmin_p[-1],Tmax_p[-1],Tm_p[-1])
    score_today=final_score(API,ET7,t_s)

    P_future=P[pastN:pastN+10]; Tmin_f=Tmin[pastN:pastN+10]; Tmax_f=Tmax[pastN:pastN+10]; Tm_f=Tm[pastN:pastN+10]; ET_f=ET0[pastN:pastN+10]
    scores=[]; r_api=API; k=k_from_half_life(half)
    for i in range(10):
        r_api=k*r_api+(P_future[i] or 0.0)
        sc=final_score(r_api, max(0.0,sum(ET_f[max(0,i-6):i+1])), temperature_suitability(Tmin_f[i],Tmax_f[i],Tm_f[i]))
        scores.append(int(round(sc)))
    s,e,m=best_window_3day(scores)

    rain_past={str(t):round(float(mm or 0.0),1) for t,mm in zip(T[:pastN], P_past)}
    rain_future={str(t):round(float(mm or 0.0),1) for t,mm in zip(T[pastN:pastN+10], P_future)}

    def harvest(sc:int,p14:float)->str:
        if sc>=75 and p14>=20: return "6–10+ porcini"
        if sc>=55 and p14>=10: return "2–5 porcini"
        if sc>=45 and p14>=8:  return "1–3 porcini"
        return "0–1 porcini"

    reasons=[]
    if P14<20: reasons.append(f"Piogge scarse 14g: {P14:.0f} mm.")
    if ET7>30: reasons.append(f"ET0 7g elevata: {ET7:.0f} mm.")
    if Tmean7<10: reasons.append(f"Tmed7 bassa: {Tmean7:.1f} °C.")
    if Tmean7>18: reasons.append(f"Tmed7 alta: {Tmean7:.1f} °C.")
    if not reasons: reasons.append("Bilancio idrico e termica favorevoli.")

    return {
        "elevation_m": round(elev_m),
        "slope_deg": slope_deg, "aspect_deg": aspect_deg, "aspect_octant": aspect_oct,
        "API_star_mm": round(API,1), "P7_mm": round(P7,1), "P14_mm": round(P14,1),
        "ET0_7d_mm": round(ET7,1), "Tmean7_c": round(Tmean7,1), "Tmin7_c": round(Tmin7,1), "Tmax7_c": round(Tmax7,1),
        "RH7_mean_%": round(sum([v for v in RH[-7:] if v is not None])/max(1,len([v for v in RH[-7:] if v is not None]))) if any(RH[-7:]) else None,
        "score_today": score_today, "scores_next10": scores, "best_window": {"start":s,"end":e,"mean":m},
        "rain_past": rain_past, "rain_future": rain_future, "harvest_estimate": harvest(score_today, P14),
        "explanation": {"reasons": reasons},
        "tips": [
            "Preferisci versanti N–NE e conche fresche; evita creste secche.",
            "Dopo piogge >10–15 mm, la finestra migliore è spesso 2–5 giorni dopo.",
            "In castagno/quercia: bordi radure e viottoli ombreggiati; in faggio: lettiera profonda."
        ]
    }








