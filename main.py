from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, math, asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

from utils import slope_aspect_from_elev_grid, best_window_3day, deg_to_octant

APP_NAME = "TrovaPorcini/0.7 (+https://netlify.app)"
HEADERS = {"User-Agent": APP_NAME}

OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")  # optional

app = FastAPI(title="TrovaPorcini API (v0.7)", version="0.7.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Providers ----------
async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "q": q, "addressdetails": 1, "limit": 1}
    async with httpx.AsyncClient(timeout=15, headers={**HEADERS, "Accept-Language": "it"}) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    if not j:
        raise httpx.HTTPError("Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0].get("display_name","")}

async def open_meteo_daily(lat: float, lon: float, past_days:int=30, forecast_days:int=10) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join([
            "precipitation_sum","temperature_2m_mean",
            "temperature_2m_min","temperature_2m_max",
            "et0_fao_evapotranspiration"
        ]),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
        r = await c.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def open_meteo_hourly(lat: float, lon: float, hours:int=72) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "precipitation_probability",
        "forecast_days": 3,
        "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
        r = await c.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m*deg_per_m_lat
    dlon = step_m*deg_per_m_lon
    coords = [{"latitude": lat + dr*dlat, "longitude": lon + dc*dlon}
              for dr in (-1,0,1) for dc in (-1,0,1)]
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": coords})
        r.raise_for_status()
        j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]

async def overpass_forest(lat: float, lon: float, radius_m:int=800) -> Optional[str]:
    q = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})[natural=wood];
      relation(around:{radius_m},{lat},{lon})[natural=wood];
      way(around:{radius_m},{lat},{lon})[landuse=forest];
      relation(around:{radius_m},{lat},{lon})[landuse=forest];
    );
    out tags;
    """
    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as c:
        r = await c.get("https://overpass-api.de/api/interpreter", params={"data": q})
        r.raise_for_status()
        j = r.json()
    labels = []
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        if "leaf_type" in tags:
            lt = tags["leaf_type"].lower()
            if "broad" in lt or lt == "broadleaved": labels.append("broadleaved")
            elif "conifer" in lt or lt == "coniferous": labels.append("coniferous")
        elif "wood" in tags:
            if tags["wood"] in ("deciduous","broadleaved"): labels.append("broadleaved")
            elif tags["wood"] in ("coniferous","needleleaved"): labels.append("coniferous")
    if labels:
        return "broadleaved" if labels.count("broadleaved")>=labels.count("coniferous") else "coniferous"
    return None

# OpenWeather (opzionale)
async def owm_current(lat:float, lon:float)->Optional[Dict[str,Any]]:
    if not OWM_KEY: return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat":lat,"lon":lon,"units":"metric","lang":"it","appid":OWM_KEY}
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def owm_forecast_5d(lat:float, lon:float)->Optional[Dict[str,Any]]:
    if not OWM_KEY: return None
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat":lat,"lon":lon,"units":"metric","lang":"it","appid":OWM_KEY}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()

# ---------- Scoring helpers v0.7 ----------
def k_from_half_life(days: float)->float:
    return 0.5**(1.0/days)

def api_decay(precip: List[float], half_life_days: float=8.0)->float:
    k = k_from_half_life(half_life_days)
    api = 0.0
    for p in precip:
        api = k*api + (p or 0.0)
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

def lat_factor(lat: float, elev_m: float) -> float:
    band = 0.0
    if lat >= 45.0: band = +0.06
    elif lat <= 41.0: band = -0.06
    else: band = 0.0
    atten = max(0.0, 1.0 - min(1.0, elev_m/1400.0))
    return 1.0 + band*atten

def aspect_modifier(aspect_deg: float) -> float:
    octants = {'N':1.08,'NE':1.06,'E':1.0,'SE':0.92,'S':0.85,'SW':0.85,'W':0.92,'NW':1.03}
    return octants.get(deg_to_octant(aspect_deg), 1.0)

def slope_modifier(slope_deg: float)->float:
    if slope_deg < 1.5: return 1.0
    if slope_deg < 12: return 1.05
    if slope_deg < 25: return 1.0
    return 0.9

def api_to_moisture(api: float)->float:
    if api <= 5: return 0.05*api
    if api <= 20: return 0.25 + 0.35*((api-5)/15.0)
    if api <= 70: return 0.60 + 0.35*((api-20)/50.0)
    if api <= 120: return 0.95 - 0.25*((api-70)/50.0)
    return 0.65

def final_score(lat:float, api_val: float, et0_7: float, t_suit: float, elev_m: float, aspect_deg: float, slope_deg: float) -> Tuple[float, Dict[str,float]]:
    moisture = max(0.0, api_to_moisture(max(0.0, api_val - 0.6*et0_7)))
    t_suit_adj = max(0.0, min(1.0, t_suit * lat_factor(lat, elev_m)))
    elev_mod = 1.05 if 700<=elev_m<=1400 else (0.6 if (elev_m<150 or elev_m>2200) else 0.95)
    asp_m = 1.0 if slope_deg < 1.5 else aspect_modifier(aspect_deg)
    slp_m = slope_modifier(slope_deg)
    base = 0.6*moisture + 0.4*t_suit_adj
    score = max(0.0, min(100.0, 100.0*base*elev_mod*asp_m*slp_m))
    parts = {"moisture":moisture,"temp":t_suit_adj,"elev_mod":elev_mod,"aspect_mod":asp_m,"slope_mod":slp_m}
    return score, parts

def classify_elev(elev_m: float)->str:
    if elev_m < 500: return "bassa"
    if elev_m < 1000: return "media"
    if elev_m < 1600: return "montana"
    return "alta"

def explanation(sc:int, P7:float, P14:float, api_star:float, et0_7:float, Tmean7:float, Tmin7:float, Tmax7:float, humid:Optional[int], last_rain_days:int)->Tuple[str,List[str]]:
    reasons = []
    if sc >= 70:
        msg = "Consigliato: umidità del suolo e temperatura sono nel range ottimale."
    elif sc >= 60:
        msg = "Moderatamente consigliato: condizioni buone ma non perfette."
    elif sc >= 50:
        msg = "Incerto: alcuni fattori sono favorevoli, altri limitanti."
    else:
        msg = "Sconsigliato oggi: condizioni non favorevoli alla fruttificazione."
    if P14 < 20: reasons.append(f"Piogge scarse negli ultimi 14 giorni ({P14:.0f} mm).")
    if api_star < 20: reasons.append("Bilancio idrico basso (terreno tendenzialmente secco).")
    if et0_7 > 30: reasons.append(f"Evapotraspirazione elevata 7g ({et0_7:.0f} mm) → disseccamento.")
    if last_rain_days > 10: reasons.append(f"Ultima pioggia utile oltre {last_rain_days} giorni fa.")
    if Tmean7 < 10: reasons.append(f"Temperature medie basse (Tmed7 {Tmean7:.1f} °C).")
    if Tmean7 > 18: reasons.append(f"Temperature medie alte (Tmed7 {Tmean7:.1f} °C).")
    if Tmax7 > 26: reasons.append(f"Massime elevate negli ultimi giorni (Tmax7 {Tmax7:.1f} °C).")
    if humid is not None and humid < 45: reasons.append(f"Umidità relativa bassa ora ({humid}%).")
    if not reasons:
        reasons.append("Piogge recenti adeguate e termica favorevole per i porcini.")
    return msg, reasons[:4]

def habitat_tips(aspect_oct:str, slope_deg:float, forest_label:str, elev_m:float, dry:bool)->List[str]:
    tips = []
    if slope_deg < 1.5:
        tips.append("Esposizione locale variabile: muoviti tra margini e piccole conche per cercare umidità.")
    elif dry:
        tips.append("Cerca versanti N–NE e conche ombreggiate per maggiore umidità.")
    else:
        tips.append("Controlla margini tra bosco e radure, specialmente dopo piogge.")
    if slope_deg < 2:
        tips.append("Evita ristagni: preferisci leggere pendenze per drenaggio.")
    elif 5 <= slope_deg <= 15:
        tips.append("Pendenze 5–15° spesso ideali per drenaggio e aerazione del suolo.")
    if "Fagus" in forest_label or "faggio" in forest_label.lower():
        tips.append("Nel faggio batte le zone fresche e ricche di lettiera, evita creste ventose.")
    elif "Castanea" in forest_label or "Quercus" in forest_label:
        tips.append("In castagno/quercia prova i margini e i viottoli ombreggiati dopo piogge regolari.")
    elif "Conifere" in forest_label:
        tips.append("In conifere cerca sotto abeti/pini adulti, evitando suoli aghifogliati troppo asciutti.")
    band = classify_elev(elev_m)
    if band == "bassa":
        tips.append("A bassa quota privilegia esposizioni fresche; con caldo prolungato sali di quota.")
    elif band == "alta":
        tips.append("A quota alta la stagione è più tardiva: osserva finestre più avanti.")
    return tips[:4]

def other_species(forest_label:str, month:int, wet:bool)->List[str]:
    out = []
    broad = ("Castanea" in forest_label) or ("Quercus" in forest_label)
    fagus = ("Fagus" in forest_label)
    conif = ("Conifere" in forest_label)
    if broad:
        out += ["Cantharellus cibarius (gallinaccio)"]
        if 6 <= month <= 9: out += ["Amanita caesarea (ovulo buono) — raccogli solo se riconosci con certezza"]
        out += ["Macrolepiota procera (mazza di tamburo)"]
    if fagus:
        out += ["Hydnum repandum (dente di cane)", "Russula cyanoxantha (colombina)"]
    if conif:
        out += ["Lactarius deliciosus (sanguinello)", "Suillus luteus (pineti)"]
    if wet:
        out += ["Craterellus tubaeformis (finferlo trombetta)"]
    seen=set(); ret=[]
    for s in out:
        if s not in seen:
            seen.add(s); ret.append(s)
    return ret[:5]

# ---------- API ----------
@app.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode(q)

@app.get("/api/score")
async def api_score(lat: float, lon: float, half: float = 8.0):
    tasks = [
        asyncio.create_task(open_meteo_daily(lat, lon, past_days=30, forecast_days=10)),
        asyncio.create_task(open_meteo_hourly(lat, lon, hours=72)),
        asyncio.create_task(open_elevation_grid(lat, lon)),
        asyncio.create_task(overpass_forest(lat, lon)),
        asyncio.create_task(owm_current(lat, lon)),
        asyncio.create_task(owm_forecast_5d(lat, lon)),
    ]
    meteo_d, meteo_h, elev_grid, forest_kind, owm_now, owm_fc = await asyncio.gather(*tasks)

    elev_m = float(elev_grid[1][1])
    slope_deg, aspect_deg, aspect_oct = slope_aspect_from_elev_grid(elev_grid, cell_size_m=30.0)

    # esposizione "variabile" se quasi piano
    if slope_deg < 1.5:
        aspect_oct_display = "Variabile"
    else:
        aspect_oct_display = aspect_oct

    daily = meteo_d["daily"]
    precip = daily["precipitation_sum"]
    tmean = daily["temperature_2m_mean"]
    tmin = daily["temperature_2m_min"]
    tmax = daily["temperature_2m_max"]
    et0  = daily.get("et0_fao_evapotranspiration", [0.0]*len(precip))

    pastN = 30
    pastP = precip[:pastN]
    pastTmin = tmin[:pastN]
    pastTmax = tmax[:pastN]
    pastTmean = tmean[:pastN]
    pastET0 = et0[:pastN]

    P7 = sum(pastP[-7:])
    P14 = sum(pastP[-14:])
    API_val = api_decay(pastP, half_life_days=half)
    ET0_7 = sum(pastET0[-7:]) if pastET0 else 0.0
    Tmean7 = sum(pastTmean[-7:])/max(1,len(pastTmean[-7:]))
    Tmin7 = min(pastTmin[-7:]) if pastTmin else 0.0
    Tmax7 = max(pastTmax[-7:]) if pastTmax else 0.0

    last_rain_days = 999
    for i, p in enumerate(reversed(pastP)):
        if (p or 0) > 1.0:
            last_rain_days = i
            break

    prob = meteo_h.get("hourly", {}).get("precipitation_probability", [])
    prob_rain_next3d = round(sum(prob)/max(1,len(prob)),1) if prob else None

    T_suit_today = temperature_suitability(pastTmin[-1], pastTmax[-1], pastTmean[-1])

    today_score, parts = final_score(lat, API_val, ET0_7, T_suit_today, elev_m, aspect_deg, slope_deg)

    futP = precip[pastN:pastN+10]
    futTmean = tmean[pastN:pastN+10]
    futTmin = tmin[pastN:pastN+10]
    futTmax = tmax[pastN:pastN+10]
    futET0 = et0[pastN:pastN+10]

    scores = []
    rolling_api = API_val
    k = k_from_half_life(half)
    for i in range(10):
        rolling_api = k*rolling_api + (futP[i] or 0.0)
        t_s = temperature_suitability(futTmin[i], futTmax[i], futTmean[i])
        sc, _ = final_score(lat, rolling_api, sum(futET0[max(0,i-6):i+1]), t_s, elev_m, aspect_deg, slope_deg)
        scores.append(int(round(sc)))

    s,e,m = best_window_3day(scores)

    humid_now = None
    rain24 = None
    if owm_now:
        try: humid_now = int(owm_now["main"]["humidity"])
        except: pass
    if owm_fc:
        try:
            lst = owm_fc.get("list",[])[:8]
            rain24 = sum([v.get("rain",{}).get("3h",0.0) for v in lst])
        except: pass

    def forest_label_from_osm_kind(kind: Optional[str], alt_m: float) -> str:
        if kind == "coniferous": return "Conifere (Pinus/Abies/Picea)"
        if kind == "broadleaved":
            if alt_m > 900: return "Fagus sylvatica (stima)"
            if 500 < alt_m <= 900: return "Castanea/Quercus (stima)"
            return "Quercus spp. (stima)"
        if alt_m > 1400: return "Conifere (Pinus/Abies/Picea)"
        if alt_m > 900: return "Fagus sylvatica (stima)"
        if alt_m > 500: return "Castanea sativa (stima)"
        return "Quercus spp. (stima)"
    forest_label = forest_label_from_osm_kind(forest_kind, elev_m)

    msg, why = explanation(int(round(today_score)), P7, P14, API_val, ET0_7, Tmean7, Tmin7, Tmax7, humid_now, last_rain_days)

    dry = (API_val < 20) or (humid_now is not None and humid_now < 45)
    tips = habitat_tips(aspect_oct_display, slope_deg, forest_label, elev_m, dry)
    wet = (API_val > 25)
    now = datetime.now(timezone.utc)
    species = other_species(forest_label, month=now.astimezone().month, wet=wet)

    agree = 0
    agree += 1 if (today_score>=60 and P14>=25) else 0
    agree += 1 if (today_score<50 and API_val<20) else 0
    agree += 1 if (60<=today_score<=80 and 10<=Tmean7<=18) else 0
    confidence = max(0.25, min(0.95, 0.5 + 0.15*(agree) - 0.2*(1-parts["moisture"])))

    return {
        "elevation_m": elev_m,
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct_display,
        "forest": forest_label,
        "API_star_mm": round(API_val,1),
        "P7_mm": round(P7,1),
        "P14_mm": round(P14,1),
        "ET0_7d_mm": round(ET0_7,1),
        "last_rain_days": int(last_rain_days if last_rain_days<900 else -1),
        "prob_rain_next3d": prob_rain_next3d,
        "Tmean7_c": round(Tmean7,1),
        "Tmin7_c": round(Tmin7,1),
        "Tmax7_c": round(Tmax7,1),
        "humidity_now": humid_now,
        "rain24h_forecast_mm": round(rain24,1) if rain24 is not None else None,
        "score_today": int(round(today_score)),
        "scores_next10": scores,
        "best_window": {"start": s, "end": e, "mean": m},
        "breakdown": parts,
        "advice": msg,
        "explanation": {"today": msg, "reasons": why},
        "habitat_tips": tips,
        "other_species": species,
        "confidence": round(confidence,2)
    }
# placeholder - main.py will be replaced by previously prepared v0.7 content if needed
