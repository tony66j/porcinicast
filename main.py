# v0.9 – PorciniCast backend (FastAPI)
# Dipendenze: fastapi, uvicorn, httpx, numpy (opzionale), python-dotenv (non obbl.)
# Avvio su Render:  uvicorn main:app --host 0.0.0.0 --port $PORT

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import math, asyncio, os
import httpx

APP_NAME = "PorciniCast-v0.9 (+https://example.org)"
HEADERS = {"User-Agent": APP_NAME}

app = FastAPI(title="PorciniCast API v0.9")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------------- UTIL -----------------------

def deg_to_octant(angle_deg: float) -> str:
    # 8 settori classici
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    i = int((angle_deg % 360) / 45.0 + 0.5) % 8
    return dirs[i]

def slope_aspect_from_elev_grid(grid: List[List[float]], cell_size_m: float = 30.0) -> Tuple[float, float, float]:
    """
    grid 3x3 elevations (m). Ritorna (elev_m, slope_deg, aspect_deg)
    """
    if len(grid) != 3 or any(len(r) != 3 for r in grid):
        return 0.0, 0.0, 0.0
    z = grid
    dzdx = ((z[0][2] + 2*z[1][2] + z[2][2]) - (z[0][0] + 2*z[1][0] + z[2][0])) / (8.0 * cell_size_m)
    dzdy = ((z[2][0] + 2*z[2][1] + z[2][2]) - (z[0][0] + 2*z[0][1] + z[0][2])) / (8.0 * cell_size_m)
    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)
    aspect_rad = math.atan2(dzdy, -dzdx)
    aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
    elev_m = float(z[1][1])
    return elev_m, slope_deg, aspect_deg

def lat_seasonal_alt_opt(lat: float, month: int) -> float:
    """
    Altitudine “ottimale” (m) per Boletus edulis ~ funzione grezza lat/mese.
    Tende a valori più bassi al Sud e in piena estate sale in quota.
    (Heuristica: da affinare con dati reali)
    """
    base = 1000 - 6.5*(lat-43)  # ~-6.5 m/°lat intorno a Italia
    summer_boost = 200 if month in (7,8,9) else (100 if month in (6,10) else 0)
    return max(200, base + summer_boost)

# ---------------------- SERVIZI ESTERNI -----------------------

async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        r = await client.get(url, params=params); r.raise_for_status()
        j = r.json()
    if not j: raise httpx.HTTPError("Località non trovata")
    return {"lat":float(j[0]["lat"]), "lon":float(j[0]["lon"]), "display": j[0]["display_name"]}

async def open_elevation_grid(lat: float, lon: float, step_m: float=30.0) -> List[List[float]]:
    # campiono 3x3 con ~30 m (griglia locale)
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m*deg_per_m_lat
    dlon = step_m*deg_per_m_lon
    coords=[]
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            coords.append({"latitude": lat+dr*dlat, "longitude": lon+dc*dlon})
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        r = await client.post("https://api.open-elevation.com/api/v1/lookup", json={"locations":coords})
        r.raise_for_status(); j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]

async def overpass_forest(lat: float, lon: float, radius_m: int=800) -> Optional[str]:
    """
    Prova a inferire broadleaved/coniferous vicino al punto. Se non c'è info, stima da quota.
    """
    q = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})["natural"="wood"];
      relation(around:{radius_m},{lat},{lon})["natural"="wood"];
      way(around:{radius_m},{lat},{lon})["landuse"="forest"];
      relation(around:{radius_m},{lat},{lon})["landuse"="forest"];
    ); out tags;
    """
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        r = await client.post("https://overpass-api.de/api/interpreter", data={"data": q})
        r.raise_for_status()
        j = r.json()
    labels=[]
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        lt = tags.get("leaf_type","").lower()
        if "broad" in lt: labels.append("broadleaved")
        elif "conif" in lt: labels.append("coniferous")
        elif "wood" in tags:
            w=tags["wood"].lower()
            if w in ("deciduous","broadleaved"): labels.append("broadleaved")
            elif w in ("coniferous","needleleaved"): labels.append("coniferous")
    if labels:
        return "broadleaved" if labels.count("broadleaved")>=labels.count("coniferous") else "coniferous"
    return None

def forest_label_from_kind(kind: Optional[str], alt_m: float) -> str:
    if kind=="coniferous": return "Pinus/Abies/Picea"
    if kind=="broadleaved":
        if alt_m>800: return "Fagus sylvatica"
        if 500<alt_m<=800: return "Castanea sativa"
        return "Quercus spp."
    # fallback alt
    if alt_m>1400: return "Pinus/Abies/Picea"
    if alt_m>900:  return "Fagus sylvatica"
    if alt_m>500:  return "Castanea sativa"
    return "Quercus spp."

async def openweather(lat: float, lon: float) -> Dict[str, Any]:
    """
    Usa OpenWeather OneCall 3.0 (serve chiave in env OPENWEATHER_API_KEY).
    Per semplicità prendiamo:
      - past 7d via 'timemachine' (limite OWM: fino a 5 gg; quando manca usiamo forecast come proxy)
      - forecast daily 10d
    NB: su piani free ci sono limiti; la logica gestisce fallback.
    """
    key = os.getenv("OPENWEATHER_API_KEY", "")
    if not key:
        raise httpx.HTTPError("OPENWEATHER_API_KEY mancante")

    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as client:
        # Forecast daily 7-10 giorni
        f_url = "https://api.openweathermap.org/data/3.0/onecall"
        fp = {"lat":lat,"lon":lon,"appid":key,"units":"metric","exclude":"minutely,hourly,alerts"}
        fr = await client.get(f_url, params=fp); fr.raise_for_status(); fj = fr.json()

    daily = fj.get("daily", [])
    # precip forecast (mm) come somma rain+snow
    fut_precip = [ (d.get("rain",0.0) or 0.0) + (d.get("snow",0.0) or 0.0) for d in daily[:10] ]
    fut_tmean  = [ (d["temp"]["min"]+d["temp"]["max"])/2.0 for d in daily[:10] ]

    # "Past days" proxy: prendiamo daily già presenti che includono 'yesterday' quando possibile
    # Se non ci sono, stimiamo 0.
    past_precip = []
    past_tmean  = []
    # Nella pratica OWM daily include day 0=oggi; usiamo gli ultimi 7 elementi "recenti" se disponibili
    for d in daily[::-1][:7]:
        past_precip.append((d.get("rain",0.0) or 0.0) + (d.get("snow",0.0) or 0.0))
        past_tmean.append((d["temp"]["min"]+d["temp"]["max"])/2.0)

    past_precip = past_precip[::-1]
    past_tmean  = past_tmean[::-1]

    # umidità e pioggia 24h prossime (stima dal primo giorno forecast)
    humidity_now = fj.get("current",{}).get("humidity")
    rain24h = daily[0].get("rain",0.0) if daily else None

    return {
        "past_precip": past_precip,     # <=7
        "past_tmean": past_tmean,       # <=7
        "fut_precip": fut_precip,       # 0..10
        "fut_tmean": fut_tmean,         # 0..10
        "humidity_now": humidity_now,
        "rain24h": rain24h
    }

# Per la mappa: centri di poligoni boschivi in bbox
async def overpass_forest_centers_bbox(minLat: float, minLon: float, maxLat: float, maxLon: float):
    query = f"""
    [out:json][timeout:25];
    (
      way["natural"="wood"]({minLat},{minLon},{maxLat},{maxLon});
      relation["natural"="wood"]({minLat},{minLon},{maxLat},{maxLon});
      way["landuse"="forest"]({minLat},{minLon},{maxLat},{maxLon});
      relation["landuse"="forest"]({minLat},{minLon},{maxLat},{maxLon});
    ); out center;
    """
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        r = await client.post("https://overpass-api.de/api/interpreter", data={"data": query})
        r.raise_for_status(); j = r.json()
    pts=[]
    for el in j.get("elements", []):
        c = el.get("center")
        if c: pts.append((float(c["lat"]), float(c["lon"])))
    return pts

# ---------------------- MODELLO/SCORING -----------------------

def composite_score(
    P14: float, Tmean7: float, elev_m: float, aspect_deg: float,
    forest_label: str, month: int, API_star: float, lat: float
) -> Tuple[int, Dict[str,float], str, List[str]]:
    """
    Combina vari fattori in 0..100 e produce spiegazione.
    """
    reasons=[]

    # pioggia ultime 2 settimane: target 20–60 mm (troppa/s poca penalizza)
    if P14<10: p14n=0.2
    elif P14>120: p14n=0.4
    else:
        # parabola blanda con max 1 tra 30 e 70
        c = max(0.0, 1 - abs((P14-50)/40))
        p14n = 0.4 + 0.6*c
    reasons.append(f"Pioggia 14g: {P14:.1f} mm.")

    # finestra termica: ideale 12–18 °C media settimanale (autunno)
    # leggero adattamento con latitudine (Nord un po' più bassa).
    t_opt = 15.0 - 0.08*(lat-43)
    t_dev = abs(Tmean7 - t_opt)
    tnorm = max(0.0, 1 - t_dev/8.0)
    reasons.append(f"Tmedia 7g: {Tmean7:.1f}°C (opt ~{t_opt:.1f}°C).")

    # bilancio idrico semplice
    # API_star già 30g con emivita ~8g: normalizziamo 0..1 intorno a 20–80 mm
    if API_star<=20: apin=0.2
    elif API_star>=120: apin=0.6
    else:
        apin = 0.3 + 0.7*max(0.0, 1-abs((API_star-60)/50))
    reasons.append(f"API* (30g): {API_star:.1f} mm.")

    # quota rispetto all'opt stagionale
    alt_opt = lat_seasonal_alt_opt(lat, month)
    a_dev = abs(elev_m - alt_opt)
    anorm = max(0.0, 1 - a_dev/700.0)
    reasons.append(f"Quota {elev_m:.0f} m (opt ~{alt_opt:.0f}).")

    # esposizione: E/SE/SW leggermente favorevoli in autunno (umidità+sole)
    asp = deg_to_octant(aspect_deg)
    asp_bonus = 1.0 if asp in ("SE","S","SW","E") else (0.9 if asp in ("W","NE") else 0.8)
    reasons.append(f"Esposizione: {asp}.")

    # compatibilità bosco
    compat = 1.1 if "Fagus" in forest_label or "Castanea" in forest_label else 1.0
    reasons.append(f"Bosco: {forest_label}.")

    # combinazione pesata
    raw = 40*p14n + 25*tnorm + 15*apin + 15*anorm
    raw *= asp_bonus * compat
    score = int(round(max(0,min(100, raw))))

    # sintesi
    if score>=75: why = "Condizioni favorevoli: piogge recenti e finestra termica adeguata."
    elif score>=60: why = "Potenziale in crescita: parametri prossimi alla finestra utile."
    elif score>=50: why = "Incerto: alcuni fattori sono ancora deboli (piogge/termica/altitudine)."
    else:          why = "Sconsigliato: bilancio idrico/termico poco favorevole."

    breakdown = {
        "p14n": p14n, "tn": tnorm, "apin": apin, "alt": anorm,
        "asp": asp_bonus, "compat": compat, "alt_opt": alt_opt
    }
    return score, breakdown, why, reasons

def best_window_3day(scores: List[int]) -> Optional[Dict[str,int]]:
    if not scores: return None
    best, idx = -1, 0
    for i in range(0, len(scores)-2):
        m = round((scores[i]+scores[i+1]+scores[i+2])/3)
        if m>best: best, idx = m, i
    return {"start": idx, "end": idx+2, "mean": best}

# ---------------------- API -----------------------

@app.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode(q)

@app.get("/api/score")
async def api_score(lat: float, lon: float):
    # parallelo: geodata + elev + forest
    m1 = asyncio.create_task(open_elevation_grid(lat, lon))
    m2 = asyncio.create_task(overpass_forest(lat, lon))
    m3 = asyncio.create_task(openweather(lat, lon))
    elev_grid, forest_kind, meteo = await asyncio.gather(m1, m2, m3)

    elev_m, slope_deg, aspect_deg = slope_aspect_from_elev_grid(elev_grid)
    forest_label = forest_label_from_kind(forest_kind, elev_m)

    # meteo
    pastP = meteo["past_precip"]
    pastT = meteo["past_tmean"]
    futP  = meteo["fut_precip"]
    futT  = meteo["fut_tmean"]

    # costruiamo serie (ultimi 14 + prossimi 10)
    past14 = pastP[-14:] if len(pastP)>=14 else ([0]*(14-len(pastP)) + pastP)
    P14 = sum(past14)
    past7 = pastT[-7:] if len(pastT)>=7 else ([pastT[-1]]*(7-len(pastT))+pastT) if pastT else [15]*7
    Tmean7 = sum(past7)/len(past7)

    # semplice bilancio con emivita (API*)
    # accumulo esponenziale su 30 gg con emivita ~8 gg: qui proxy su 14 passati + 10 futuri
    def exp_decay(seq, half=8.0):
        k = math.log(2)/half
        out = 0.0
        for i, x in enumerate(seq[::-1]):  # più recenti pesano di più
            out += x*math.exp(-k*i)
        return out
    API_star = exp_decay(past14 + futP[:10], 8.0)

    now = datetime.now(timezone.utc)
    month = now.month

    # oggi + futuro
    score_today, breakdown, why_today, reasons = composite_score(
        P14, Tmean7, elev_m, aspect_deg, forest_label, month, API_star, lat
    )
    scores = []
    rolling_P = P14
    rolling_T = past7[-1] if past7 else 15.0
    for d in range(10):
        rolling_P = max(0.0, rolling_P + (futP[d] if d < len(futP) else 0.0) - (past14[d] if d < len(past14) else 0.0))
        rolling_T = futT[d] if d < len(futT) else rolling_T
        sc, _, _, _ = composite_score(rolling_P, rolling_T, elev_m, aspect_deg, forest_label, month, API_star, lat)
        scores.append(sc)

    bw = best_window_3day(scores)
    why_forecast = "Nei prossimi giorni la finestra migliore è D+{0}→D+{1} (media {2}).".format(bw["start"], bw["end"], bw["mean"]) if bw else ""

    return {
        "elevation_m": elev_m,
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "aspect_octant": deg_to_octant(aspect_deg),
        "forest": forest_label,
        "P14_mm": round(P14,1),
        "Tmean7_c": round(Tmean7,1),
        "API_star_mm": round(API_star,1),
        "score_today": score_today,
        "scores_next11": [score_today] + scores,   # oggi + 10
        "best_window": bw,
        "humidity_now": meteo.get("humidity_now"),
        "rain24h_forecast_mm": meteo.get("rain24h"),
        "explanation": {
            "today": why_today,
            "forecast": why_forecast,
            "reasons": reasons
        },
        "confidence": 0.9  # placeholder; reale: dipende da copertura dati
    }

@app.get("/api/score_forest_points")
async def api_score_forest_points(
    bbox: str = Query(..., description="minLat,minLon,maxLat,maxLon"),
    day: int = Query(0, ge=0, le=10),
    limit: int = Query(180, ge=20, le=400),
):
    minLat, minLon, maxLat, maxLon = map(float, bbox.split(","))
    centers = await overpass_forest_centers_bbox(minLat, minLon, maxLat, maxLon)
    if not centers:
        return {"type":"FeatureCollection","features":[]}
    centers = centers[:limit]

    async def one(lat, lon):
        d = await api_score(lat, lon)
        return {
            "type":"Feature",
            "geometry":{"type":"Point","coordinates":[lon,lat]},
            "properties":{
                "score": d["scores_next11"][day],
                "forest": d["forest"]
            }
        }

    out=[]
    for i in range(0, len(centers), 12):
        res = await asyncio.gather(*[one(a,b) for (a,b) in centers[i:i+12]])
        out.extend(res)

    return {"type":"FeatureCollection","features": out}

