from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
import math, asyncio, os

import httpx

from utils import slope_aspect_from_elev_grid, deg_to_octant, make_grid_bbox

APP_NAME = "TrovaPorcini-v0.8 (+https://example.org)"
HEADERS = {"User-Agent": APP_NAME}

app = FastAPI(title="TrovaPorcini API v0.8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- servizi remoti di base -------------------

async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format":"json","q":q,"addressdetails":"1","limit":1}
    async with httpx.AsyncClient(timeout=15, headers={**HEADERS,"Accept-Language":"it"}) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    if not j: 
        raise httpx.HTTPError("Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0]["display_name"]}

async def open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join(["precipitation_sum","temperature_2m_mean","temperature_2m_min","temperature_2m_max","et0_fao_evapotranspiration"]),
        "hourly": "relative_humidity_2m,precipitation",
        "past_days": 14, "forecast_days": 10, "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0):
    """griglia 3x3 (circa 30m) attorno al punto, per pendenza/aspect + quota media"""
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m*deg_per_m_lat
    dlon = step_m*deg_per_m_lon
    coords = []
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            coords.append({"latitude": lat + dr*dlat, "longitude": lon + dc*dlon})
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        r = await client.post(url, json={"locations": coords})
        r.raise_for_status()
        j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]

async def overpass_forest(lat: float, lon: float, radius_m: int = 800) -> Optional[str]:
    """Inferisce broadleaved/coniferous se vicino a tag boschivi (OSM)."""
    query = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})[natural=wood];
      relation(around:{radius_m},{lat},{lon})[natural=wood];
      way(around:{radius_m},{lat},{lon})[landuse=forest];
      relation(around:{radius_m},{lat},{lon})[landuse=forest];
    );
    out tags;
    """
    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as client:
        r = await client.post("https://overpass-api.de/api/interpreter", data={"data":query})
        r.raise_for_status()
        j = r.json()
    labels=[]
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        if "leaf_type" in tags:
            lt = tags["leaf_type"].lower()
            if "broad" in lt or lt=="broadleaved": labels.append("broadleaved")
            elif "conifer" in lt or lt=="coniferous": labels.append("coniferous")
        elif "wood" in tags:
            if tags["wood"] in ("deciduous","broadleaved"): labels.append("broadleaved")
            elif tags["wood"] in ("coniferous","needleleaved"): labels.append("coniferous")
    if labels:
        return "broadleaved" if labels.count("broadleaved")>=labels.count("coniferous") else "coniferous"
    return None

def forest_label_from_osm_kind(kind: Optional[str], alt_m: float) -> str:
    if kind == "coniferous": return "Pinus/Abies/Picea"
    if kind == "broadleaved":
        if alt_m > 800: return "Fagus sylvatica"
        if 500 < alt_m <= 800: return "Castanea sativa"
        return "Quercus spp."
    # fallback quota
    if alt_m > 1400: return "Pinus/Abies/Picea"
    if alt_m > 900: return "Fagus sylvatica"
    if alt_m > 500: return "Castanea sativa"
    return "Quercus spp."

# ------------------- componenti indice -------------------

def api_star_series(past14: List[float], fut10: List[float], half: float = 8.0) -> Tuple[float, List[int]]:
    """API* (emivita=half) + previsione futura con kernel biologico (picco ~9 gg)."""
    lam = 2.0**(-1.0/half)
    # passato: somma ultimi 14 giorni
    P14 = sum(past14)
    # stato API* oggi (rolling)
    api_star = 0.0
    for p in past14[-14:]:
        api_star = lam*api_star + p
    # kernel ritardo (gaussiana centrata a 9 giorni)
    mu, sigma = 9.0, 2.0
    K = [math.exp(-0.5*((d-mu)/sigma)**2) for d in range(0, 11)]
    # normalizza contributo futuro in modo che sommi ~alla pioggia totale prevista
    ksum = sum(K) if sum(K)>0 else 1.0
    K = [k/ksum for k in K]
    # scala leggermente (0.9) per prudenza
    K = [0.9*k for k in K]

    scores = []
    s = api_star
    for d in range(11):
        # aggiornamento giornaliero: decadimento + pioggia giornaliera pesata
        fut_contrib = sum(fut10[i]*K[d] if i==d else 0.0 for i in range(len(fut10)))
        s = lam*s + (past14[d] if d<0 else 0.0) + fut_contrib
        # mappatura semplificata 0–100
        # 0 => 0 mm; 100 => ~40 mm API*
        score_api = max(0.0, min(100.0, (s/40.0)*100.0))
        scores.append(int(round(score_api)))
    return P14, scores

def thermal_score(tmean7: float, tmin7: float, tmax7: float, lat: float) -> float:
    # comfort autunnale leggermente modulato dalla latitudine
    # Nord (lat>44) preferisce 14–18; Sud (lat<41) 16–20
    base = 17.0 + (41.0 - lat)*0.15
    ideal = base
    span = 5.0
    s = max(0.0, 100.0*(1.0 - abs(tmean7-ideal)/span))
    # penalità per tmin molto bassa o tmax troppo alta
    if tmin7 < 4.0: s *= 0.7
    if tmax7 > 26.0: s *= 0.7
    return s

def topo_modifiers(elev_m: float, slope_deg: float, aspect_oct: str, lat: float) -> float:
    # bonus lieve vicino al range classico 700–1200 m in Appennino/Alpi
    bonus = 0.0
    if 700 <= elev_m <= 1200: bonus += 8.0
    # esposizione: se pendenza > 1.5°, Nord aumenta in estate, Sud in tard’autunno
    if slope_deg > 1.5:
        if lat >= 43.5:  # nord/centro
            if aspect_oct in ("NE","N","NW"): bonus += 3.0
        else:  # sud
            if aspect_oct in ("SE","S","SW"): bonus += 3.0
    return bonus

# ------------------- API principali -------------------

@app.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode(q)

@app.get("/api/score")
async def api_score(lat: float, lon: float, half: float = 8.0):
    # parallelo: meteo + quota/griglia + forest
    om_task = asyncio.create_task(open_meteo(lat, lon))
    grid_task = asyncio.create_task(open_elevation_grid(lat, lon))
    forest_kind = await overpass_forest(lat, lon)
    om, grid = await asyncio.gather(om_task, grid_task)

    elev_m = float(grid[1][1])
    slope_deg, aspect_deg, aspect_oct = slope_aspect_from_elev_grid(grid, cell_size_m=30.0)

    daily = om["daily"]
    precip = daily["precipitation_sum"]        # 14 passati + 10 futuri
    tempm  = daily["temperature_2m_mean"]
    tmin   = daily["temperature_2m_min"]
    tmax   = daily["temperature_2m_max"]
    et0    = daily.get("et0_fao_evapotranspiration", [0]*len(tempm))

    past14 = precip[:14]
    fut10  = precip[14:24]
    P14, api_scores = api_star_series(past14, fut10, half=half)

    pastT = tempm[:14]
    last7T = pastT[-7:] if len(pastT)>=7 else pastT
    Tmean7 = sum(last7T)/max(1, len(last7T))
    Tmin7  = sum(tmin[:7])/max(1, len(tmin[:7]))
    Tmax7  = sum(tmax[:7])/max(1, len(tmax[:7]))
    ET0_7d = sum(et0[:7]) if et0 else 0.0

    # componente termica
    therm = thermal_score(Tmean7, Tmin7, Tmax7, lat)
    # idrico: api* oggi (api_scores[0]) corretto con ET0
    hyd = max(0.0, api_scores[0] - min(30.0, ET0_7d*2.0))
    # topografia
    bonus_topo = topo_modifiers(elev_m, slope_deg, aspect_oct, lat)

    # combinazione pesata
    score_today = max(0, min(100, int(round(0.45*hyd + 0.35*therm + 0.20*(50+bonus_topo)))))

    # proiezione futuro (D+0..D+10)
    scores_next = []
    for d in range(11):
        hyd_d = max(0.0, api_scores[d] - min(30.0, ET0_7d*2.0))  # ET0 costante su breve
        therm_d = therm  # MVP: manteniamo termica media 7g
        sc = max(0, min(100, int(round(0.45*hyd_d + 0.35*therm_d + 0.20*(50+bonus_topo)))))
        scores_next.append(sc)

    # migliore finestra 3-giorni
    best_s, best_e, best_m = 0, 0, -1
    for i in range(0, 9):
        m = sum(scores_next[i:i+3])/3.0
        if m > best_m:
            best_m, best_s, best_e = m, i, i+2

    # spiegazioni
    reasons = []
    if P14 < 8: reasons.append("Piogge scarse nelle 2 settimane (P14 < 8 mm).")
    else: reasons.append(f"Piogge utili nelle 2 settimane: P14 ≈ {round(P14,1)} mm.")
    if 14 <= Tmean7 <= 20: reasons.append("Termica favorevole (Tmed 7g 14–20 °C).")
    else: reasons.append(f"Termica sub-ottimale (Tmed 7g {round(Tmean7,1)} °C).")
    if 700 <= elev_m <= 1200: reasons.append("Quota in fascia classica 700–1200 m.")
    if slope_deg > 1.5: reasons.append(f"Esposizione {aspect_oct} su versante pendente.")

    forecast_line = None
    if best_m >= 60:
        forecast_line = f"Possibile finestra D+{best_s}→D+{best_e} (media {int(round(best_m))})."
        reasons.append("Finestra guidata da piogge previste e bilancio idrico crescente.")

    # umidità/24h da OWM (facoltativo, solo per arricchire la scheda)
    humidity_now = None
    rain24h = None
    try:
        owk = os.getenv("OPENWEATHER_API_KEY")
        if owk:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("https://api.openweathermap.org/data/2.5/weather",
                                params={"lat":lat,"lon":lon,"appid":owk,"units":"metric"})
                j = r.json()
            humidity_now = int(j["main"]["humidity"])
            # stima pioggia prossime 24h da forecast 3h
            r = await httpx.AsyncClient(timeout=12).get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"lat":lat,"lon":lon,"appid":owk,"units":"metric"}
            )
            j = r.json()
            rain24h = 0.0
            for k, it in enumerate(j["list"][:8]):  # 8*3h=24h
                rain24h += float(it.get("rain", {}).get("3h", 0.0))
    except Exception:
        pass

    return {
        "elevation_m": elev_m,
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "aspect_octant": deg_to_octant(aspect_deg),
        "forest": forest_label_from_osm_kind(forest_kind, elev_m),

        "P7_mm": sum(past14[-7:]),
        "P14_mm": round(P14,1),
        "API_star_mm": round(api_scores[0]*0.4,1),  # scala inversa della mappatura 40mm->100
        "ET0_7d_mm": round(ET0_7d,1),

        "Tmean7_c": round(Tmean7,1),
        "Tmin7_c": round(Tmin7,1),
        "Tmax7_c": round(Tmax7,1),

        "humidity_now": humidity_now,
        "rain24h_forecast_mm": rain24h,

        "score_today": score_today,
        "scores_next11": scores_next,
        "best_window": {"start": best_s, "end": best_e, "mean": int(round(best_m))},

        "explanation": {
            "today": " • ".join(reasons[:2]),
            "reasons": reasons,
            "forecast": forecast_line
        },

        "confidence": 0.75,  # MVP: fisso; alzare con concordanza modelli
        "map_hint": {"lat": lat, "lon": lon, "radius_km": 60}
    }

@app.get("/api/score_grid")
async def api_score_grid(
    day: int = Query(0, ge=0, le=10),
    bbox: str = Query(..., description="minLat,minLon,maxLat,maxLon"),
    step_km: float = Query(10.0, ge=3.0, le=25.0)
):
    """Ritorna una griglia (GeoJSON) di punteggi per la mappa heat del giorno D."""
    minLat, minLon, maxLat, maxLon = map(float, bbox.split(","))
    pts = make_grid_bbox(minLat, minLon, maxLat, maxLon, step_km)
    feats = []

    async def one(p):
        d = await api_score(p["lat"], p["lon"])
        return {
            "type":"Feature",
            "geometry":{"type":"Point","coordinates":[p["lon"], p["lat"]]},
            "properties":{"score": d["scores_next11"][day]}
        }

    # throttle semplice
    out = []
    for i in range(0, len(pts), 10):
        chunk = pts[i:i+10]
        res = await asyncio.gather(*[one(p) for p in chunk])
        out.extend(res)
    return {"type":"FeatureCollection","features": out}

# root
@app.get("/")
def root():
    return {"ok": True, "service": "TrovaPorcini v0.8"}
