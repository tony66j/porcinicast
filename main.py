from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx, math, asyncio, time
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from .utils import slope_aspect_from_elev_grid, composite_score, best_window_3day, deg_to_octant

APP_NAME = "PorciniCast-MVP/0.3 (+https://example.org)"
HEADERS = {"User-Agent": APP_NAME}

app = FastAPI(title="PorciniCast MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=15, headers={**HEADERS, "Accept-Language":"it"}) as client:
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
        "daily": ",".join(["precipitation_sum","temperature_2m_mean"]),
        "past_days": 14, "forecast_days": 10, "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    """
    Campiona una griglia 3x3 attorno al punto usando Open-Elevation (chiamate multiple).
    Convertiamo ~30m di passo in gradi (approssimazione: 1 deg lat ~ 111320 m).
    """
    deg_per_m_lat = 1.0/111320.0
    # Longitude degrees depend on latitude
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
        grid = [els[0:3], els[3:6], els[6:9]]
        return grid

async def overpass_forest(lat: float, lon: float, radius_m: int = 800) -> Optional[str]:
    """
    Cerca nelle vicinanze tag boschivi e prova a inferire broadleaved/coniferous.
    """
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
    url = "https://overpass-api.de/api/interpreter"
    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as client:
        r = await client.post(url, data={"data": query})
        r.raise_for_status()
        j = r.json()
        labels = []
        for el in j.get("elements", []):
            tags = el.get("tags", {})
            if "leaf_type" in tags:
                lt = tags["leaf_type"].lower()
                if "broad" in lt or lt == "broadleaved":
                    labels.append("broadleaved")
                elif "conifer" in lt or lt == "coniferous":
                    labels.append("coniferous")
            elif "wood" in tags:
                if tags["wood"] in ("deciduous","broadleaved"):
                    labels.append("broadleaved")
                elif tags["wood"] in ("coniferous","needleleaved"):
                    labels.append("coniferous")
        if labels:
            # maggioranza
            b = labels.count("broadleaved")
            c = labels.count("coniferous")
            return "broadleaved" if b>=c else "coniferous"
        return None

def forest_label_from_osm_kind(kind: Optional[str], alt_m: float) -> str:
    if kind == "coniferous":
        return "Pinus/Abies/Picea"
    if kind == "broadleaved":
        # in quota medio-alta può comunque essere faggio
        if alt_m > 800: return "Fagus sylvatica"
        # media collina in Italia: quercia/castagno
        if 500 < alt_m <= 800: return "Castanea sativa"
        return "Quercus spp."
    # fallback per altitudine
    if alt_m > 1400: return "Pinus/Abies/Picea"
    if alt_m > 900: return "Fagus sylvatica"
    if alt_m > 500: return "Castanea sativa"
    return "Quercus spp."

@app.get("/api/geocode")
async def api_geocode(q: str):
    g = await geocode(q)
    return g

@app.get("/api/forest")
async def api_forest(lat: float, lon: float, alt: Optional[float] = None):
    kind = await overpass_forest(lat, lon)
    # stima alt se non fornita
    if alt is None:
        grid = await open_elevation_grid(lat, lon)
        alt = float(grid[1][1])
    label = forest_label_from_osm_kind(kind, alt)
    species_map = {
        'Fagus sylvatica': ['Boletus edulis s.l.', 'B. reticulatus (tarda estate in basso)'],
        'Quercus spp.': ['B. reticulatus (estivo)', 'B. edulis (autunno)'],
        'Castanea sativa': ['B. edulis s.l. (fine estate–autunno)'],
        'Pinus/Abies/Picea': ['B. pinophilus (montano)', 'B. edulis']
    }
    return {"forest": label, "species": species_map.get(label, [])}

@app.get("/api/score")
async def api_score(lat: float, lon: float):
    # parallelizza meteo + terreno + forest
    mtask = asyncio.create_task(open_meteo(lat, lon))
    gridtask = asyncio.create_task(open_elevation_grid(lat, lon))
    geodata, elev_grid = await asyncio.gather(mtask, gridtask)
    elev_m = float(elev_grid[1][1])
    slope_deg, aspect_deg, aspect_oct = slope_aspect_from_elev_grid(elev_grid, cell_size_m=30.0)

    # forest (usa alt come aiuto)
    kind = await overpass_forest(lat, lon)
    forest_label = forest_label_from_osm_kind(kind, elev_m)

    # meteo
    daily = geodata["daily"]
    precip = daily["precipitation_sum"]
    tempm  = daily["temperature_2m_mean"]
    # Separiamo passato/futuro (ultimi 14 passati, 10 futuri)
    past14 = precip[:len(precip)-10]
    P14 = sum([p or 0.0 for p in past14])
    pastT = tempm[:len(tempm)-10]
    last7T = pastT[-7:] if len(pastT) >= 7 else pastT
    Tmean7 = sum([t or 0.0 for t in last7T]) / max(1, len(last7T))

    now = datetime.now(timezone.utc)
    month = int(now.astimezone().month)

    score_today, breakdown = composite_score(P14, Tmean7, elev_m, aspect_oct, forest_label, month, lat)

    # forecast D+0..D+9: aggiorna T con previsione giornaliera; pioggia aggiorna P14 naive (sliding grossolano)
    futP = precip[-10:]
    futT = tempm[-10:]
    scores = []
    rolling_P14 = P14
    for i in range(10):
        rolling_P14 = max(0.0, rolling_P14 + (futP[i] or 0.0) - (past14[i] if i < len(past14) else 0.0))
        sc, _ = composite_score(rolling_P14, futT[i], elev_m, aspect_oct, forest_label, month, lat)
        scores.append(int(round(sc)))

    s,e,m = best_window_3day(scores)

    return {
        "elevation_m": elev_m,
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct,
        "forest": forest_label,
        "P14_mm": round(P14,1),
        "Tmean7_c": round(Tmean7,1),
        "score_today": int(round(score_today)),
        "scores_next10": scores,
        "best_window": {"start": s, "end": e, "mean": m},
        "breakdown": breakdown
    }
