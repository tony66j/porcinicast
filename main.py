from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import math, asyncio
import httpx

APP_NAME = "PorciniCast-API/0.9.1"
HEADERS  = {"User-Agent": APP_NAME}

app = FastAPI(title="PorciniCast API v0.9.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# -----------------------
# Helper scientifici
# -----------------------

def deg_to_octant(deg: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW","N"]
    i = int((deg % 360) / 45.0 + 0.5)
    return dirs[i]

def slope_aspect_from_elev_grid(grid: List[List[float]], cell_m: float = 30.0) -> Tuple[float,float,float]:
    """
    grid 3x3 di quota (m). Ritorna: elev_m (centro), slope_deg, aspect_deg (0=N).
    """
    z = grid
    # Filtri robusti
    if len(z) != 3 or any(len(r) != 3 for r in z):
        return float("nan"), 0.0, 0.0
    elev_c = float(z[1][1])
    # Horn's method
    dzdx = ((z[0][2] + 2*z[1][2] + z[2][2]) - (z[0][0] + 2*z[1][0] + z[2][0])) / (8*cell_m)
    dzdy = ((z[2][0] + 2*z[2][1] + z[2][2]) - (z[0][0] + 2*z[0][1] + z[0][2])) / (8*cell_m)
    slope_rad = math.atan(math.sqrt(dzdx*dzdx + dzdy*dzdy))
    slope_deg = math.degrees(slope_rad)
    aspect_rad = math.atan2(dzdx, dzdy)   # 0 = N
    aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
    return elev_c, slope_deg, aspect_deg

def best_window_3day(scores: List[int]) -> Tuple[int,int,int]:
    """massimo media mobile 3gg, ritorna start, end, mean"""
    if not scores:
        return 0, 0, 0
    best_mean, best_i = -1, 0
    for i in range(0, len(scores)-2):
        m = (scores[i]+scores[i+1]+scores[i+2]) / 3
        if m > best_mean:
            best_mean, best_i = m, i
    return best_i, best_i+2, round(best_mean)

def composite_score(P14: float, Tmean7: float, elev_m: float, aspect_oct: str,
                    forest_label: str, month: int, extras: Dict[str,float]) -> Tuple[float, Dict[str, float]]:
    """
    Indice 0–100. Molto semplificato ma trasparente. pese tarate “ragionevoli”.
    """
    # Pioggia: 15–70 mm/14gg ottimale (campana)
    p_opt = 42.5
    p_spread = 25
    p_raw = math.exp(-((P14 - p_opt)**2)/(2*p_spread**2))  # 0..1

    # Temperatura media settimanale: 12–18°C target
    t_opt = 15.0
    t_spread = 6.0
    t_raw = math.exp(-((Tmean7 - t_opt)**2)/(2*t_spread**2))

    # Quota: finestra larga per latitudine italiana
    alt_opt = 1100.0 if month in (8,9) else 900.0   # estate-inizio autunno un po’ più in alto
    a_raw = math.exp(-((elev_m - alt_opt)**2)/(2*(450.0**2)))

    # Esposizione: Nord / NE / NW leggermente premiate in estate
    asp_bonus = 1.0
    if month in (7,8,9):
        if aspect_oct in ("N","NE","NW"): asp_bonus = 1.1
        if aspect_oct in ("S","SE","SW"): asp_bonus = 0.9

    # Bosco: bonus castagno/faggio > quercia > conifere (indicativo)
    compat = 1.0
    k = forest_label.lower()
    if "castanea" in k or "fagus" in k: compat = 1.15
    elif "quercus" in k:                compat = 1.05
    elif "pinus" in k or "abies" in k:  compat = 0.9

    # Pesi
    score = (0.40*p_raw + 0.35*t_raw + 0.20*a_raw) * asp_bonus * compat
    score = max(0.0, min(1.0, score)) * 100.0

    breakdown = {
        "p14n": round(p_raw, 3),
        "tn":   round(t_raw, 3),
        "alt_n": round(a_raw, 3),
        "asp":  asp_bonus,
        "compat": compat
    }
    return score, breakdown

# -----------------------
# Chiamate esterne robuste
# -----------------------

async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    if not j:
        raise HTTPException(status_code=404, detail="Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0].get("display_name","")}

async def open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join(["precipitation_sum","temperature_2m_mean"]),
        "past_days": 14, "forecast_days": 10, "timezone":"auto"
    }
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    # 3x3 usando Open-Elevation
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m*deg_per_m_lat
    dlon = step_m*deg_per_m_lon
    coords = [{"latitude": lat+dr*dlat, "longitude": lon+dc*dlon}
              for dr in (-1,0,1) for dc in (-1,0,1)]
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        r = await client.post(url, json={"locations": coords})
        r.raise_for_status()
        j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]

async def overpass_forest(lat: float, lon: float, radius_m: int = 800) -> Optional[str]:
    """
    Tenta di inferire broadleaved/coniferous. In caso di errore -> None (fallback su quota).
    """
    q = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})["natural"="wood"];
      relation(around:{radius_m},{lat},{lon})["natural"="wood"];
    );
    out tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    try:
        async with httpx.AsyncClient(timeout=25, headers=HEADERS) as client:
            r = await client.post(url, data={"data": q})
            r.raise_for_status()
            j = r.json()
    except Exception:
        return None

    labels = []
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        lt = tags.get("leaf_type","").lower()
        if "broad" in lt or lt == "broadleaved": labels.append("broadleaved")
        elif "conifer" in lt or lt == "coniferous": labels.append("coniferous")
        elif "wood" in tags:
            if tags["wood"] in ("deciduous","broadleaved"): labels.append("broadleaved")
            if tags["wood"] in ("coniferous","needleleaved"): labels.append("coniferous")
    if labels:
        return "broadleaved" if labels.count("broadleaved") >= labels.count("coniferous") else "coniferous"
    return None

def forest_label_from_osm_kind(kind: Optional[str], alt_m: float) -> str:
    if kind == "coniferous": return "Pinus/Abies/Picea"
    if kind == "broadleaved":
        if alt_m > 800:    return "Fagus sylvatica"
        if 500 < alt_m <= 800: return "Castanea sativa"
        return "Quercus spp."
    # Fallback su quota
    if alt_m > 1400: return "Pinus/Abies/Picea"
    if alt_m > 900:  return "Fagus sylvatica"
    if alt_m > 500:  return "Castanea sativa"
    return "Quercus spp."

# -----------------------
# API
# -----------------------

@app.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode(q)

@app.get("/api/score")
async def api_score(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    d: int = Query(0, ge=0, le=10)  # giorno futuro 0..10
):
    """
    Ritorna indice oggi + serie prossimi 10 gg, more info e 'reason' esplicativa.
    """
    # Meteo + quota/aspetto + bosco (robusto a errori esterni)
    try:
        geodata = await open_meteo(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo error: {type(e).__name__}")

    try:
        elev_grid = await open_elevation_grid(lat, lon)
        elev_m, slope_deg, aspect_deg = slope_aspect_from_elev_grid(elev_grid, cell_m=30.0)
    except Exception:
        elev_m, slope_deg, aspect_deg = float("nan"), 0.0, 0.0

    try:
        kind = await overpass_forest(lat, lon)  # può tornare None
    except Exception:
        kind = None
    forest_label = forest_label_from_osm_kind(kind, elev_m)

    daily = geodata.get("daily", {})
    precip = daily.get("precipitation_sum", [])
    tempm  = daily.get("temperature_2m_mean", [])
    if not precip or not tempm:
        raise HTTPException(status_code=502, detail="Meteo incompleto")

    past14 = precip[:14]
    P14    = sum(p or 0.0 for p in past14)
    pastT  = tempm[:10]  # usa ultimi 10 giorni disponibili
    last7T = pastT[-7:] if len(pastT) >= 7 else pastT
    Tmean7 = sum(t or 0.0 for t in last7T) / max(1, len(last7T))

    # serie proiezione prossimi 10 giorni (aggiorno P14 con sliding window + previsione)
    futP = precip[14:24] if len(precip) >= 24 else [0.0]*10
    futT = tempm[10:20]  if len(tempm)  >= 20 else [Tmean7]*10

    scores = []
    rolling_P14 = P14
    now = datetime.now(timezone.utc)
    month = now.month
    aspect_oct = deg_to_octant(aspect_deg)
    reasons = []

    # score oggi
    score_today, breakdown = composite_score(P14, Tmean7, elev_m, aspect_oct, forest_label, month, {})
    # proiezioni
    for i in range(10):
        rolling_P14 = max(0.0, rolling_P14 + (futP[i] or 0.0) - (past14[i] if i < len(past14) else 0.0))
        sc, _ = composite_score(rolling_P14, futT[i], elev_m, aspect_oct, forest_label, month, {})
        scores.append(int(round(sc)))

    s,e,m = best_window_3day(scores)

    # reason: sintetica e leggibile
    def cat(x):
        if x >= 75: return "molto alta"
        if x >= 55: return "buona"
        if x >= 35: return "moderata"
        return "bassa"

    if P14 < 15: reasons.append("piogge scarse negli ultimi 14 giorni")
    elif P14 > 80: reasons.append("piogge molto abbondanti (possibile dilavamento)")
    else: reasons.append("piogge recenti nella fascia ottimale")

    if   Tmean7 < 10: reasons.append("temperatura fresca per i porcini")
    elif Tmean7 > 20: reasons.append("temperatura un po' alta")
    else:             reasons.append("temperatura favorevole")

    reasons.append(f"quota {int(round(elev_m))} m (finestra stagionale)")
    reasons.append(f"bosco indicativo: {forest_label}")
    reasons.append(f"esposizione {aspect_oct}")

    out = {
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
        "breakdown": breakdown,
        "reason": "; ".join(reasons)
    }

    # se d>0 ritorno anche uno "score_on_day"
    if d > 0:
        out["score_on_day"] = scores[d-1]
    return out


