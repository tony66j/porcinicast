# main.py  (TrovaPorcini v4.2)
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, math, asyncio, re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

app = FastAPI(title="TrovaPorcini API v4.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------- utilità base ---------
def parse_float(s: str) -> float:
    return float(s.replace(",", ".").strip())

def parse_coords(s: str) -> Tuple[float, float]:
    """
    Accetta:
      - "41.41948, 14.47275"
      - "41,41948, 14,47275" (virgole decimali italiane)
      - "41.41948 14.47275"
    """
    s2 = s.replace(",", ".")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s2)
    if len(nums) < 2:
        raise ValueError("Coordinate non riconosciute")
    return float(nums[0]), float(nums[1])

def clamp(x, a, b): return max(a, min(b, x))

def exp_decay_weights(n: int, half_life_days: float = 7.0) -> List[float]:
    # peso per giorno “i giorni fa”: w = 0.5 ** (age/half_life)
    lam = math.log(2.0) / half_life_days
    return [math.exp(-lam * d) for d in range(n)]

def deg_to_octant(angle_deg: float) -> str:
    d = (angle_deg + 22.5) % 360
    idx = int(d // 45)
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][idx]

# --------- terreno: quota / slope / aspect ---------
async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    """griglia 3x3 per slope/aspect; usa open-elevation (batch)"""
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)))
    dlat = step_m * deg_per_m_lat
    dlon = step_m * deg_per_m_lon
    coords = [{"latitude": lat + dr * dlat, "longitude": lon + dc * dlon}
              for dr in (-1,0,1) for dc in (-1,0,1)]
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json={"locations": coords})
        r.raise_for_status()
        els = [p["elevation"] for p in r.json()["results"]]
        grid = [els[0:3], els[3:6], els[6:9]]
        return grid

def slope_aspect_from_grid(grid: List[List[float]], cell_m: float = 30.0) -> Tuple[float, float]:
    """Horn 3x3 filter -> pendenza (deg), aspect (deg da N, CW)."""
    z = grid
    dzdx = ((z[0][2] + 2*z[1][2] + z[2][2]) - (z[0][0] + 2*z[1][0] + z[2][0])) / (8*cell_m)
    dzdy = ((z[2][0] + 2*z[2][1] + z[2][2]) - (z[0][0] + 2*z[0][1] + z[0][2])) / (8*cell_m)
    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)
    aspect_rad = math.atan2(dzdx, dzdy)  # attenzione: x su est, y su nord
    aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
    return slope_deg, aspect_deg

# --------- geocoding / bosco ---------
async def geocode_place(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "q": q, "limit": 1, "addressdetails": 1}
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "TrovaPorcini/4.2"}) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        arr = r.json()
        if not arr: raise HTTPException(status_code=404, detail="Località non trovata")
        j = arr[0]
        return {"lat": float(j["lat"]), "lon": float(j["lon"]), "name": j.get("display_name", q)}

async def overpass_forest(lat: float, lon: float, radius_m: int=800) -> Optional[str]:
    """stima 'broadleaved' vs 'coniferous' da OSM tag nel raggio"""
    q = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})[natural=wood];
      relation(around:{radius_m},{lat},{lon})[natural=wood];
    ); out tags 20;
    """
    url = "https://overpass-api.de/api/interpreter"
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(url, data={"data": q})
        r.raise_for_status()
        j = r.json()
        labels = []
        for el in j.get("elements", []):
            tags = el.get("tags", {})
            lt = (tags.get("leaf_type") or "").lower()
            if "broad" in lt or lt == "broadleaved": labels.append("broadleaved")
            elif "conifer" in lt or lt == "needleleaved": labels.append("coniferous")
            elif "wood" in tags:
                if tags["wood"] in ("deciduous","broadleaved"): labels.append("broadleaved")
                if tags["wood"] in ("coniferous","needleleaved"): labels.append("coniferous")
        if labels:
            return "broadleaved" if labels.count("broadleaved") >= labels.count("coniferous") else "coniferous"
    return None

def forest_label_from_kind(kind: Optional[str], alt_m: float) -> str:
    if kind == "coniferous": return "Pinus/Abies/Picea"
    if kind == "broadleaved":
        if alt_m > 900: return "Fagus sylvatica"
        if 500 < alt_m <= 900: return "Castanea sativa"
        return "Quercus spp."
    # fallback per altitudine
    if alt_m > 1400: return "Pinus/Abies/Picea"
    if alt_m > 900: return "Fagus sylvatica"
    if alt_m > 500: return "Castanea sativa"
    return "Quercus spp."

# --------- meteo (Open-Meteo) ---------
async def open_meteo_daily(lat: float, lon: float) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join([
            "precipitation_sum",
            "temperature_2m_mean",
            "et0_fao_evapotranspiration",
            "shortwave_radiation_sum",
            "relative_humidity_2m_mean"
        ]),
        "past_days": 14, "forecast_days": 10,
        "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        return r.json()["daily"]

# --------- modello v4.2 ---------
def thermal_window_score(T7: float, month: int, lat: float) -> float:
    # 9–14°C in autunno, 14–19°C in alta estate montana, transizione morbida
    if month in (9,10,11):  # autunno
        lo, hi = 9.0, 14.0
    else:
        lo, hi = 14.0, 19.0
    if T7 <= lo-3: return 0.0
    if T7 >= hi+3: return 0.0
    if T7 <= lo:  return (T7-(lo-3))/3.0*0.6  # segue morbido
    if T7 >= hi:  return (hi+3-T7)/3.0*0.6
    return 0.6 + 0.4*((T7-lo)/(hi-lo))

def alt_score(alt_m: float, lat: float) -> float:
    # optimum ~ 900 m a 45°N;  +25m/° verso Nord, -25m/° verso Sud
    opt = 900.0 + (lat-45.0)*25.0
    return clamp(1.0 - abs(alt_m - opt)/800.0, 0.0, 1.0)

def aspect_slope_bonus(aspect_deg: float, slope_deg: float, hot: bool) -> float:
    # bonus lieve per N-NE-NW con caldo; penalità S in caldo; 5–20° di pendenza aiuta
    octa = deg_to_octant(aspect_deg)
    b = 0.0
    if hot:
        if octa in ("N","NE","NW"): b += 0.15
        if octa in ("S","SE","SW"): b -= 0.10
    b += clamp((slope_deg-5)/15.0, 0, 0.10)  # da 5° a 20° fino +0.10
    return b

def humidity_score(rh7: Optional[float]) -> float:
    if rh7 is None: return 0.5
    if rh7 >= 85: return 1.0
    if rh7 <= 55: return 0.0
    return (rh7-55)/30.0

def radiation_penalty(sw7: Optional[float]) -> float:
    if sw7 is None: return 0.0
    # ~ troppo irradiamento estivo se > 20 MJ/m2/d
    return -clamp((sw7-18.0)/10.0, 0.0, 0.25)

def moisture_effective(r14: List[float], et0_7: float) -> float:
    # somma decrescente (half-life 7d) sugli ultimi 14 gg; penalità ET0
    w = exp_decay_weights(14, 7.0)
    eff = sum(r14[i]*w[i] for i in range(14))
    eff -= 0.6 * et0_7 * 7.0         # evapotraspirazione “mangia” acqua utile
    return max(0.0, eff)

def index_components(
    lat: float, lon: float, alt_m: float, slope_deg: float, aspect_deg: float,
    dates: List[str], precip: List[float], tmean: List[float],
    et0: List[Optional[float]], sw: List[Optional[float]], rh: List[Optional[float]],
) -> Dict[str, Any]:
    # serie: 14 passati + 10 futuri (24 totali)
    today_idx = 14-1  # ultimo dei passati è “oggi”
    res_idx: List[float] = []
    caps: List[Tuple[int,int]] = []
    why: List[str] = []
    tips: List[str] = []
    rain_info: List[Dict[str,Any]] = []

    # pre-compute utili
    month = datetime.now(timezone.utc).astimezone().month
    lat_abs_hot = (sum(t for t in tmean[max(0,today_idx-6):today_idx+1]) / 7.0) if today_idx >= 6 else tmean[today_idx]
    is_hot = (lat_abs_hot is not None) and (lat_abs_hot >= 18.0)

    for d in range(10):  # D+0..D+9
        k = today_idx + d
        # pioggia utile: decadimento che include anche giorni futuri
        r14 = []
        for back in range(14):
            pos = k - back
            if pos < 0: r14.append(0.0)
            else: r14.append(precip[pos])
        eff = moisture_effective(r14, sum(x for x in et0[max(0,k-6):k+1] if x is not None)/max(1, len([x for x in et0[max(0,k-6):k+1] if x is not None])) if et0 else 0.0)

        T7 = sum(t for t in tmean[max(0,k-6):k+1]) / max(1, len(tmean[max(0,k-6):k+1]))
        win = thermal_window_score(T7, month, lat)
        altc = alt_score(alt_m, lat)
        aspb = aspect_slope_bonus(aspect_deg, slope_deg, hot=is_hot)
        hum  = humidity_score(sum(x for x in rh[max(0,k-6):k+1] if x is not None)/max(1,len([x for x in rh[max(0,k-6):k+1] if x is not None])) if rh else None)
        radp = radiation_penalty(sum(x for x in sw[max(0,k-6):k+1] if x is not None)/max(1,len([x for x in sw[max(0,k-6):k+1] if x is not None])) if sw else None)

        # normalizzazione “moisture”: 20–60 mm effettivi = 0–1 (oltre 60 satura)
        moist = clamp((eff - 20.0) / 40.0, 0.0, 1.0)

        # pesi (robusti)
        idx01 = (
            0.35*moist +
            0.25*win   +
            0.15*altc  +
            0.10*clamp(0.5+aspb, 0.0, 1.0) +
            0.10*hum   +
            0.05*clamp(1.0+radp, 0.0, 1.0)
        )
        idx = round(100.0 * clamp(idx01, 0.0, 1.0))
        res_idx.append(idx)

        # stima cappelli/3h (range)
        # base dalla scala indice; fattore compatibilità bosco e incertezza minima
        if idx < 30: lo, hi = 0, 1
        elif idx < 50: lo, hi = 1, 3
        elif idx < 65: lo, hi = 3, 6
        elif idx < 80: lo, hi = 6, 12
        else: lo, hi = 12, 25
        caps.append((lo, hi))

        # spiegazione sintetica
        why.append(
            f"P14_eff≈{eff:.1f} mm; T7={T7:.1f}°C; quota={int(alt_m)} m; "
            f"pend={slope_deg:.0f}° asp={deg_to_octant(aspect_deg)}; "
            f"RH7≈{(sum(x for x in rh[max(0,k-6):k+1] if x is not None)/max(1,len([x for x in rh[max(0,k-6):k+1] if x is not None]))):.0f}% "
            if rh and any(r is not None for r in rh) else
            f"P14_eff≈{eff:.1f} mm; T7={T7:.1f}°C; quota={int(alt_m)} m; pend={slope_deg:.0f}° asp={deg_to_octant(aspect_deg)}"
        )

        # consigli pratici dinamici
        tip = []
        if idx >= 70:
            tip.append("Giornata buona: muoviti presto; cerca faggi/castagneti con lettiera umida.")
        elif idx >= 50:
            tip.append("Possibile ma non ovunque: preferisci versanti N-NE-NW e suolo che trattiene umidità.")
        else:
            tip.append("Bassa probabilità: esplora zone fresche, fossi, impluvi; attendi piogge.")
        if is_hot: tip.append("Con caldo: evita esposizioni S; ombra e quota leggermente maggiore aiutano.")
        if slope_deg < 5: tip.append("Pend. lieve: prova aree 5–15° per drenaggio migliore.")
        tips.append(" ".join(tip))

        # piogge utili (storico vs prossimi)
        rain_info.append({
            "p14_mm": round(sum(precip[max(0,k-13):k+1]), 1),
            "next10_total_mm": round(sum(precip[k+1:k+11]), 1),
        })

    return {
        "idx": res_idx,
        "caps": caps,
        "why": why,
        "tips": tips,
        "rain": rain_info
    }

# --------- endpoint ---------
@app.get("/api/geocode")
async def api_geocode(q: str):
    g = await geocode_place(q)
    return g

@app.get("/api/score")
async def api_score(
    q: Optional[str] = Query(None, description="Località (opzionale)"),
    coords: Optional[str] = Query(None, description="lat,lon (opzionale)"),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None)
):
    # 1) risolvi coordinate
    if coords:
        try:
            lat_p, lon_p = parse_coords(coords)
            lat, lon = lat_p, lon_p
        except Exception:
            raise HTTPException(status_code=400, detail="Coordinate non riconosciute")
    elif lat is None or lon is None:
        if q:
            g = await geocode_place(q)
            lat, lon = g["lat"], g["lon"]
        else:
            raise HTTPException(status_code=400, detail="Inserisci località o coordinate")

    # 2) terreno
    try:
        grid = await open_elevation_grid(lat, lon)
        elev_m = float(grid[1][1])
        slope_deg, aspect_deg = slope_aspect_from_grid(grid)
    except Exception:
        elev_m, slope_deg, aspect_deg = 700.0, 8.0, 0.0  # fallback robusto

    try:
        kind = await overpass_forest(lat, lon)
    except Exception:
        kind = None
    forest_label = forest_label_from_kind(kind, elev_m)

    # 3) meteo
    daily = await open_meteo_daily(lat, lon)
    dates = daily["time"]  # ISO
    precip = daily["precipitation_sum"]
    tmean  = daily["temperature_2m_mean"]
    et0    = daily.get("et0_fao_evapotranspiration", [None]*len(precip))
    sw     = daily.get("shortwave_radiation_sum",  [None]*len(precip))
    rh     = daily.get("relative_humidity_2m_mean", [None]*len(precip))

    # 4) indice e serie
    comp = index_components(lat, lon, elev_m, slope_deg, aspect_deg, dates, precip, tmean, et0, sw, rh)

    # ultime/prossime piogge (per “considerazioni”)
    last_rain_days = None
    for i in range(13, -1, -1):
        if precip[i] >= 1.0:
            last_rain_days = 13 - i
            break
    next_rain_in = None
    next_rain_mm = 0.0
    for d in range(1, 11):
        if precip[13+d] >= 1.0 and next_rain_in is None:
            next_rain_in = d
        next_rain_mm += precip[13+d]

    return {
        "coords": {"lat": lat, "lon": lon},
        "terrain": {
            "elev_m": elev_m,
            "slope_deg": round(slope_deg,1),
            "aspect_deg": round(aspect_deg,1),
            "aspect_oct": deg_to_octant(aspect_deg),
            "forest": forest_label
        },
        "series": {
            "dates": dates,
            "precip": precip,
            "tmean": tmean
        },
        "daily": comp,  # idx, caps, tips, why, rain
        "meta": {
            "p14_mm": round(sum(precip[0:14]), 1),
            "tmean7_c": round(sum(tmean[7:14])/7.0, 1),
            "last_rain_days": last_rain_days,
            "next_rain_in_days": next_rain_in,
            "next_rain_10d_mm": round(next_rain_mm, 1),
            "source": "open-meteo"
        }
    }



