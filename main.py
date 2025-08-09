# main.py  — TrovaPorcini PRO PLUS
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx, math, asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta

APP_NAME = "TrovaPorcini-ProPlus/1.0 (+https://example.org)"
HEADERS = {"User-Agent": APP_NAME}

app = FastAPI(title="TrovaPorcini PRO PLUS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------ utilità di base ------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def parse_coords(s: str) -> Optional[Tuple[float, float]]:
    """
    Accetta formati Google Maps con virgola decimale italiana:
    "41,4170010, 14,4707168" oppure "41.4170010,14.4707168"
    """
    if not s:
        return None
    s = s.strip()
    # normalizza virgole decimali
    parts = [p.strip().replace(",", ".") for p in s.split(",")]
    if len(parts) < 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
    except Exception:
        return None
    return None

# ------------------ chiamate esterne ------------------

async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "q": q, "addressdetails": 1, "limit": 1}
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        js = r.json()
        if not js:
            raise HTTPException(status_code=404, detail="Località non trovata")
        top = js[0]
        return {"lat": float(top["lat"]), "lon": float(top["lon"]), "display": top.get("display_name", q)}

async def open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    """
    Daily: 14 passati + 10 futuri; ET0, Tmean, precip. Open-Meteo fa il backfill storico.
    """
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join([
            "precipitation_sum",
            "temperature_2m_mean",
            "et0_fao_evapotranspiration",
            "rain_sum",
        ]),
        "past_days": 14,
        "forecast_days": 10,
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def elev_grid(lat: float, lon: float) -> List[List[float]]:
    """
    Griglia 3x3 da Open-Elevation (multiple lookup) per slope/aspect.
    """
    step_m = 30.0
    # conversione approssimata gradi -> metri
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)))
    dlat = step_m * deg_per_m_lat
    dlon = step_m * deg_per_m_lon
    coords = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            coords.append({"latitude": lat + dr * dlat, "longitude": lon + dc * dlon})
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        r = await client.post(url, json={"locations": coords})
        r.raise_for_status()
        js = r.json()["results"]
        # griglia come 3 righe da 3
        h = [p["elevation"] for p in js]
        return [h[0:3], h[3:6], h[6:9]]

def slope_aspect(grid: List[List[float]], cell_m: float = 30.0) -> Tuple[float, float]:
    """
    Horn’s method 3x3 per slope (°) e aspect (° da Nord; 0=N, 90=E)
    """
    z = grid
    dzdx = ((z[0][2] + 2*z[1][2] + z[2][2]) - (z[0][0] + 2*z[1][0] + z[2][0])) / (8*cell_m)
    dzdy = ((z[2][0] + 2*z[2][1] + z[2][2]) - (z[0][0] + 2*z[0][1] + z[0][2])) / (8*cell_m)
    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)
    aspect_rad = math.atan2(dzdx, -dzdy)  # rotato per avere 0=N
    aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
    return slope_deg, aspect_deg

# ------------------ modello previsione ------------------

def half_life_decay(days: List[float], amounts: List[float], hl: float = 7.0) -> float:
    """
    Somma piogge con decadimento esponenziale (emivita = hl).
    'days' 0..14 (0 oggi), 'amounts' mm
    """
    if not days or not amounts:
        return 0.0
    lam = math.log(2.0) / hl
    tot = 0.0
    for d, mm in zip(days, amounts):
        if mm is None:
            continue
        if mm <= 0:
            continue
        # filtro: contano >1 mm
        w = math.exp(-lam * d)
        tot += mm * w
    return tot

def thermal_window_score(t7: float, month: int, lat: float) -> float:
    """
    Finestra termica dinamica: banda ottimale dipende da stagione / latitudine.
    Output 0..1.
    """
    # base autunnale
    lo, hi = 9.0, 14.0
    # estate montana / lat alto
    if month in (6,7,8) or lat > 45.5:
        lo, hi = 14.0, 19.0
    # primavere fresche
    if month in (4,5):
        lo, hi = 10.0, 16.0
    if t7 <= lo - 2:
        return 0.0
    if t7 >= hi + 2:
        return 0.0
    if t7 < lo:
        return (t7 - (lo - 2)) / 2.0 * 0.5
    if t7 > hi:
        return (1.0 - (t7 - hi) / 2.0) * 0.5
    return 1.0

def et0_penalty(et0_7: float) -> float:
    """Più ET₀ alto => suolo che si asciuga → penalità morbida."""
    if et0_7 <= 10.0:
        return 1.0
    if et0_7 >= 30.0:
        return 0.6
    # scala lineare 10→30
    p = 1.0 - (et0_7 - 10.0) * (0.4 / 20.0)
    return clamp(p, 0.6, 1.0)

def altitude_score(alt_m: float, lat: float) -> float:
    """Quota ottimale ~ 900 + (lat-45)*(-25). Clamp 300..1400 m."""
    opt = 900.0 + (lat - 45.0) * (-25.0)
    opt = clamp(opt, 300.0, 1400.0)
    if alt_m <= 200 or alt_m >= 2200:
        return 0.2
    diff = abs(alt_m - opt)
    if diff <= 150:
        return 1.0
    if diff >= 600:
        return 0.5
    return 1.0 - (diff - 150.0) / 900.0  # scende dolce

def aspect_slope_score(aspect_deg: float, slope_deg: float, month: int, t7: float) -> float:
    """
    Bonus versanti ombreggiati (N–NE–NW) quando fa caldo; lieve bonus pend. 5–20°
    Penalty S con caldo.
    """
    bonus = 1.0
    # espo
    if t7 >= 18.0 or month in (6,7,8):
        # prefer N sector (315–45)
        in_n = (aspect_deg >= 315 or aspect_deg <= 45)
        in_ne = (45 < aspect_deg <= 90)
        in_nw = (270 <= aspect_deg < 315)
        if in_n or in_ne or in_nw:
            bonus += 0.05
        # penalty sud
        if 135 <= aspect_deg <= 225:
            bonus -= 0.1
    # pendenza
    if 5.0 <= slope_deg <= 20.0:
        bonus += 0.05
    return clamp(bonus, 0.7, 1.1)

def humidity_score(rh: Optional[float]) -> float:
    if rh is None:
        return 1.0
    if rh >= 85.0:
        return 1.05
    if rh >= 75.0:
        return 1.0
    if rh <= 55.0:
        return 0.85
    # 55→75
    return 0.85 + (rh - 55.0) * (0.15 / 20.0)

def estimate_caps_per3h(index0_100: float, forest_ok: float, uncert: float) -> Tuple[int, int]:
    """
    Stima range cappelli in 3h (non kg), dipende da indice, compatibilità bosco (0.8–1.1) e incertezza.
    """
    base = index0_100 / 10.0  # 0..10
    base *= forest_ok
    low = int(max(0, math.floor(base * 0.6)))
    high = int(max(low, math.ceil(base * 1.6)))
    # allarga se incertezza alta
    if uncert > 0.6:
        high = int(math.ceil(high * 1.3))
    return (low, high)

def forest_label_from_alt(kind_hint: Optional[str], alt_m: float) -> str:
    # fallback semplice: stima specie porcini
    if alt_m > 1400:
        return "Pinus/Abies/Picea"
    if alt_m > 900:
        return "Fagus sylvatica"
    if alt_m > 500:
        return "Castanea sativa"
    return "Quercus spp."

# ------------------ API principale ------------------

@app.get("/api/score")
async def api_score(
    q: Optional[str] = Query(None, description="Località (es. 'Bocca della Selva')"),
    coords: Optional[str] = Query(None, description="Coordinate 'lat,lon' (accetta virgola italiana)"),
):
    # 1) risolvi posizione
    if coords:
        xy = parse_coords(coords)
        if not xy:
            raise HTTPException(status_code=422, detail="Coordinate non valide")
        lat, lon = xy
        place = f"{lat:.5f},{lon:.5f}"
    elif q:
        g = await geocode(q)
        lat, lon, place = g["lat"], g["lon"], g["display"]
    else:
        raise HTTPException(status_code=400, detail="Inserisci località oppure coordinate")

    # 2) dati esterni in parallelo
    async def get_grid():
        try:
            return await elev_grid(lat, lon)
        except Exception:
            return None

    meteo_task = asyncio.create_task(open_meteo(lat, lon))
    grid_task = asyncio.create_task(get_grid())

    meteo = await meteo_task
    grid = await grid_task

    # slope/aspect/alt
    if grid:
        slope_deg, aspect_deg = slope_aspect(grid)
        elev_m = float(grid[1][1])
    else:
        slope_deg, aspect_deg, elev_m = (0.0, 180.0, safe_float(meteo.get("elevation", 0.0)))

    # 3) estrai serie giornaliere
    daily = meteo.get("daily", {})
    dates: List[str] = daily.get("time", [])
    precip = [safe_float(x, None) for x in daily.get("precipitation_sum", [])]
    tmean = [safe_float(x, None) for x in daily.get("temperature_2m_mean", [])]
    et0 = [safe_float(x, None) for x in daily.get("et0_fao_evapotranspiration", [])]
    # availability
    n = len(dates)
    if n == 0:
        raise HTTPException(status_code=502, detail="Meteo non disponibile")

    # 14 passati + 10 futuri => usiamo indici:
    past_precip = precip[:14]
    fut_precip = precip[14:]
    t7 = sum([p for p in tmean[max(0, 14-7):14] if p is not None]) / max(1, len([p for p in tmean[max(0, 14-7):14] if p is not None]))
    et0_7 = sum([p for p in et0[max(0, 14-7):14] if p is not None]) / max(1, len([p for p in et0[max(0, 14-7):14] if p is not None]))

    # pioggia efficace P14 con emivita
    days_ago = list(range(13, -1, -1))  # 13..0 (oggi)
    p14_eff = half_life_decay(days_ago, list(reversed(past_precip)), hl=7.0)

    # penalty ET0
    et0_p = et0_penalty(et0_7)

    # finestra termica
    now = datetime.now(timezone.utc).astimezone()
    month = now.month
    tw = thermal_window_score(t7, month, lat)

    # quota/lat
    alt_sc = altitude_score(elev_m, lat)

    # esposizione/pendenza
    asp_sc = aspect_slope_score(aspect_deg, slope_deg, month, t7)

    # umidità relativa (non disponibile in OM daily) -> usare proxy: se ET0 bassa e pioggia recente → bonus
    rh_proxy = None
    if p14_eff >= 30.0 and et0_7 <= 14.0:
        rh_proxy = 80.0
    rh_sc = humidity_score(rh_proxy)

    # composizione pesata 0..1
    # pesi: acqua 0.35, termica 0.25, quota 0.15, asp/slope 0.10, umidità 0.10, rad/ET0 0.05
    # normalizza p14_eff ~ 20..60 “buono”
    water_norm = clamp((p14_eff - 10.0) / 50.0, 0.0, 1.0)
    et0_norm = et0_p  # già 0.6..1.05

    score01 = (
        0.35 * water_norm +
        0.25 * tw +
        0.15 * alt_sc +
        0.10 * asp_sc +
        0.10 * rh_sc +
        0.05 * et0_norm
    )
    index = int(round(clamp(score01, 0.0, 1.0) * 100.0))

    # incertezza: varianza prossimi 10 giorni delle componenti principali (semplificato)
    fut_t = [x for x in tmean[14:] if x is not None]
    fut_p = [x for x in fut_precip if x is not None]
    var_t = (max(fut_t) - min(fut_t)) if len(fut_t) >= 2 else 0.0
    var_p = (max(fut_p) - min(fut_p)) if len(fut_p) >= 2 else 0.0
    uncert = clamp(0.3 + 0.02 * var_t + 0.01 * var_p, 0.2, 0.9)

    # stima cappelli/3h (range)
    forest_label = forest_label_from_alt(None, elev_m)
    forest_ok = 1.05 if ("Fagus" in forest_label or "Castanea" in forest_label) else 0.95
    caps_low, caps_high = estimate_caps_per3h(index, forest_ok, uncert)

    # prossima pioggia utile nei prossimi 10 gg
    next_rain_mm = None
    next_rain_day = None
    for i, mm in enumerate(fut_precip):
        if mm is not None and mm >= 2.0:
            next_rain_mm = float(mm)
            next_rain_day = i  # D+ i
            break

    # ultime piogge (ultimi 14 gg) e classi
    rain_hist = []
    for i in range(14):
        mm = past_precip[i]
        day = dates[i]
        clas = "assente"
        if mm is None or mm < 1.0:
            clas = "assente"
        elif mm < 5.0:
            clas = "modesta"
        elif mm < 15.0:
            clas = "utile"
        else:
            clas = "forte"
        rain_hist.append({"date": day, "mm": mm, "class": clas})

    # consigli/Perché (dipendono dal giorno: qui per "oggi"; il frontend li ricalcola con lo slider chiedendo D+N)
    consigli = []
    if index >= 70:
        consigli.append("Vai presto; punta a faggete/castagneti con lettiera umida.")
    elif index >= 50:
        consigli.append("Possibile raccolta: cerca ombreggiato (N–NE–NW), quote 800–1200 m.")
    else:
        consigli.append("Oggi debole: monitora prossime piogge; valuta esposizioni fresche e conca boschiva.")

    if slope_deg >= 5 and slope_deg <= 20:
        consigli.append("Pendenze 5–20° spesso produttive (drenaggio e micro-umidità).")
    if aspect_deg >= 135 and aspect_deg <= 225 and (t7 >= 18.0 or month in (6,7,8)):
        consigli.append("Evita versanti S in giornate calde (suolo si asciuga).")

    perche = (
        f"P14 efficace ~{p14_eff:.1f} mm con emivita 7 gg; "
        f"finestra termica T7={t7:.1f} °C → {int(thermal_window_score(t7, month, lat)*100)}%; "
        f"quota {int(elev_m)} m (opt lat={lat:.1f}); ET0 7g={et0_7:.1f} mm → pen {int(et0_p*100)}%; "
        f"esposizione {int(aspect_deg)}° ({'N' if aspect_deg<=45 or aspect_deg>=315 else 'S' if 135<=aspect_deg<=225 else 'E/O'}) "
        f"pendenza {slope_deg:.0f}°."
    )

    # pacchetto JSON
    out = {
        "place": place,
        "lat": lat, "lon": lon,
        "elevation_m": elev_m,
        "slope_deg": slope_deg, "aspect_deg": aspect_deg,
        "forest": forest_label,
        "index_today": index,
        "caps_3h": {"low": caps_low, "high": caps_high},
        "dates": dates,                          # per la timeline
        "precip_14": past_precip,
        "precip_next10": fut_precip,
        "tmean": tmean,
        "et0": et0,
        "next_rain_mm": next_rain_mm,
        "next_rain_in_days": next_rain_day,
        "uncert": uncert,
        "tips_today": consigli,
        "why_today": perche,
        "rain_history": rain_hist,
    }
    return out







