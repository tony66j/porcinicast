
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
import math, asyncio, re, httpx

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

APP_NAME = "TrovaPorcini-v5.1"
HEADERS  = {"User-Agent": APP_NAME}
app = FastAPI(title="TrovaPorcini API v5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────────────────────

COORD_RE = re.compile(r"[-+]?\d{1,3}(?:[.,]\d+)?")

def parse_coords(text: str) -> Optional[Tuple[float, float]]:
    """
    Accetta formati tipo:
      - "41.4170010, 14.4707168"
      - "41,4170010 14,4707168"
      - "lat=41.417; lon=14.4707"
    Restituisce (lat, lon) oppure None se non trova due numeri validi.
    """
    if not text:
        return None
    nums = [n.group(0) for n in COORD_RE.finditer(text)]
    if len(nums) < 2:
        return None
    lat_s, lon_s = nums[0], nums[1]
    # virgola italiana → punto
    lat = float(lat_s.replace(",", "."))
    lon = float(lon_s.replace(",", "."))
    # range veloci
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return (lat, lon)

async def geocode(q: str) -> Optional[Dict[str, Any]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        j = r.json()
        if not j:
            return None
        return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0]["display_name"]}

async def open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    """
    Daily: ultimi 14 gg + prossimi 10 gg in un colpo solo (Open-Meteo).
    Include precip, T media, ET0 FAO, umidità e radiazione solare.
    """
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "timezone": "auto",
        "past_days": 14, "forecast_days": 10,
        "daily": ",".join([
            "precipitation_sum",
            "temperature_2m_mean",
            "et0_fao_evapotranspiration",
            "shortwave_radiation_sum",
            "relative_humidity_2m_mean"
        ])
    }
    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        return r.json()

async def elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    """
    Griglia 3x3 da Open-Elevation per calcolare pendenza/aspetto in modo robusto.
    """
    # 1° ~ 111_320 m; approssimazione sufficiente
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)))
    dlat = step_m * deg_per_m_lat
    dlon = step_m * deg_per_m_lon
    coords = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            coords.append({"latitude": lat + dr*dlat, "longitude": lon + dc*dlon})
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        r = await client.post(url, json={"locations": coords})
        r.raise_for_status()
        els = [p["elevation"] for p in r.json()["results"]]
        return [els[0:3], els[3:6], els[6:9]]

def slope_aspect_from_grid(grid: List[List[float]], cell_size_m: float = 30.0) -> Tuple[float, float]:
    """
    Restituisce (pendenza in gradi, aspect in gradi 0=N → 360).
    """
    import numpy as np
    Z = np.array(grid, dtype=float)
    # Sobel-like
    dzdx = ((Z[0,2]+2*Z[1,2]+Z[2,2]) - (Z[0,0]+2*Z[1,0]+Z[2,0])) / (8*cell_size_m)
    dzdy = ((Z[2,0]+2*Z[2,1]+Z[2,2]) - (Z[0,0]+2*Z[0,1]+Z[0,2])) / (8*cell_size_m)
    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)
    aspect_rad = math.atan2(dzdx, dzdy)  # y verso nord
    aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
    return slope_deg, aspect_deg

def aspect_octant_label(aspect_deg: float) -> str:
    labels = ["N","NE","E","SE","S","SW","W","NW"]
    idx = int(((aspect_deg + 22.5) % 360) // 45)
    return labels[idx]

# ──────────────────────────────────────────────────────────────────────────────
# MODELLO PREVISIONALE (pesi e calcoli)
# ──────────────────────────────────────────────────────────────────────────────

def half_life_decay(weights_days: List[float], half_life_days: float = 7.0) -> List[float]:
    lam = math.log(2.0)/half_life_days
    return [math.exp(-lam*d) for d in weights_days]

def window_score(T7: float, month: int, lat: float) -> float:
    """
    Finestra termica dinamica: 9–14 °C in autunno; 14–19 °C in estate montana (lat più alta → finestra più bassa).
    """
    if month in (6,7,8):  # estate
        target = (14.0, 19.0)
    else:
        target = (9.0, 14.0)
    # latitudine: al Nord la finestra si abbassa leggermente
    lat_adj = (45.0 - lat) * 0.05
    lo, hi = target[0] + lat_adj, target[1] + lat_adj
    if T7 <= lo: return max(0.0, 1.0 - (lo - T7)/8.0)
    if T7 >= hi: return max(0.0, 1.0 - (T7 - hi)/8.0)
    return 1.0

def alt_score(alt_m: float, lat: float) -> float:
    # optimum ~ 900 + (lat-45)*(-25)  (clamp 300–1400)
    opt = 900.0 + (lat - 45.0) * (-25.0)
    opt = max(300.0, min(1400.0, opt))
    return max(0.0, 1.0 - abs(alt_m - opt)/800.0)

def exp_slope_score(aspect_deg: float, slope_deg: float, hot: bool) -> float:
    # prefer N–NE–NW con caldo; penalità S in caldo secco; bonus lieve con pendenze 5–20°
    aspect_bonus = 1.0
    if hot:
        if 300 <= aspect_deg or aspect_deg <= 60:  # N sector
            aspect_bonus = 1.0
        elif 120 <= aspect_deg <= 240:            # S sector
            aspect_bonus = 0.7
        else:
            aspect_bonus = 0.9
    slope_bonus = 1.0 + (0.1 if 5.0 <= slope_deg <= 20.0 else 0.0)
    return max(0.0, min(1.2, aspect_bonus*slope_bonus))

def humidity_score(rh7: float) -> float:
    if rh7 is None: return 0.7
    if rh7 >= 85: return 1.0
    if rh7 >= 75: return 0.9
    if rh7 >= 65: return 0.75
    return 0.5

def rad_penalty(sw7: float) -> float:
    # tanta radiazione secca la lettiera → piccola penalità
    if sw7 is None: return 1.0
    # normalizza circa 10–20 MJ/m2/d
    if sw7 <= 10: return 1.0
    if sw7 >= 22: return 0.8
    return 1.0 - 0.2*(sw7-10.0)/(12.0)

def moist_effective(p14_mm: List[float], et0_mm: List[float]) -> float:
    """
    Acqua “utile” con decadimento: eventi vecchi valgono meno; ET0 maggiore riduce il beneficio.
    """
    days = list(range(len(p14_mm)-1, -1, -1))  # 13..0
    decay = half_life_decay(days, 7.0)
    eff = 0.0
    for i, mm in enumerate(p14_mm):
        age_w = decay[len(p14_mm)-1-i]
        et = et0_mm[i] if et0_mm and i < len(et0_mm) else 3.0
        et_pen = max(0.5, 1.0 - (et-3.0)*0.06)  # ET0 alta → riduzione
        eff += mm * age_w * et_pen
    # normalizzazione “buona” ~ 20–60 mm
    if eff <= 15: return eff/30.0
    if eff >= 70: return 1.0
    return 0.5 + 0.5*(eff-15)/55.0

def day_score(d: Dict[str, Any]) -> float:
    """
    Combinazione pesata dei 6 pilastri (0–1) → ritorna 0–100.
    """
    w = {"moist":0.35, "window":0.25, "alt":0.15, "expslope":0.10, "humidity":0.10, "rad":0.05}
    s = (w["moist"]    * d["moist"]    +
         w["window"]   * d["window"]   +
         w["alt"]      * d["alt"]      +
         w["expslope"] * d["expslope"] +
         w["humidity"] * d["humidity"] +
         w["rad"]      * d["rad"])
    return max(0.0, min(100.0, 100.0*s))

def caps_3h_estimate(score: float) -> str:
    """
    Stima “cappelli/3h” indicativa (intervallo): più robusta e comprensibile.
    """
    if score < 30: return "0–2"
    if score < 45: return "1–4"
    if score < 60: return "3–8"
    if score < 75: return "6–12"
    if score < 85: return "9–15"
    return "12–20"

# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/score")
async def api_score(
    q: Optional[str] = Query(None, description="Località"),
    coords: Optional[str] = Query(None, description="Stringa coordinate varie (lat,lon)"),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
):
    """
    Le coordinate hanno priorità:
      - se 'coords' è valorizzato, provo a parsare; altrimenti uso 'lat' e 'lon';
      - se nessuna coord valida, uso geocoding su q.
    """
    # 1) risolvi coordinate
    point = None
    if coords:
        try:
            point = parse_coords(coords)
        except Exception:
            point = None
    if point is None and (lat is not None and lon is not None):
        point = (lat, lon)
    if point is None and q:
        g = await geocode(q)
        if not g:
            # niente: errore di località
            raise httpx.HTTPStatusError("Località non trovata", request=None, response=None)
        point = (g["lat"], g["lon"])
        display = g["display"]
    else:
        display = None

    if point is None:
        # errore “pulito” lato client
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Coordinate non valide")

    lat, lon = point

    # 2) richieste in parallelo: meteo + elevazione
    mtask = asyncio.create_task(open_meteo(lat, lon))
    gtask = asyncio.create_task(elevation_grid(lat, lon))
    meteo, grid = await asyncio.gather(mtask, gtask)

    daily = meteo["daily"]
    dates  = daily["time"]               # 14 passati + 10 futuri
    precip = daily["precipitation_sum"]  # mm
    tmean  = daily["temperature_2m_mean"]
    et0    = daily.get("et0_fao_evapotranspiration", [3.0]*len(dates))
    rh     = daily.get("relative_humidity_2m_mean", [75.0]*len(dates))
    sw     = daily.get("shortwave_radiation_sum", [15.0]*len(dates))

    # separa ultimi 14 e prossimi 10
    pastN  = 14
    futN   = 10
    past_prec = precip[:pastN]
    past_et0  = et0[:pastN]
    fut_prec  = precip[pastN:]
    fut_t     = tmean[pastN:]
    fut_et0   = et0[pastN:]
    fut_rh    = rh[pastN:]
    fut_sw    = sw[pastN:]

    # pendenza/aspetto/altitudine
    elev_m = grid[1][1]
    slope_deg, aspect_deg = slope_aspect_from_grid(grid)
    aspect_label = aspect_octant_label(aspect_deg)

    # Indici day-by-day per prossimi 10 giorni (D+0..D+9)
    # moisture calcolato su coda (ultimi 14) + finestra dinamica con T7 (ultimi 7 di ciascun giorno)
    scores = []
    why    = []
    today  = datetime.now(timezone.utc).astimezone()
    lat_abs = abs(lat)

    # player “prossima pioggia”
    next_rain_day = None
    next_rain_mm  = 0.0
    for i, mm in enumerate(fut_prec):
        if mm >= 0.5:
            next_rain_day = i
            next_rain_mm  = float(mm)
            break

    # calcolo rolling per ciascun giorno futuro (usiamo i 14 giorni che “scorrono” aggiungendo futur)
    rolling_p = past_prec[:]
    rolling_et = past_et0[:]
    for d in range(futN):
        # moisture con decadimento ed ET0
        moist = moist_effective(rolling_p[-14:], rolling_et[-14:])

        # T media 7 gg per la finestra termica
        # prendo ultimi 7 (passati 7, o se servono futuri uso fut_t)
        t7_window = tmean[pastN-7:pastN]  # base: ultimi 7 storici
        # se servono giorni futuri perché siamo oltre D+0, allunga
        if d > 0:
            need = d
            extra = fut_t[:need]
            # shift: togli i più vecchi e aggiungi extra
            t7 = (tmean[pastN-7+need:pastN] + extra) if need < 7 else extra[-7:]
        else:
            t7 = t7_window
        T7 = sum(t7)/len(t7) if t7 else fut_t[d]

        # humidity e rad (uso future arrays sincronizzati)
        RH7 = fut_rh[d] if fut_rh[d] is not None else 75.0
        SW7 = fut_sw[d] if fut_sw[d] is not None else 15.0

        hot = T7 >= 18.0
        window = window_score(T7, month=today.month, lat=lat)
        es = exp_slope_score(aspect_deg, slope_deg, hot)
        alt = alt_score(elev_m, lat)

        sc = day_score({
            "moist": moist,
            "window": window,
            "alt": alt,
            "expslope": es,
            "humidity": humidity_score(RH7),
            "rad": rad_penalty(SW7),
        })
        scores.append(int(round(sc)))

        why.append({
            "T7_c": round(T7,1),
            "RH7_pct": round(RH7,1) if RH7 is not None else None,
            "SW7_MJ": round(SW7,1) if SW7 is not None else None,
            "moist_eff": round(moist,2),
        })

        # aggiorna rolling con il giorno d+1
        rolling_p.append(fut_prec[d])
        rolling_et.append(fut_et0[d] if d < len(fut_et0) else 3.0)

    # indice oggi, stima cappelli 3h
    score_today = scores[0] if scores else 0
    caps_today  = caps_3h_estimate(score_today)

    # pioggia “prossima” (mm e data)
    next_rain_date = (today + timedelta(days=next_rain_day)).date().isoformat() if next_rain_day is not None else None

    # costruisci consigli/“perché” testuali per il giorno selezionato lato client (ma offro un default per oggi)
    consigli_oggi = (
        f"cerca ombreggiato (N–NE–NW) e suoli che trattengono umidità; "
        f"evita versanti troppo ripidi; esposizione locale {aspect_label.lower()} | "
        f"se pioggia è attesa, torna tra 7–10 giorni per la finestra"
    )
    perche_oggi = (
        f"Condizioni possibili: P14 efficace ~{round(moist_effective(past_prec[-14:], past_et0[-14:]),1)} mm "
        f"(emivita 7 gg); finestra T7 ~{round(sum(tmean[pastN-7:pastN])/7.0,1)} °C; "
        f"ET0 recente media ~{round(sum(past_et0[-7:])/7.0,1)} mm; quota {int(elev_m)} m; "
        f"esposizione {aspect_label}; prossima pioggia utile: "
        f"{(str(next_rain_mm)+' mm') if next_rain_day is not None else '—'}"
    )

    return {
        "display": display,
        "coords": {"lat": lat, "lon": lon},
        "elevation_m": round(elev_m,1),
        "slope_deg": round(slope_deg,1),
        "aspect_deg": round(aspect_deg,1),
        "aspect_octant": aspect_label,
        "score_today": score_today,
        "caps_3h": caps_today,
        "dates": dates[pastN:],              # 10 date ISO
        "scores_next10": scores,             # 10 valori
        "rain_next10_mm": fut_prec,          # mm (10)
        "tmean_next10_c": fut_t,             # °C (10)
        "next_rain": {
            "days_ahead": next_rain_day,
            "mm": next_rain_mm,
            "date": next_rain_date
        },
        "advice_today": consigli_oggi,
        "why_today": perche_oggi,
        "tech": {
            "P14_last14_mm": sum(past_prec),
            "Tmean7_c": round(sum(tmean[pastN-7:pastN])/7.0,1),
            "ET0_7d_mm": round(sum(past_et0[-7:])/7.0,1),
            "RH7_pct": round(fut_rh[0],1) if fut_rh and fut_rh[0] is not None else None,
            "SW7_MJ": round(fut_sw[0],1) if fut_sw and fut_sw[0] is not None else None
        }
    }

@app.get("/api/geocode")
async def api_geocode(q: str):
    g = await geocode(q)
    if not g:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Località non trovata")
    return g

@app.get("/")
def root():
    return {"ok": True, "msg": "TrovaPorcini API v5.1"}






