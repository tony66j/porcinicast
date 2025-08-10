from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, math, asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta

app = FastAPI(title="TrovaPorcini v5.0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

APP_HEADERS = {"User-Agent": "TrovaPorcini/5.0 (+https://example.org)"}

# ------------------------ UTILS ------------------------

def parse_coords(s: str) -> Tuple[float, float]:
    """
    Accetta:
      - "41.4170010, 14.4707168"
      - "41,4170010, 14,4707168"
      - "41.4170010 14.4707168"
    Restituisce (lat, lon) float.
    """
    if not s or not isinstance(s, str):
        raise ValueError("empty")
    t = s.strip()
    # normalizza virgole decimali italiane solo dentro ai numeri
    # 1) sostituisci "lat, lon" -> separatore ';'
    t = t.replace(";", " ").replace("|", " ").replace("\t", " ")
    # se ha due virgole e nessun punto, è probabile "lat,latdec, lon,londec"
    # strategia: prima separa su ultimo separatore tra i due numeri
    if "," in t and " " not in t:
        # caso "lat,lon" con virgole decimali italiane: trasformiamo la
        # PRIMA virgola decimale di ciascun numero in punto
        parts = t.split(",")
        if len(parts) >= 2:
            # prova a ricombinare: primi due blocchi per lat, ultimi due per lon
            # esempio "41,4170010,14,4707168"
            try:
                lat = float(parts[0] + "." + parts[1])
                lon = float(parts[2] + "." + parts[3]) if len(parts) >= 4 else float(parts[2])
                return lat, lon
            except Exception:
                pass
    # sostituisci virgola tra due numeri come separatore con spazio
    t = t.replace(",", " ")
    # ora ogni numero può avere il punto decimale
    pieces = [p for p in t.split() if p]
    if len(pieces) < 2:
        raise ValueError("missing two numbers")
    lat = float(pieces[0].replace(",", "."))
    lon = float(pieces[1].replace(",", "."))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("range")
    return lat, lon


def deg_to_octant(deg: float) -> str:
    dirs = ["N", "N-E", "E", "S-E", "S", "S-W", "W", "N-W"]
    i = int((deg % 360) / 45.0 + 0.5) % 8
    return dirs[i]


def clamp(v: float, lo: float, hi: float) -> float:
    return hi if v > hi else lo if v < lo else v


def exp_decay(days: float, half_life: float) -> float:
    # peso = 2^(-days/half_life)
    return 2 ** (-(days / max(half_life, 1e-6)))


# ------------------------ GEO & METEO ------------------------

async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "q": q, "addressdetails": 0, "limit": 1}
    async with httpx.AsyncClient(timeout=20, headers=APP_HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    if not j:
        raise HTTPException(404, "Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0].get("display_name", q)}


async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    """
    Campiona una 3x3 (≈ 30 m) attorno al punto; ritorna 3 righe x 3 colonne (m s.l.m.).
    """
    # conversione grossolana m->deg
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)))
    dlat = step_m * deg_per_m_lat
    dlon = step_m * deg_per_m_lon
    coords = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            coords.append({"latitude": lat + dr * dlat, "longitude": lon + dc * dlon})
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=15, headers=APP_HEADERS) as client:
        r = await client.post(url, json={"locations": coords})
        r.raise_for_status()
        j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]


def slope_aspect_from_grid(grid: List[List[float]], cell_m: float = 30.0) -> Tuple[float, float, float]:
    """
    Restituisce (elev_m, slope_deg, aspect_deg).
    """
    z = grid
    elev = float(z[1][1])
    # semplice gradiente centrale
    dzdx = (z[1][2] - z[1][0]) / (2 * cell_m)
    dzdy = (z[2][1] - z[0][1]) / (2 * cell_m)
    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)
    aspect_rad = math.atan2(-dzdx, dzdy)  # 0 = N
    aspect_deg = (math.degrees(aspect_rad) + 360) % 360
    return elev, slope_deg, aspect_deg


async def open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join([
            "precipitation_sum",
            "temperature_2m_mean",
            "et0_fao_evapotranspiration"
        ]),
        "hourly": "relative_humidity_2m",
        "past_days": 14, "forecast_days": 10,
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=25, headers=APP_HEADERS) as client:
        r = await client.get(base, params=params)
        r.raise_for_status()
        return r.json()


# ------------------------ MODELLO ------------------------

def thermal_window_score(T7: float, month: int, lat: float) -> float:
    # finestre utili (lineari morbide) in base a stagione
    if month in (9, 10, 11):      # autunno
        lo, hi = 9.0, 14.0
    elif month in (6, 7, 8):      # estate montana
        lo, hi = 14.0, 19.0
    else:                         # spalle
        lo, hi = 8.0, 13.0
    if T7 <= lo:  return clamp((T7 - (lo - 4)) / 4.0, 0.0, 1.0)
    if T7 >= hi:  return clamp(((hi + 4) - T7) / 4.0, 0.0, 1.0)
    return 1.0


def alt_lat_opt_score(alt_m: float, lat: float) -> float:
    # quota ottimale che scende al crescere della latitudine (N più freddo)
    opt = 900.0 + (lat - 45.0) * (-25.0)  # ≈ 900 @45N, 1300 @35N, 650 @55N
    diff = abs(alt_m - opt)
    # 0 diff -> 1.0; 500 m diff -> 0.5; 1000 m -> 0
    return clamp(1.0 - diff / 1000.0, 0.0, 1.0)


def aspect_slope_score(aspect_deg: float, slope_deg: float, hot: bool) -> float:
    # bonus N-NE-NW in caldo; penalità S in caldo; lieve bonus pendenza 5–20°
    if hot:
        bonus = 1.0 if aspect_deg is None else (
            1.0 if 300 <= aspect_deg or aspect_deg <= 60 else (0.6 if 120 <= aspect_deg <= 240 else 0.8)
        )
    else:
        # neutro fuori caldo
        bonus = 0.9
    pen_slope = 1.0
    if slope_deg is not None:
        if 5 <= slope_deg <= 20:
            pen_slope = 1.05
        elif slope_deg > 35:
            pen_slope = 0.85
    return clamp(bonus * pen_slope, 0.6, 1.1)


def humidity_score(rh7: Optional[float]) -> float:
    if rh7 is None:  # se mancano dati, neutro leggermente prudente
        return 0.9
    if rh7 >= 85: return 1.05
    if rh7 >= 75: return 1.0
    if rh7 >= 65: return 0.9
    return 0.8


def radiation_penalty(et0_7: float) -> float:
    # ET0 alto = più asciugatura lettiera
    if et0_7 <= 2.0: return 1.0
    if et0_7 >= 6.0: return 0.8
    # lineare tra 2 e 6
    return 1.0 - (et0_7 - 2.0) * 0.05


def moisture_component(precip14: List[float], et0_14: List[float]) -> Tuple[float, float, float]:
    # pioggia efficace (emivita 7 gg) normalizzata da ET0 cumulata
    eff = 0.0
    for i, p in enumerate(precip14):
        age = len(precip14) - 1 - i  # 0 = oggi, 13 = 13 gg fa
        w = exp_decay(age, 7.0)
        eff += p * w
    et0_7 = sum(et0_14[-7:]) / 7.0 if et0_14 else 0.0
    # normalizzazione: ~20–60 mm eff = 0.4–1.0 (sopra penalità meno forte)
    if eff <= 10: m = 0.2
    elif eff >= 60: m = 1.0
    else: m = 0.2 + (eff - 10) * (0.8 / 50.0)
    # penalità ET0 alto
    if et0_7 > 0:
        m *= radiation_penalty(et0_7)
    return clamp(m, 0.0, 1.0), eff, et0_7


def species_guess(forest_label: str, alt_m: float, month: int) -> List[str]:
    out = []
    if "broad" in forest_label.lower() or "castanea" in forest_label.lower() or "fagus" in forest_label.lower():
        out.append("Boletus edulis / reticulatus")
    if "conifer" in forest_label.lower() or "pinus" in forest_label.lower() or "abies" in forest_label.lower():
        out.append("Boletus pinophilus")
    if not out:
        out = ["Boletus spp."]
    return out


async def overpass_forest(lat: float, lon: float, radius_m: int = 800) -> Optional[str]:
    # prova a derivare broadleaved/coniferous dall'intorno
    query = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})[natural=wood];
      relation(around:{radius_m},{lat},{lon})[natural=wood];
    );
    out tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    async with httpx.AsyncClient(timeout=25, headers=APP_HEADERS) as client:
        r = await client.post(url, data={"data": query})
        r.raise_for_status()
        j = r.json()
    labels = []
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        lt = tags.get("leaf_type", "").lower()
        if "broad" in lt:
            labels.append("broadleaved")
        elif "conifer" in lt:
            labels.append("coniferous")
        elif "wood" in tags:
            if tags["wood"] in ("deciduous", "broadleaved"):
                labels.append("broadleaved")
            elif tags["wood"] in ("coniferous", "needleleaved"):
                labels.append("coniferous")
    if labels:
        b = labels.count("broadleaved")
        c = labels.count("coniferous")
        return "broadleaved" if b >= c else "coniferous"
    return None


def advice_text(day_idx: int, hot: bool, aspect_deg: float, slope_deg: float,
                alt_m: float, next_rain_days: Optional[int]) -> str:
    octant = deg_to_octant(aspect_deg) if aspect_deg is not None else "—"
    tips = []
    if hot:
        tips.append("cerca ombreggiato (N–NE–NW) e suoli che trattengono umidità")
    else:
        tips.append("cerca lettiere mature e suoli ben drenati")
    if 5 <= (slope_deg or 0) <= 20:
        tips.append("pendenze 5–20° spesso produttive")
    elif (slope_deg or 0) > 35:
        tips.append("evita versanti troppo ripidi")
    tips.append(f"esposizione locale {octant}")
    if next_rain_days is not None and next_rain_days <= 3:
        tips.append("dopo la pioggia attesa, torna entro 7–10 giorni per la finestra")
    return "; ".join(tips)


def because_text(idx: int, eff_mm: float, et0_7: float, T7: float, alt_m: float, lat: float,
                 aspect_deg: float, rh7: Optional[float], month: int, next_rain: Optional[Tuple[str,float,int]]) -> str:
    octant = deg_to_octant(aspect_deg) if aspect_deg is not None else "—"
    parts = []
    parts.append(f"P14 efficace ~{round(eff_mm,1)} mm (emivita 7 gg)")
    parts.append(f"T media 7 gg: {round(T7,1)} °C")
    parts.append(f"ET0 7 gg: {round(et0_7,1)} mm → effetto asciugatura")
    parts.append(f"quota {int(round(alt_m))} m @ lat {round(lat,3)}; esposizione {octant}")
    if rh7 is not None:
        parts.append(f"UR media 7 gg ~{int(round(rh7))}%")
    if next_rain:
        dt, mm, d = next_rain
        parts.append(f"prossima pioggia utile: {mm} mm il {dt} (D+{d})")
    reason = " | ".join(parts)
    if idx >= 70:
        reason = "Condizioni buone: " + reason
    elif idx >= 40:
        reason = "Condizioni possibili: " + reason
    else:
        reason = "Condizioni deboli: " + reason
    return reason


def caps_3h_estimate(idx: int, forest_ok: float, variance: float) -> Tuple[int, int]:
    # base 0–100 -> 0–18 cappelli
    base = idx / 100.0 * 18.0
    base *= forest_ok
    # ampiezza intervallo cresce con varianza (0..1)
    spread = 2 + 6 * clamp(variance, 0.0, 1.0)
    lo = max(0, int(round(base - spread/2)))
    hi = max(lo, int(round(base + spread/2)))
    return lo, hi


# ------------------------ ENDPOINT ------------------------

@app.get("/api/score")
async def api_score(q: Optional[str] = Query(None),
                    lat: Optional[float] = Query(None),
                    lon: Optional[float] = Query(None),
                    coords: Optional[str] = Query(None),
                    day: int = Query(0, ge=0, le=9)) -> Dict[str, Any]:
    """
    Calcola indice e dettagli per oggi + prossimi 9 giorni.
    Parametri accettati:
      - q: stringa località (Nominatim)
      - lat, lon: float
      - coords: stringa "lat,lon" (accetta virgola italiana)
      - day: 0..9 (giorno selezionato)
    Le coordinate, se presenti, hanno priorità su 'q'.
    """
    # 1) coordinate
    src = "geocode"
    if coords:
        try:
            lat, lon = parse_coords(coords)
            src = "coords"
        except Exception:
            raise HTTPException(422, "Coordinate non valide")
    if lat is None or lon is None:
        if q:
            g = await geocode(q)
            lat, lon = g["lat"], g["lon"]
            src = "geocode"
        else:
            raise HTTPException(422, "Inserisci località o coordinate")

    # 2) dati geo: quota, slope, aspect
    grid = await open_elevation_grid(lat, lon)
    elev_m, slope_deg, aspect_deg = slope_aspect_from_grid(grid)

    # 3) bosco (best effort, opzionale)
    forest_kind = await overpass_forest(lat, lon) or "unknown"
    forest_label = "broadleaved" if forest_kind == "broadleaved" else ("coniferous" if forest_kind == "coniferous" else "mixed/unknown")

    # 4) meteo
    meteo = await open_meteo(lat, lon)
    d = meteo["daily"]
    dates_all: List[str] = d["time"]                      # 14 passati + 10 futuri (ordine crescente)
    pr_all: List[float] = d["precipitation_sum"]
    tm_all: List[float] = d["temperature_2m_mean"]
    et0_all: List[float] = d.get("et0_fao_evapotranspiration", [0.0] * len(dates_all))

    # separa 14 passati e 10 futuri
    past14_dates = dates_all[:14]
    next10_dates = dates_all[14:]
    past14_pr = pr_all[:14]
    next10_pr = pr_all[14:]
    past14_et0 = et0_all[:14]
    next10_et0 = et0_all[14:]
    past14_tm = tm_all[:14]
    next10_tm = tm_all[14:]

    # T7, RH7 (da hourly se disponibile)
    rh7 = None
    try:
        hh = meteo.get("hourly", {})
        ts = hh.get("time", [])
        rh = hh.get("relative_humidity_2m", [])
        if ts and rh:
            # ultime 7*24
            rh7 = sum(rh[-7*24:]) / (7*24)
    except Exception:
        rh7 = None

    T7 = sum(past14_tm[-7:]) / 7.0
    month = datetime.now(timezone.utc).astimezone().month
    hot = month in (6,7,8)

    # Moisture component
    moist_s, eff_mm, et0_7 = moisture_component(past14_pr, past14_et0)
    # Thermal window
    therm_s = thermal_window_score(T7, month, lat)
    # Alt/lat
    alt_s = alt_lat_opt_score(elev_m, lat)
    # Aspect/slope
    asp_s = aspect_slope_score(aspect_deg, slope_deg, hot)
    # Humidity
    hum_s = humidity_score(rh7)

    # Ponderazione
    comp = (
        0.35 * moist_s +
        0.25 * therm_s +
        0.15 * alt_s +
        0.10 * asp_s +
        0.10 * hum_s +
        0.05 * radiation_penalty(et0_7)
    )
    index_today = int(round(clamp(comp, 0.0, 1.0) * 100))

    # Serie prossimi 10 gg: ricalcola indice per D+0..D+9 usando previsioni
    scores_next10: List[int] = []
    for i in range(10):
        # per semplicità: ricalcolo moisture con P14 (13 passati + oggi_futuro_i)
        # usiamo le stesse componenti ma T dal next10_tm[i], ET0 dal next10_et0[i]
        Tsel = (sum(past14_tm[-6:]) + next10_tm[i]) / 7.0
        therm_i = thermal_window_score(Tsel, month, lat)
        # moisture: somma P14 storica con decadimento + aggiungi forecast cumulata
        pr14 = past14_pr.copy()
        pr14[-1] = pr14[-1]  # oggi reale
        # non “muoviamo” la finestra qui: aggiusta peso con forecast immediato
        pr14_eff = pr14
        moist_i, eff_i, et0_dummy = moisture_component(pr14_eff, past14_et0)
        # bonus se nei prossimi i giorni piove
        if next10_pr[i] >= 3.0:
            moist_i = clamp(moist_i + 0.08, 0.0, 1.0)
        score_i = (
            0.35 * moist_i +
            0.25 * therm_i +
            0.15 * alt_s +
            0.10 * asp_s +
            0.10 * hum_s +
            0.05 * radiation_penalty(next10_et0[i] if i < len(next10_et0) else et0_7)
        )
        scores_next10.append(int(round(clamp(score_i, 0.0, 1.0) * 100)))

    # prossima pioggia utile nei prossimi 10 giorni
    next_rain = None
    for i, mm in enumerate(next10_pr):
        if mm >= 1.0:
            next_rain = (next10_dates[i], round(mm, 1), i)  # (data, mm, D+i)
            break

    # specie probabili
    species = species_guess(forest_label, elev_m, month)

    # stima cappelli 3h
    variance = max(0.0, 1.0 - (sum(scores_next10)/10.0)/100.0)  # rozza incertezza
    forest_ok = 1.05 if "broad" in forest_label else (1.0 if "mixed" in forest_label else 0.95)
    lo_caps, hi_caps = caps_3h_estimate(index_today, forest_ok, variance)

    # testi
    because = because_text(index_today, eff_mm, et0_7, T7, elev_m, lat, aspect_deg, rh7, month, next_rain)
    advice = advice_text(day, hot, aspect_deg, slope_deg, elev_m, next_rain[2] if next_rain else None)

    # piogge passate/future (lista data+mm)
    past_rains = [{"date": d, "mm": round(mm, 1)} for d, mm in zip(past14_dates, past14_pr)]
    next_rains = [{"date": d, "mm": round(mm, 1)} for d, mm in zip(next10_dates, next10_pr)]

    result = {
        "lat": round(lat, 6), "lon": round(lon, 6),
        "elev_m": round(elev_m, 1),
        "slope_deg": round(slope_deg, 1),
        "aspect_deg": round(aspect_deg, 1),
        "aspect_octant": deg_to_octant(aspect_deg),
        "forest": forest_label,
        "index": index_today,
        "scores_next10": scores_next10,
        "p14_mm": round(sum(past14_pr), 1),
        "tmean7_c": round(T7, 1),
        "et0_7g": round(et0_7, 1),
        "advice": advice,
        "because": because,
        "caps_3h": {"low": lo_caps, "high": hi_caps},
        "species_prob": species,
        "dates_past14": past14_dates,
        "rain_past14_mm": [round(x,1) for x in past14_pr],
        "dates_next10": next10_dates,
        "rain_next10_mm": [round(x,1) for x in next10_pr],
        "tmean_next10_c": [round(x,1) for x in next10_tm],
        "next_rain": {"date": next_rain[0], "mm": next_rain[1], "d": next_rain[2]} if next_rain else None,
        "source": src,
    }
    return result







