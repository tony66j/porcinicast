from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import math, asyncio, httpx

APP_NAME = "PorciniCast-API/0.9.3"
HEADERS  = {"User-Agent": APP_NAME}

app = FastAPI(title="PorciniCast API v0.9.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# ------------------ helper numerici ------------------

def deg_to_octant(deg: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW","N"]
    i = int((deg % 360) / 45.0 + 0.5)
    return dirs[i]

def slope_aspect_from_elev_grid(grid: List[List[float]], cell_m: float = 30.0) -> Tuple[float,float,float]:
    z = grid
    if len(z) != 3 or any(len(r) != 3 for r in z):
        return float("nan"), 0.0, 0.0
    elev_c = float(z[1][1])
    dzdx = ((z[0][2] + 2*z[1][2] + z[2][2]) - (z[0][0] + 2*z[1][0] + z[2][0])) / (8*cell_m)
    dzdy = ((z[2][0] + 2*z[2][1] + z[2][2]) - (z[0][0] + 2*z[0][1] + z[0][2])) / (8*cell_m)
    slope_deg = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect_rad = math.atan2(dzdy, -dzdx)
    if aspect_rad < 0: aspect_rad += 2*math.pi
    aspect_deg = math.degrees(aspect_rad)
    return elev_c, slope_deg, aspect_deg

def best_window_3day(scores: List[int]) -> Tuple[int,int,int]:
    if not scores or len(scores) < 3:
        return 0,0,0
    best_mean, best_i = -1, 0
    for i in range(0, len(scores)-2):
        m = (scores[i]+scores[i+1]+scores[i+2]) / 3
        if m > best_mean:
            best_mean, best_i = m, i
    return best_i, best_i+2, round(best_mean)

# ------------------ modello ------------------

def composite_score(P14: float, Tmean7: float, elev_m: float, aspect_oct: str,
                    forest_label: str, month: int, water_idx: float) -> Tuple[float, Dict[str,float]]:
    # pioggia 14g: gauss su 15–70 mm
    p_opt, p_spread = 42.5, 25.0
    p_raw = math.exp(-((P14 - p_opt)**2)/(2*p_spread**2))
    # temperatura 7g: target 12–18 °C
    t_opt, t_spread = 15.0, 6.0
    t_raw = math.exp(-((Tmean7 - t_opt)**2)/(2*t_spread**2))
    # quota stagionale (alta in estate, più bassa primavera/autunno)
    alt_opt = 1100.0 if month in (8,9) else 900.0
    a_raw = math.exp(-((elev_m - alt_opt)**2)/(2*(450.0**2)))
    # esposizione: estate + ombra
    asp_bonus = 1.0
    if month in (7,8,9):
        if aspect_oct in ("N","NE","NW"): asp_bonus = 1.12
        if aspect_oct in ("S","SE","SW"): asp_bonus = 0.9
    # compat bosco
    compat = 1.0
    k = (forest_label or "").lower()
    if "castanea" in k or "fagus" in k: compat = 1.15
    elif "quercus" in k:                compat = 1.05
    elif "pinus" in k or "abies" in k:  compat = 0.9
    # bilancio idrico (precip vs ET0: 0 molto secco, 1 ottimale)
    w_raw = max(0.0, min(1.0, water_idx))

    score = (0.36*p_raw + 0.28*t_raw + 0.16*a_raw + 0.12*w_raw) * asp_bonus * compat
    score = max(0.0, min(1.0, score))*100.0
    breakdown = {"p14n":round(p_raw,3),"tn":round(t_raw,3),"alt_n":round(a_raw,3),
                 "water":round(w_raw,3),"asp":asp_bonus,"compat":compat}
    return score, breakdown

def porcini_species(forest: str, T: float) -> List[str]:
    f = (forest or "").lower()
    warm = T >= 16
    cool = T <= 16
    res = []
    if ("quercus" in f or "castanea" in f) and warm:
        res += ["Boletus aereus (areus)", "Boletus reticulatus"]
    if ("fagus" in f) and cool:
        res += ["Boletus edulis"]
    if ("pinus" in f or "abies" in f) and cool:
        res += ["Boletus pinophilus"]
    if not res:
        res = ["Boletus edulis"]
    return res

def where_to_search(P14: float, water_idx: float, aspect_oct: str, slope_deg: float,
                    forest: str, month: int) -> str:
    tips = []
    # umidità/idrico
    if water_idx < 0.35: tips.append("cerca zone più fresche e ombrose, con suolo che trattiene umidità")
    elif water_idx > 0.75: tips.append("evita avvallamenti fradici; meglio pendii con drenaggio moderato")
    else: tips.append("suolo con umidità bilanciata: esplora margini e cambi di vegetazione")
    # esposizione stagionale
    if month in (7,8,9):
        if aspect_oct in ("N","NE","NW"): tips.append("mantieni versanti ombreggiati (N–NE–NW)")
        else: tips.append("cerca canaloni/ombreggiature rispetto al versante principale")
    else:
        tips.append("considera versanti moderatamente soleggiati nelle ore centrali")
    # pendenza
    if slope_deg < 5: tips.append("evita piani troppo umidi: prova contropendenze lievi (5–15°)")
    elif slope_deg > 25: tips.append("pendenze forti drenano: cerca terrazzi/cenge")
    else: tips.append("pendenze 5–20° spesso produttive")
    # bosco
    f=(forest or "").lower()
    if "castanea" in f: tips.append("lungo vecchi castagni e margini con quercia/faggio")
    elif "quercus" in f: tips.append("cerca suoli ben drenati: *aereus/reticulatus*")
    elif "fagus" in f: tips.append("radure, cigli e sottobosco luminoso (900–1400 m)")
    else: tips.append("preferisci latifoglie mature; evita conifere fitte")
    return "; ".join(tips)

def reason_text(P14: float, Tmean7: float, elev_m: float, aspect_oct: str, forest: str,
                last_rain_days: Optional[int], water_idx: float, month: int) -> str:
    r = []
    # pioggia
    if P14 < 15: r.append("piogge scarse negli ultimi 14 giorni")
    elif P14 > 80: r.append("piogge molto abbondanti (possibile dilavamento)")
    else: r.append("apporto di pioggia vicino al range ottimale")
    # T
    if   Tmean7 < 10: r.append("temperature fresche per i porcini")
    elif Tmean7 > 20: r.append("temperature alte ma gestibili in ombra")
    else: r.append("temperature favorevoli")
    # idrico
    if   water_idx < 0.35: r.append("suolo tendenzialmente secco (ET0>precip)")
    elif water_idx > 0.75: r.append("suolo molto bagnato (precip≫ET0)")
    else: r.append("bilancio idrico bilanciato")
    # quota/esposizione/bosco
    r.append(f"quota {int(round(elev_m))} m; esposizione {aspect_oct}; bosco: {forest}")
    # ultima pioggia
    if last_rain_days is not None:
        r.append(f"ultima pioggia ~{last_rain_days} gg fa")
    return "; ".join(r)

# ------------------ meteo ------------------

async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as c:
        r = await c.get(url, params=params); r.raise_for_status(); j = r.json()
    if not j: raise HTTPException(status_code=404, detail="Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0].get("display_name","")}

async def open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    daily = ["precipitation_sum","temperature_2m_mean","et0_fao_evapotranspiration","relative_humidity_2m_mean"]
    params = {"latitude":lat,"longitude":lon,"daily":",".join(daily),"past_days":14,"forecast_days":10,"timezone":"auto"}
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
        r = await c.get(base, params=params); r.raise_for_status(); return r.json()

async def open_elevation_grid(lat: float, lon: float, step_m: float = 30.0) -> List[List[float]]:
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m*deg_per_m_lat; dlon = step_m*deg_per_m_lon
    coords = [{"latitude": lat+dr*dlat, "longitude": lon+dc*dlon} for dr in (-1,0,1) for dc in (-1,0,1)]
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as c:
        r = await c.post(url, json={"locations": coords}); r.raise_for_status(); j = r.json()
    els = [p["elevation"] for p in j["results"]]
    return [els[0:3], els[3:6], els[6:9]]

async def overpass_forest(lat: float, lon: float, radius_m: int = 1500) -> Optional[str]:
    q = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})["natural"="wood"];
      way(around:{radius_m},{lat},{lon})["landuse"="forest"];
      relation(around:{radius_m},{lat},{lon})["natural"="wood"];
      relation(around:{radius_m},{lat},{lon})["landuse"="forest"];
    );
    out tags;
    """
    try:
        async with httpx.AsyncClient(timeout=25, headers=HEADERS) as c:
            r = await c.post("https://overpass-api.de/api/interpreter", data={"data": q})
            r.raise_for_status(); j = r.json()
    except Exception:
        return None
    labels=[]
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        lt = (tags.get("leaf_type","") or "").lower()
        if "broad" in lt or lt == "broadleaved": labels.append("broadleaved")
        elif "conifer" in lt or lt == "coniferous": labels.append("coniferous")
        else:
            w = tags.get("wood","")
            if w in ("deciduous","broadleaved"): labels.append("broadleaved")
            elif w in ("coniferous","needleleaved"): labels.append("coniferous")
    if labels:
        return "broadleaved" if labels.count("broadleaved") >= labels.count("coniferous") else "coniferous"
    return None

def forest_label_from_osm_kind(kind: Optional[str], alt_m: float) -> str:
    if kind == "coniferous": return "Pinus/Abies/Picea"
    if kind == "broadleaved":
        if alt_m > 800: return "Fagus sylvatica"
        elif alt_m > 500: return "Castanea sativa"
        else: return "Quercus spp."
    if alt_m > 1400: return "Pinus/Abies/Picea"
    if alt_m > 900:  return "Fagus sylvatica"
    if alt_m > 500:  return "Castanea sativa"
    return "Quercus spp."

# ------------------ diagnostica giorno-per-giorno ------------------

def last_rain_in_window(win14: List[float]) -> Optional[int]:
    for i in range(len(win14)-1, -1, -1):
        if (win14[i] or 0.0) > 0.5:
            return len(win14)-1 - i
    return None

def water_index(P14: float, et0_7: Optional[float]) -> float:
    if not et0_7 or et0_7 <= 0:  # se manca ET0, usa solo P14 normalizzato
        return max(0.0, min(1.0, (P14/70.0)))
    # semplice mappa  (prec - ET0) → [0..1]
    x = (P14 - et0_7)/(70.0)
    return max(0.0, min(1.0, 0.5 + x))

# ------------------ endpoints ------------------

@app.get("/api/geocode")
async def api_geocode(q: str):
    return await geocode(q)

@app.get("/api/score")
async def api_score(lat: float = Query(..., ge=-90, le=90),
                    lon: float = Query(..., ge=-180, le=180)):
    # meteo & terreno
    geodata = await open_meteo(lat, lon)
    elev_grid = await open_elevation_grid(lat, lon)
    elev_m, slope_deg, aspect_deg = slope_aspect_from_elev_grid(elev_grid, cell_m=30.0)
    kind = await overpass_forest(lat, lon)
    forest_label = forest_label_from_osm_kind(kind, elev_m)

    daily = geodata.get("daily", {})
    precip = daily.get("precipitation_sum", [])
    tempm  = daily.get("temperature_2m_mean", [])
    et0    = daily.get("et0_fao_evapotranspiration", []) or []
    rhmean = daily.get("relative_humidity_2m_mean", []) or []
    dates  = daily.get("time", [])
    if not precip or not tempm or not dates:
        raise HTTPException(status_code=502, detail="Meteo incompleto")

    # separa passato/futuro
    past14 = precip[:14]
    P14 = sum(p or 0.0 for p in past14)
    pastT = tempm[:10]
    last7T = pastT[-7:] if len(pastT)>=7 else pastT
    Tmean7 = sum(last7T)/max(1,len(last7T))
    # ET0 7 giorni (oggi): somma ultimi 7 valori disponibili
    et0_7 = None
    if et0 and len(et0) >= 7:
        et0_7 = sum((et0[i] or 0.0) for i in range(7,14))  # gli ultimi 7 del passato

    futP = precip[14:24] if len(precip) >= 24 else [0.0]*10
    futT = tempm[10:20]  if len(tempm)  >= 20 else [Tmean7]*10

    now = datetime.now(timezone.utc)
    month = now.month
    aspect_oct = deg_to_octant(aspect_deg)

    # oggi
    widx = water_index(P14, et0_7)
    score_today, breakdown = composite_score(P14, Tmean7, elev_m, aspect_oct, forest_label, month, widx)
    scores = []
    diags  = []
    win14 = list(past14)  # finestra scorrevole dei 14 giorni precedenti
    for i in range(10):
        # aggiorna finestra 14g per giorno i (domani = i=1)
        win14.pop(0); win14.append(futP[i] or 0.0)
        wP14 = sum(win14)
        Tday = futT[i]
        wEt0 = et0_7  # per semplicità manteniamo la stessa scala (migliorabile con ET0 previste)
        wid  = water_index(wP14, wEt0)
        sc,_ = composite_score(wP14, Tday, elev_m, aspect_oct, forest_label, month, wid)
        scores.append(int(round(sc)))
        lr  = last_rain_in_window(win14)
        diag = {
            "advice": ("Consigliato" if sc>=75 else ("Possibile" if sc>=55 else "Sconsigliato")),
            "reason": reason_text(wP14, Tday, elev_m, aspect_oct, forest_label, lr, wid, month),
            "where":  where_to_search(wP14, wid, aspect_oct, slope_deg, forest_label, month),
            "species_porcini": porcini_species(forest_label, Tday),
            "expected_yield": {"range_3h": ("buona" if sc>=80 else "discreta" if sc>=65 else "povera"),
                               "note":"stima qualitativa"}
        }
        diags.append(diag)

    # ultima pioggia oggi (scansione nei 14 passati)
    def last_rain_today(win): 
        for i in range(len(win)-1,-1,-1):
            if (win[i] or 0.0) > 0.5: return len(win)-1-i
        return None

    out_today = {
        "advice": ("Consigliato" if score_today>=75 else ("Possibile" if score_today>=55 else "Sconsigliato")),
        "reason": reason_text(P14, Tmean7, elev_m, aspect_oct, forest_label, last_rain_today(past14), widx, month),
        "where":  where_to_search(P14, widx, aspect_oct, slope_deg, forest_label, month),
        "species_porcini": porcini_species(forest_label, Tmean7),
        "expected_yield": {"range_3h": ("buona" if score_today>=80 else "discreta" if score_today>=65 else "povera"),
                           "note":"stima qualitativa"}
    }

    s,e,m = best_window_3day(scores)
    result = {
        "elevation_m": elev_m,
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "aspect_octant": aspect_oct,
        "forest": forest_label,
        "P14_mm": round(P14,1),
        "Tmean7_c": round(Tmean7,1),
        "ET0_7d_mm": round(et0_7,1) if et0_7 is not None else None,
        "RHmean7_pct": round(sum(rhmean[7:14])/7.0,1) if rhmean and len(rhmean)>=14 else None,

        "score_today": int(round(score_today)),
        "scores_next10": scores,
        "best_window": {"start": s, "end": e, "mean": m},
        "breakdown": breakdown,

        "diag_today": out_today,
        "diag_next10": diags
    }
    return result




