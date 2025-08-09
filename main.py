from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import math, asyncio, os
import httpx

# ——————————————————————————————————————————————————————————
# App (Uvicorn cerca 'app', noi usiamo APP e aliasiamo)
# ——————————————————————————————————————————————————————————
APP = FastAPI(title="TrovaPorcini API v0.96")
app = APP  # 👈 alias richiesto da Uvicorn: uvicorn main:app

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UA = "TrovaPorcini/0.96 (+https://example.org)"
HEAD = {"User-Agent": UA}
OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

# ——————————————————————————————————————————————————————————
# Utilità numeriche / geografiche
# ——————————————————————————————————————————————————————————
def clamp(x, a, b): 
    return max(a, min(b, x))

def deg_to_octant(deg: float) -> str:
    # N, NE, E, SE, S, SW, W, NW
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    i = int((deg % 360) / 45.0 + 0.5) % 8
    return dirs[i]

def exposure_label(aspect_deg: Optional[float]) -> str:
    if aspect_deg is None: return "N/A"
    return deg_to_octant(aspect_deg)

def slope_aspect_from_grid(elev9: List[List[float]], cell_m: float = 30.0) -> Tuple[float, float]:
    # ele9 è 3x3 [r][c], r=0..2 (S→N), c=0..2 (W→E). Calcolo Horn.
    z = elev9
    if not z or len(z)!=3 or any(len(r)!=3 for r in z): 
        return 0.0, None
    dzdx = ((z[0][2] + 2*z[1][2] + z[2][2]) - (z[0][0] + 2*z[1][0] + z[2][0]))/(8*cell_m)
    dzdy = ((z[2][0] + 2*z[2][1] + z[2][2]) - (z[0][0] + 2*z[0][1] + z[0][2]))/(8*cell_m)
    slope_rad = math.atan(math.sqrt(dzdx*dzdx + dzdy*dzdy))
    slope_deg = math.degrees(slope_rad)
    # aspect azimut (0=N, 90=E); Horn
    if dzdx==0 and dzdy==0:
        aspect = None
    else:
        aspect = math.degrees(math.atan2(dzdx, dzdy))
        if aspect < 0: aspect += 360.0
    return slope_deg, aspect

# ——————————————————————————————————————————————————————————
# Servizi esterni
# ——————————————————————————————————————————————————————————
async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"format":"json","q":q,"addressdetails":1,"limit":1}
    async with httpx.AsyncClient(timeout=15, headers={**HEAD,"Accept-Language":"it"}) as cli:
        r = await cli.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    if not j: 
        return {}
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0]["display_name"]}

async def open_elev_grid(lat: float, lon: float) -> List[List[float]]:
    # campiono 3x3 con ~30m spacing (approssimazione in gradi)
    step_m = 30.0
    deg_per_m_lat = 1.0/111320.0
    deg_per_m_lon = 1.0/(111320.0*math.cos(math.radians(lat)))
    dlat = step_m * deg_per_m_lat
    dlon = step_m * deg_per_m_lon
    coords = []
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            coords.append({"latitude": lat + dr*dlat, "longitude": lon + dc*dlon})
    url = "https://api.open-elevation.com/api/v1/lookup"
    async with httpx.AsyncClient(timeout=15, headers=HEAD) as cli:
        r = await cli.post(url, json={"locations": coords})
        r.raise_for_status()
        j = r.json()
    vals = [p["elevation"] for p in j["results"]]
    return [vals[0:3], vals[3:6], vals[6:9]]

async def overpass_forest(lat: float, lon: float, radius_m: int = 800) -> Optional[str]:
    # prova a distinguere broadleaved/coniferous
    q = f"""
[out:json][timeout:25];
(
  way(around:{radius_m},{lat},{lon})[natural=wood];
  relation(around:{radius_m},{lat},{lon})[natural=wood];
);
out tags center 50;
"""
    url = "https://overpass-api.de/api/interpreter"
    async with httpx.AsyncClient(timeout=30, headers=HEAD) as cli:
        r = await cli.post(url, data={"data": q})
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
            v = tags["wood"]
            if v in ("deciduous","broadleaved"): labels.append("broadleaved")
            if v in ("coniferous","needleleaved"): labels.append("coniferous")
    if not labels: return None
    return "broadleaved" if labels.count("broadleaved") >= labels.count("coniferous") else "coniferous"

def forest_label(kind: Optional[str], alt_m: float) -> str:
    # fallback semplice per quota
    if kind == "coniferous": return "Pinus/Abies/Picea"
    if kind == "broadleaved":
        if alt_m > 800: return "Fagus sylvatica"
        if 500 < alt_m <= 800: return "Castanea sativa"
        return "Quercus spp."
    # solo altitudine
    if alt_m > 1400: return "Pinus/Abies/Picea"
    if alt_m > 900:  return "Fagus sylvatica"
    if alt_m > 500:  return "Castanea sativa"
    return "Quercus spp."

async def meteo_openmeteo(lat: float, lon: float) -> Dict[str, Any]:
    """
    Open-Meteo: daily con past_days=14, forecast_days=10.
    Usiamo precipitation_sum, temperature_2m_mean, relative_humidity_2m_mean, et0_fao_evapotranspiration.
    """
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join(["precipitation_sum","temperature_2m_mean","relative_humidity_2m_mean","et0_fao_evapotranspiration"]),
        "past_days": 14, "forecast_days": 10, "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=25, headers=HEAD) as cli:
        r = await cli.get(base, params=params)
        r.raise_for_status()
        j = r.json()
    return j

async def meteo_openweather_daily(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    if not OWM_KEY: return None
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {"lat":lat, "lon":lon, "exclude":"minutely,hourly,alerts", "units":"metric", "appid": OWM_KEY}
    async with httpx.AsyncClient(timeout=20, headers=HEAD) as cli:
        r = await cli.get(url, params=params)
        if r.status_code != 200:
            return None
        return r.json()

# ——————————————————————————————————————————————————————————
# Modello indice/consigli (semplificato ma coerente)
# ——————————————————————————————————————————————————————————
def window_3day(scores: List[int]) -> Tuple[int,int,float]:
    best_s = -1.0; best_i = 0
    for i in range(0, len(scores)-2):
        m = (scores[i]+scores[i+1]+scores[i+2])/3.0
        if m > best_s: best_s, best_i = m, i
    return best_i, best_i+2, round(best_s,1)

def composite_score(P14: float, Tmean7: float, elev_m: float, aspect_oct: str, forest_lbl: str, month: int) -> Tuple[float, Dict[str,float]]:
    # componenti 0..1
    # pioggia: emivita ~8 gg → più recente pesa di più (già riassunta in P14)
    p = clamp(P14/35.0, 0, 1)  # 35 mm in 14g ~ buono
    # temp stagionale (ottimo 12–20°C)
    if Tmean7 < 5: t=0
    elif Tmean7 > 24: t=0.2
    else:
        t = 1.0 - abs((Tmean7-16)/11.0)  # campana larga
        t = clamp(t, 0, 1)
    # quota/latitudine stagionale grossolana
    # autunno: ok 400–1500 m; estate: 900–1800 m; primavera: 300–1000 m
    if month in (9,10,11):
        prefer = (400,1500)
    elif month in (6,7,8):
        prefer = (900,1800)
    else:
        prefer = (300,1000)
    if elev_m < prefer[0]: z= max(0.0, 1.0-(prefer[0]-elev_m)/400.0)
    elif elev_m > prefer[1]: z= max(0.0, 1.0-(elev_m-prefer[1])/600.0)
    else: z = 1.0
    # esposizione: NE–NW leggero bonus (più umido), S penalità in estate
    asp_bonus = {"N":0.1,"NE":0.15,"E":0.05,"SE":-0.05,"S":-0.1,"SW":-0.05,"W":0.0,"NW":0.1}
    a = 0.5 + asp_bonus.get(aspect_oct,0.0)  # 0–1
    a = clamp(a,0,1)
    # bosco: broadleaved → castagno/faggio bonus
    if "Fagus" in forest_lbl: b=1.0
    elif "Castanea" in forest_lbl: b=0.9
    elif "Quercus" in forest_lbl: b=0.7
    else: b=0.5
    # score pesato
    s = 100.0*(0.38*p + 0.28*t + 0.18*z + 0.10*b + 0.06*a)
    return s, {"p14n": round(p,2), "tn": round(t,2), "zn": round(z,2), "compat": round(b,2), "asp": round(a,2)}

def tips_dynamic(Tmean7: float, P14: float, slope: float, aspect_oct: str, forest_lbl: str, day_tag: str) -> str:
    parts = []
    if P14 < 8:
        parts.append("pioggia scarsa di recente; cerca suoli che trattengono umidità (ombreggio, con lettiera)")
    elif P14 > 30:
        parts.append("buon apporto idrico: estendi la ricerca anche ai crinali con suolo profondo")
    if Tmean7 < 8:
        parts.append("preferisci versanti esposti a NE–E con microclima più mite")
    elif Tmean7 > 20:
        parts.append("ricerca a quota leggermente maggiore e su esposizioni N–NE–NW")
    if slope < 5:
        parts.append("pianeggianti poveri di drenaggio: spostati su pendenze 5–20°")
    if "Fagus" in forest_lbl:
        parts.append("nei faggeti cerca nelle chiarie e bordi sentiero")
    if "Castanea" in forest_lbl:
        parts.append("castagneti: ottimi margini e zone muschiose")
    if "Quercus" in forest_lbl:
        parts.append("querceti: cerca nei tratti più freschi e ombrosi")
    if not parts: parts = ["nessuna criticità meteo: esplora i versanti ombreggiati e suolo profondo"]
    return f"{day_tag}: " + "; ".join(parts)

def porcini_species(forest_lbl: str, lat: float, elev_m: float, month: int) -> List[str]:
    out=[]
    # molto semplice ma coerente: in quercia/castagno più reticulatus/aereus; in faggio pinophilus
    if "Fagus" in forest_lbl:
        out.append("Boletus pinophilus")
        if month>=8: out.append("Boletus edulis")
    elif "Castanea" in forest_lbl or "Quercus" in forest_lbl:
        # Sud/centro con estate calda → aereus più probabile
        if lat < 43 and month in (6,7,8,9):
            out.append("Boletus aereus")
        out.append("Boletus reticulatus")
        if month>=9: out.append("Boletus edulis")
    else:
        out.append("Boletus edulis")
    # rimuovi duplicati
    seen=set(); res=[]
    for s in out:
        if s not in seen:
            res.append(s); seen.add(s)
    return res

def estimate_kg_3h(score: float, compat: float, uncert: float, month: int, lat: float) -> Tuple[float,str]:
    """
    stima molto prudente basata su score (0–100), compatibilità bosco e incertezza (0..1 => 0=alta certezza)
    """
    base = max(0.0, (score-40)/60.0)  # 0 a 1 sopra 40
    factor = 0.6 + 0.6*compat - 0.3*uncert
    seas = 1.0
    if month in (6,7,8) and lat<43: seas *= 1.1  # estate sud talvolta più generosa
    kg = 0.8 * base * factor * seas  # 0..~1.3 kg in 3h per singolo cercatore, prudente
    tip = "prudente"
    if kg>1.2: tip="ottimista (condizioni eccellenti)"
    elif kg>0.6: tip="realistica con buona esperienza"
    return round(max(0.0, kg),2), tip

# ——————————————————————————————————————————————————————————
# Endpoints
# ——————————————————————————————————————————————————————————
@APP.get("/api/geocode")
async def api_geocode(q: str):
    g = await geocode(q)
    return g

@APP.get("/api/score")
async def api_score(
    lat: float, lon: float, day: int = Query(0, ge=0, le=9)
):
    """
    day=0..9: 0 oggi, 1 domani, ... 9 fra 9 giorni
    """
    # elevazione + aspetto/pendenza
    elev_grid = await open_elev_grid(lat, lon)
    elev_m = float(elev_grid[1][1])
    slope_deg, aspect_deg = slope_aspect_from_grid(elev_grid, cell_m=30.0)
    aspect_oct = exposure_label(aspect_deg)

    # bosco: overpass + fallback altimetrico
    kind = await overpass_forest(lat, lon)
    forest_lbl = forest_label(kind, elev_m)

    # meteo: Open-Meteo + (facoltativo OWM)
    meteo = await meteo_openmeteo(lat, lon)
    daily = meteo["daily"]
    precip = daily["precipitation_sum"]            # len = 25 (14 passati + 11 incluso oggi)
    tempm  = daily["temperature_2m_mean"]
    rhm    = daily.get("relative_humidity_2m_mean", [None]*len(precip))
    et0    = daily.get("et0_fao_evapotranspiration", [0.0]*len(precip))

    # separa passato (ultimi 14) e futuro (oggi + 9)
    past14 = precip[:14]
    futP   = precip[14:14+10]       # oggi..D+9
    futT   = tempm[14:14+10]
    futRH  = rhm[14:14+10] if rhm else [None]*10
    futET0 = et0[14:14+10] if et0 else [0.0]*10

    P14 = sum(past14)                         # mm/14g
    Tmean7 = sum(futT[:7])/max(1,len(futT[:7]))
    # giorni dall'ultima pioggia
    lr = 0
    for x in reversed(past14):
        if x>=0.5: break
        lr += 1
    # prossima pioggia prevista (somma su prossimi 5 giorni e primo giorno >0.5mm)
    next_rain_mm = 0.0; next_rain_in = None
    for i,mm in enumerate(futP[1:], start=1):
        next_rain_mm += mm
        if next_rain_in is None and mm>=0.5: next_rain_in = i

    # score per ogni giorno (aggiorno P14 scorrendo)
    scores=[]
    rollingP = P14
    for i in range(10):
        # aggiorna pioggia con finestra mobile molto semplice: tolgo 1/14 del “magazzino” e aggiungo previsto i
        # (approssimazione: funziona per la priorità relativa tra giorni)
        add = futP[i] if i<len(futP) else 0.0
        rollingP = max(0.0, rollingP*(13.0/14.0) + add)  # emivita ~14g
        s, br = composite_score(rollingP, futT[i], elev_m, aspect_oct, forest_lbl, int(datetime.now(timezone.utc).month))
        scores.append(int(round(s)))
    s_today = scores[0]
    s_sel   = scores[day]
    s0, s2, smean = window_3day(scores)

    # incertezza semplice: se overpass non dà bosco o ET0 manca → alza
    uncert = 0.3
    if kind is None: uncert += 0.2
    if all(v is None for v in futRH): uncert += 0.1
    uncert = clamp(uncert, 0.0, 1.0)

    # consigli/“perché” dipendenti dal giorno selezionato
    day_tag = "oggi" if day==0 else f"D+{day}"
    why = (
        f"piogge ultimi 14 gg: {round(P14,1)} mm; T media 7 gg: {round(Tmean7,1)} °C; "
        f"suolo {'tendenzialmente secco' if sum(futET0[:7])>sum(futP[:7]) else 'con bilancio idrico positivo'}; "
        f"quota {int(round(elev_m))} m; esposizione {aspect_oct}; bosco: {forest_lbl}; "
        f"ultima pioggia ~{lr} gg fa"
    )
    tips = tips_dynamic(Tmean7, P14, slope_deg, aspect_oct, forest_lbl, day_tag)

    # specie probabili e stima kg/3h
    species = porcini_species(forest_lbl, lat, elev_m, int(datetime.now(timezone.utc).month))
    kg, kg_note = estimate_kg_3h(s_sel, 0.9 if ("Fagus" in forest_lbl or "Castanea" in forest_lbl) else 0.7, uncert, int(datetime.now(timezone.utc).month), lat)

    # nota finestra futura
    window_note = None
    if next_rain_in is not None and next_rain_mm>=8.0:
        window_note = f"Prevista pioggia utile (~{round(next_rain_mm,1)} mm) tra {next_rain_in} giorni: possibile apertura finestra 7–12 giorni dopo."

    return {
        "elevation_m": elev_m,
        "slope_deg": round(slope_deg,1),
        "aspect_deg": round(aspect_deg,1) if aspect_deg is not None else None,
        "aspect_octant": aspect_oct,
        "forest": forest_lbl,
        "score_today": s_today,
        "scores_next10": scores,
        "score_selected": s_sel,
        "best_window": {"start": s0, "end": s2, "mean": smean},
        "last_rain_days": lr,
        "next_rain_mm_5d": round(next_rain_mm,1),
        "next_rain_in_days": next_rain_in,
        "window_note": window_note,
        "tips": tips,
        "why": why,
        "species_probable": species,
        "kg_estimate_3h": kg,
        "kg_note": kg_note,
        "tech": {
            "P14_mm": round(P14,1),
            "Tmean7_C": round(Tmean7,1),
        }
    }



