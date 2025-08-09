from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import math, os, asyncio
import httpx

APP = FastAPI(title="TrovaPorcini v2.0 API")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_headers=["*"], allow_methods=["*"], allow_credentials=True,
)

OWM_KEY = os.getenv("OPENWEATHER_API_KEY")  # <-- assicurati esista su Render
HEAD = {"User-Agent": "TrovaPorcini-2.0 (+https://example.org)"}

# ----------------------------
# Utilità
# ----------------------------
def clamp(x, a, b): return max(a, min(b, x))

def days_since_last_rain(daily_mm: List[float]) -> Optional[int]:
    # daily_mm[0] = oggi, 1 = ieri, ...
    for d, mm in enumerate(daily_mm):
        if (mm or 0.0) > 0.2:
            return d
    return None

def next_rain(daily_mm: List[float]) -> Optional[Tuple[int, float]]:
    # ritorna (giorni_da_oggi, totale_mm) della prossima giornata piovosa
    for d, mm in enumerate(daily_mm):
        if (mm or 0.0) >= 1.0:
            return d, float(mm or 0.0)
    return None

def hargreaves_et0(tmin, tmax, tmean, lat_deg: float) -> float:
    # stima ET0 giornaliera (mm) semplificata (serve trend idrico)
    t_range = max(0.0, (tmax - tmin))
    phi = math.radians(lat_deg)
    # radiazione extraterrestre semplificata ~ funzione della latitudine
    r_a = 25 + 10 * abs(math.sin(phi))  # MJ/m2/d (grezza ma sufficiente per pesi relativi)
    et0 = 0.0023 * (tmean + 17.8) * math.sqrt(t_range) * (r_a / 2.45)
    return clamp(et0, 0.0, 12.0)

def forest_label_from_osm_kind(kind: Optional[str], alt_m: float) -> str:
    # fallback ragionato quando Overpass non risponde
    if kind == "coniferous": return "Pinus/Abies/Picea"
    if kind == "broadleaved":
        if alt_m > 900: return "Fagus sylvatica"
        if 500 < alt_m <= 900: return "Castanea sativa"
        return "Quercus spp."
    # solo altitudine
    if alt_m > 1400: return "Pinus/Abies/Picea"
    if alt_m > 900:  return "Fagus sylvatica"
    if alt_m > 500:  return "Castanea sativa"
    return "Quercus spp."

def porcini_species(lat: float, alt_m: float, forest: str) -> List[str]:
    S = []
    # areus più termofilo e mediterraneo, reticulatus/castagna collinare, edulis/pinicola più freschi
    if "Castanea" in forest or "Quercus" in forest:
        if lat < 43.5 and 300 <= alt_m <= 1100:
            S.append("Boletus aereus")
        S.append("Boletus reticulatus")
    if "Fagus" in forest or alt_m >= 900:
        S.append("Boletus edulis")
    if "Pinus" in forest or "Abies" in forest or "Picea" in forest:
        S.append("Boletus pinophilus")
    # unica volta ciascuna
    seen, out = set(), []
    for x in S:
        if x not in seen:
            out.append(x); seen.add(x)
    return out or ["Boletus edulis"]

def aspect_from_grid(grid3x3: List[List[float]], cell_m: float = 30.0) -> Tuple[float, float]:
    # slope & aspect da una 3x3 di quota (se disponibile)
    # qui lo teniamo come placeholder -> ritorniamo "N" (0°) e 10° di pendenza se non fornita
    return 10.0, 0.0

def aspect_octant(deg: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    i = int((deg % 360)/45.0 + 0.5) % 8
    return dirs[i]

# ----------------------------
# Chiamate ai provider
# ----------------------------
async def geocode(q: str) -> Dict[str, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format":"json", "limit":1, "addressdetails":1}
    async with httpx.AsyncClient(timeout=20, headers=HEAD) as cli:
        r = await cli.get(url, params=params); r.raise_for_status()
        j = r.json()
    if not j: raise httpx.HTTPError("Località non trovata")
    return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "display": j[0].get("display_name","")}

async def openweather(lat: float, lon: float) -> Dict[str, Any]:
    if not OWM_KEY: raise httpx.HTTPError("Manca OPENWEATHER_API_KEY")
    # One Call 3.0: daily 16 giorni; usiamo daily 14 passati approx con 'timemachine' emulando?
    # Qui: prendiamo 'daily' per 8-10 gg futuri + 'hourly' per prossime 48h (per prox piogge)
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat, "lon": lon, "appid": OWM_KEY,
        "units": "metric", "exclude": "minutely,alerts"
    }
    async with httpx.AsyncClient(timeout=25, headers=HEAD) as cli:
        r = await cli.get(url, params=params); r.raise_for_status()
        j = r.json()
    return j

async def overpass_forest(lat: float, lon: float) -> Optional[str]:
    # inferenza broadleaved/coniferous
    query = f"""
    [out:json][timeout:25];
    (
      way(around:800,{lat},{lon})[natural=wood];
      relation(around:800,{lat},{lon})[natural=wood];
    );out tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    async with httpx.AsyncClient(timeout=30, headers=HEAD) as cli:
        r = await cli.post(url, data={"data": query}); r.raise_for_status()
        j = r.json()
    labels = []
    for el in j.get("elements", []):
        tags = el.get("tags", {})
        if "leaf_type" in tags:
            lt = tags["leaf_type"].lower()
            if "broad" in lt: labels.append("broadleaved")
            elif "conifer" in lt: labels.append("coniferous")
        elif "wood" in tags:
            w = tags["wood"]
            if w in ("deciduous","broadleaved"): labels.append("broadleaved")
            if w in ("coniferous","needleleaved"): labels.append("coniferous")
    if labels:
        b = labels.count("broadleaved")
        c = labels.count("coniferous")
        return "broadleaved" if b >= c else "coniferous"
    return None

# ----------------------------
# Scoring
# ----------------------------
def seasonal_alt_window(lat: float) -> Tuple[int,int]:
    # quota ottimale media che si sposta con la latitudine
    # centro ottimo ~ 1100 m al Nord, ~ 900 m Centro, ~ 700 m Sud
    if lat > 44.5: base = 1100
    elif lat > 41.5: base = 900
    else: base = 700
    return base-450, base+350

def composite_score(
    p14_mm: float, tmean7: float, et0_7: float, alt_m: float,
    aspect_deg: float, forest: str, lat: float, rain_next10: List[float]
) -> Tuple[float, Dict[str,float], str, str]:
    # Idrico: bilancio P-ET0 ultimi 7 gg + accumulo 14 gg
    water = clamp( 0.6*clamp(p14_mm/40.0, 0, 1) + 0.4*clamp((p14_mm - 0.8*et0_7*7)/40.0, -0.5, 1), 0, 1)

    # Termico: ottimo 14–18 °C (capace), decresce <10 o >22
    def bell_T(t):
        if t<=6 or t>=26: return 0
        if 14<=t<=18: return 1
        if t<14: return (t-6)/(14-6)
        return (26-t)/(26-18)
    thermo = clamp(bell_T(tmean7), 0, 1)

    # Quota stagionale
    a_lo,a_hi = seasonal_alt_window(lat)
    if alt_m<=a_lo-300 or alt_m>=a_hi+300: altfit=0
    elif a_lo<=alt_m<=a_hi: altfit=1
    else:
        # transizioni morbide
        if alt_m<a_lo: altfit=(alt_m-(a_lo-300))/300
        else:         altfit=((a_hi+300)-alt_m)/300
        altfit = clamp(altfit, 0, 1)

    # Aspetto (NE–NW preferito nei mesi caldi); 0=Nord…
    asp = aspect_deg % 360
    # distanza dall’arco 300..60 (NW..NE)
    def arc_score(d):
        # più vicino a N (0°) meglio in estate; qui diamo 1 a 330..30, scende verso S
        d = min(abs((d-0+180)%360-180), abs((d-360+180)%360-180))
        return clamp(1 - d/180.0, 0, 1)
    aspect_s = arc_score(asp)

    # Bosco
    compat = 0.7 if ("Castanea" in forest or "Quercus" in forest) else \
             0.8 if ("Fagus" in forest) else \
             0.6 if ("Pinus" in forest or "Abies" in forest or "Picea" in forest) else 0.5

    # Piogge future: finestre potenziali (media prossimi 3 gg)
    fut3 = sum(rain_next10[:3])
    future = clamp(fut3/25.0, 0, 1)

    # Score finale (0..100)
    s = (0.34*water + 0.26*thermo + 0.18*altfit + 0.10*aspect_s + 0.12*compat) * 100.0
    s = clamp(s, 0, 100)

    breakdown = {
        "water": round(water,3), "thermo": round(thermo,3),
        "alt": round(altfit,3), "aspect": round(aspect_s,3),
        "compat": round(compat,3), "future": round(future,3)
    }

    why = []
    if water<0.35:  why.append("suolo tendenzialmente secco (P–ET₀ basso)")
    if thermo<0.35: why.append("temperatura media poco favorevole")
    if altfit<0.4:  why.append("quota non ottimale per la stagione/latitudine")
    if aspect_s<0.4: why.append("versanti poco ombreggiati (S–SE–SW)")
    if compat<0.6:  why.append("bosco poco compatibile")
    why_text = "; ".join(why) if why else "condizioni complessivamente favorevoli"

    advice = []
    if water<0.45: advice.append("cerca suoli profondi/ombrosi e impluvi che trattengono umidità")
    if thermo>0.6: advice.append("esposizioni N–NE–NW e coperture folte nelle ore calde")
    if altfit<0.6:
        advice.append("valuta quota più " + ("alta" if alt_m<a_lo else "bassa"))
    if "Castanea" in forest: advice.append("castagneti maturi spesso produttivi dopo piogge di 20–30 mm")
    if "Fagus" in forest: advice.append("faggete buone con T media 12–18 °C e suolo fresco")
    if "Pinus" in forest: advice.append("in conifere cerca margini e radure con lettiera umida")
    if fut3>=12: advice.append("piogge in arrivo → finestra probabile 7–12 giorni dopo")
    advice_text = "; ".join(advice) if advice else "esplora versanti ombrosi con suolo fresco"

    return s, breakdown, why_text, advice_text

def caps_estimate(score: float, compat: float, uncert: float) -> Tuple[int, Tuple[int,int]]:
    # stima cappelli/3h: base sullo score, modulata da compatibilità e incertezza
    # 0..20: 0–1; 20..40: 1–3; 40..60: 3–8; 60..80: 8–18; 80..100: 18–35
    if score<20: base=(0,1)
    elif score<40: base=(1,3)
    elif score<60: base=(3,8)
    elif score<80: base=(8,18)
    else: base=(18,35)
    m = 0.6 + 0.6*compat - 0.3*uncert  # compat ↑ ⇒ più; incertezza ↑ ⇒ meno
    lo = max(0, int(round(base[0]*m)))
    hi = max(lo+1, int(round(base[1]*m)))
    mean = int(round(0.6*lo+0.4*hi))
    return mean, (lo,hi)

# ----------------------------
# Endpoint
# ----------------------------
@APP.get("/api/geocode")
async def api_geocode(q: str): return await geocode(q)

@APP.get("/api/score")
async def api_score(
    lat: float = Query(...), lon: float = Query(...), day: int = Query(0, ge=0, le=9),
    alt_m: Optional[float] = None
):
    # Meteo
    j = await openweather(lat, lon)
    daily = j.get("daily", [])
    if not daily: raise httpx.HTTPError("Dati meteo non disponibili")

    # serie piogge/temperature
    rain_fut = [float(d.get("rain", 0.0)) for d in daily[:10]]
    tmean = [ (d["temp"]["min"]+d["temp"]["max"])/2.0 for d in daily[:10] ]
    t7 = sum(tmean[max(0,day-6):day+1]) / max(1, (day - max(0,day-6) + 1))
    et0_7 = sum(hargreaves_et0(d["temp"]["min"], d["temp"]["max"], (d["temp"]["min"]+d["temp"]["max"])/2.0, lat) for d in daily[max(0,day-6):day+1])

    # Usiamo 14 gg cumulati stimando P14 da daily futuri + hourly (se servisse) — qui: approccio coerente
    # accumulo ultimi 14 usando daily passati non sempre presenti: stima retro con decadimento (= più recente pesa di più)
    # per robustezza usiamo piogge dei primi max 5 giorni *decaduti* come proxy degli scorsi 14 gg.
    w = [0.35,0.25,0.18,0.12,0.10]  # pesi
    p14_proxy = sum( w[i]*rain_fut[i] for i in range(min(5,len(rain_fut))) ) * 2.2  # fattore per avvicinare 14 gg

    # altitudine
    elev = alt_m if alt_m is not None else j.get("elevation", 800.0) if isinstance(j.get("elevation"), (int,float)) else 800.0

    # bosco: OSM con fallback
    try:
        kind = await overpass_forest(lat, lon)
    except Exception:
        kind = None
    forest = forest_label_from_osm_kind(kind, elev)

    # aspect/pendenza (placeholder robusto)
    slope_deg, aspect_deg = aspect_from_grid([[elev]*3]*3, 30.0)

    score, breakdown, why, advice = composite_score(
        p14_proxy, t7, et0_7, elev, aspect_deg, forest, lat, rain_fut[day:]
    )

    # incertezza semplice: + alta se p14_proxy stimato molto basso o forest derivato per fallback
    uncert = 0.15
    if p14_proxy < 6: uncert += 0.1
    if kind is None: uncert += 0.06
    uncert = clamp(uncert, 0.05, 0.35)

    mean_caps, rng = caps_estimate(score, breakdown["compat"], uncert)

    # ultima e prossima pioggia (sulle serie disponibili)
    last = days_since_last_rain([rain_fut[0]] + [0,0,0,0])  # proxy minimale su futuri: oggi/non oggi
    nxt = next_rain(rain_fut[1:])  # da domani
    ntext = None
    if nxt:
        dd, mm = nxt
        if mm >= 8:
            ntext = f"Possibile apertura finestra tra ~7–12 gg se cadono ~{mm:.0f} mm (D+{dd+1})."

    return {
        "lat": lat, "lon": lon, "alt_m": round(elev,1),
        "forest": forest, "slope_deg": slope_deg, "aspect_deg": aspect_deg,
        "aspect_octant": aspect_octant(aspect_deg),
        "score_today": int(round(score)) if day==0 else None,
        "score": int(round(score)), "day": day,
        "breakdown": breakdown,
        "why": why, "advice": advice,
        "caps_3h": {"mean": mean_caps, "range": rng, "uncertainty": round(uncert,2)},
        "p14_proxy_mm": round(p14_proxy,1), "tmean7_c": round(t7,1), "et0_7mm": round(et0_7,1),
        "rain_next10_mm": rain_fut, "note_next_window": ntext,
    }

@APP.get("/api/heatmap")
async def api_heatmap(day: int = Query(0, ge=0, le=9)):
    # Griglia leggera su Italia
    bbox = (36.5, 6.5, 47.2, 18.8)  # lat_min, lon_min, lat_max, lon_max
    step = 0.7
    lats = [bbox[0] + i*step for i in range(int((bbox[2]-bbox[0])/step)+1)]
    lons = [bbox[1] + j*step for j in range(int((bbox[3]-bbox[1])/step)+1)]
    pts = []
    # Parallelizza debolmente
    async def one(lat, lon):
        try:
            j = await openweather(lat, lon)
            daily = j.get("daily", [])
            if not daily: return
            rain_fut = [float(d.get("rain", 0.0)) for d in daily[:10]]
            tmean = [ (d["temp"]["min"]+d["temp"]["max"])/2.0 for d in daily[:10] ]
            t7 = tmean[min(day, len(tmean)-1)]
            et0_7 = hargreaves_et0(daily[day]["temp"]["min"], daily[day]["temp"]["max"], t7, lat)
            p14_proxy = 2.2 * sum([0.35,0.25,0.18,0.12,0.10][:min(5,len(rain_fut))])
            elev = j.get("elevation", 800.0) if isinstance(j.get("elevation"), (int,float)) else 800.0
            forest = forest_label_from_osm_kind(None, elev)
            _, aspect_deg = 10.0, 0.0
            s, _, _, _ = composite_score(p14_proxy, t7, et0_7, elev, aspect_deg, forest, lat, rain_fut[day:])
            pts.append({"lat": lat, "lon": lon, "s": int(round(s))})
        except Exception:
            pass

    tasks = [one(la, lo) for la in lats for lo in lons]
    await asyncio.gather(*tasks)
    return {"day": day, "points": pts}



