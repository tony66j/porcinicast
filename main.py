from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime, timedelta, timezone
import math
import httpx
from typing import Dict, Any, Tuple

app = FastAPI(title="Trovapolcini API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

OWM_KEY = os.getenv("OPENWEATHER_API_KEY")  # deve essere già impostata su Render

# ------------------ helpers ------------------
def iso_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def clamp(x, a, b): 
    return max(a, min(b, x))

# ------------------ fetchers (httpx) ------------------
async def fetch_openweather_forecast(client: httpx.AsyncClient, lat: float, lon: float) -> Dict[str, Any]:
    """5 giorni/3h forecast."""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric", "lang": "it"}
    r = await client.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

async def fetch_openweather_past_day(client: httpx.AsyncClient, lat: float, lon: float, days_ago: int) -> Tuple[str, float]:
    """
    Prova One Call timemachine (potrebbe non essere abilitato su alcuni piani).
    Se fallisce, ritorna (data, -1.0) e lasceremo al fallback Open-Meteo.
    """
    target = datetime.utcnow() - timedelta(days=days_ago)
    dt_unix = int(target.timestamp())
    url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    params = {"lat": lat, "lon": lon, "dt": dt_unix, "appid": OWM_KEY, "units": "metric"}
    try:
        r = await client.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return (iso_date(target), -1.0)
        j = r.json()
        daily_rain = 0.0
        for hour in j.get("hourly", []):
            daily_rain += float(hour.get("rain", {}).get("1h", 0.0))
        return (iso_date(target), round(daily_rain, 2))
    except Exception:
        return (iso_date(target), -1.0)

async def fetch_openmeteo_past(client: httpx.AsyncClient, lat: float, lon: float, days: int = 10) -> Dict[str, float]:
    """
    Fallback gratuito: pioggia giornaliera degli ultimi 'days' da Open-Meteo.
    Ritorna { 'YYYY-MM-DD': mm }.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": "UTC",
        "daily": "precipitation_sum", "past_days": days, "forecast_days": 1
    }
    r = await client.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    out = {}
    d = j.get("daily", {})
    for t, mm in zip(d.get("time", []), d.get("precipitation_sum", [])):
        try:
            out[str(t)] = round(float(mm or 0.0), 2)
        except Exception:
            pass
    # togli il giorno "oggi" (forecast_days=1 lo include)
    today = iso_date(datetime.utcnow())
    if today in out:
        out.pop(today, None)
    return out

# ------------------ logica indice & stima ------------------
def porcini_index_and_yield(
    future_rain: Dict[str, float],
    past_rain: Dict[str, float],
    avg_temp_5d: float,
    avg_hum_5d: float,
    altitude_m: float
) -> Tuple[int, str]:
    """
    Indice 0–10 e stima racc. 2h.
    Logica:
     - finestra termica ottimale 12–18 °C (bonus), 10–12 / 18–22 ok (minore bonus)
     - umidità media 5gg: >75% bonus, 60–75% piccolo bonus
     - piogge passate: soglia utilità ~15 mm; future: >5 mm indica trigger imminente
     - altitudine: +1 sopra 800 m in estate (semplificazione)
    """
    idx = 0

    # temperatura
    if 12 <= avg_temp_5d <= 18:
        idx += 4
    elif 10 <= avg_temp_5d < 12 or 18 < avg_temp_5d <= 22:
        idx += 2
    # umidità
    if avg_hum_5d >= 75:
        idx += 3
    elif avg_hum_5d >= 60:
        idx += 1

    past_total = sum(past_rain.values())
    fut_total = sum(future_rain.values())

    # acqua disponibile/trigger
    if past_total >= 15 and fut_total >= 5:
        idx += 3
    elif past_total >= 8:
        idx += 1

    if altitude_m >= 800:
        idx += 1

    idx = clamp(idx, 0, 10)

    # stima raccolto (molto indicativa)
    if idx >= 8:
        harvest = "6–10+ porcini"
    elif idx >= 5:
        harvest = "2–5 porcini"
    else:
        harvest = "0–1 porcini"

    return int(idx), harvest

# ------------------ API ------------------
@app.get("/api/health")
async def health():
    return {"ok": True, "app": "Trovapolcini", "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/score")
async def score(
    lat: float = Query(...),
    lon: float = Query(...),
    alt: float = Query(500.0)
):
    if not OWM_KEY:
        raise HTTPException(500, detail="OPENWEATHER_API_KEY mancante")

    async with httpx.AsyncClient() as client:
        # preleva forecast (5 giorni a step 3h) da OpenWeather
        fc = await fetch_openweather_forecast(client, lat, lon)

        # piogge future aggregate per giorno
        future_rain: Dict[str, float] = {}
        for entry in fc.get("list", []):
            date = entry.get("dt_txt", "").split(" ")[0]
            mm = float(entry.get("rain", {}).get("3h", 0.0))
            future_rain[date] = future_rain.get(date, 0.0) + mm

        # medie 5 giorni (temperatura e umidità)
        items = fc.get("list", [])[:40]  # ~ 5 giorni
        temps = [float(x["main"]["temp"]) for x in items if "main" in x]
        hums  = [float(x["main"]["humidity"]) for x in items if "main" in x]
        avg_temp_5d = sum(temps)/len(temps) if temps else 0.0
        avg_hum_5d  = sum(hums)/len(hums) if hums else 0.0

        # piogge passate: prova OneCall timemachine (ultimi 5 giorni)
        past_rain: Dict[str, float] = {}
        use_fallback = False
        for i in range(1, 6):
            d, mm = await fetch_openweather_past_day(client, lat, lon, i)
            if mm < 0:
                use_fallback = True
                break
            past_rain[d] = mm

        # fallback gratuito con Open-Meteo (ultimi 10 giorni)
        if use_fallback:
            past_rain = await fetch_openmeteo_past(client, lat, lon, days=10)

        # indice + stima
        idx, harvest = porcini_index_and_yield(
            future_rain=future_rain,
            past_rain=past_rain,
            avg_temp_5d=avg_temp_5d,
            avg_hum_5d=avg_hum_5d,
            altitude_m=alt
        )

        # consigli semplici
        tips = []
        if idx >= 8:
            tips.append("Finestra molto favorevole: controlla versanti N/NE e lettiere profonde in faggete/castagneti.")
        elif idx >= 5:
            tips.append("Condizioni discrete: privilegia conche ombreggiate e microdepressioni dopo i rovesci.")
        else:
            tips.append("Poco favorevole: attendi nuovi episodi di pioggia >10–15 mm e calo termico.")

        if avg_hum_5d < 55:
            tips.append("Umidità prevista bassa: ricerca nelle valli riparie e vicino a sorgenti.")
        if sum(future_rain.values()) >= 8:
            tips.append("Con piogge in arrivo, la finestra migliore potrebbe aprirsi 2–5 giorni dopo l’evento.")

        return {
            "indice_attuale": idx,
            "stima_raccolto": harvest,
            "piogge_passate": past_rain,   # {"YYYY-MM-DD": mm}
            "piogge_future": future_rain,  # {"YYYY-MM-DD": mm}
            "fattori": {
                "temperatura_media": round(avg_temp_5d, 1),
                "umidita_media": round(avg_hum_5d, 1),
                "pioggia_tot_passata_mm": round(sum(past_rain.values()), 1),
                "pioggia_tot_futura_mm": round(sum(future_rain.values()), 1),
                "altitudine": round(alt)
            },
            "consigli": tips
        }






