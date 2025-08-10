from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime, timedelta
import os

app = FastAPI()

# Abilita CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "LA_TUA_CHIAVE_API")  # <-- già impostata

def get_weather_data(lat, lon):
    """Ottiene dati storici e previsionali da OpenWeather."""
    url_forecast = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&lang=it"
    forecast_data = requests.get(url_forecast).json()

    # Piogge future raggruppate per giorno
    future_rain = {}
    for entry in forecast_data.get("list", []):
        date = entry["dt_txt"].split(" ")[0]
        rain_mm = entry.get("rain", {}).get("3h", 0)
        future_rain[date] = future_rain.get(date, 0) + rain_mm

    # Piogge passate: uso One Call Timemachine per ultimi 5 giorni (OpenWeather free limit)
    past_rain = {}
    for i in range(1, 6):
        dt_unix = int((datetime.utcnow() - timedelta(days=i)).timestamp())
        url_past = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={dt_unix}&appid={OPENWEATHER_API_KEY}&units=metric&lang=it"
        past_data = requests.get(url_past).json()
        daily_rain = 0
        for hour in past_data.get("hourly", []):
            daily_rain += hour.get("rain", {}).get("1h", 0)
        date_str = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        past_rain[date_str] = round(daily_rain, 2)

    return {
        "future_rain": future_rain,
        "past_rain": past_rain,
        "forecast_data": forecast_data
    }

def porcini_index(weather, alt):
    """Calcola indice e stima raccolto basandosi su piogge, temperatura, umidità, altitudine."""
    future_rain_total = sum(weather["future_rain"].values())
    past_rain_total = sum(weather["past_rain"].values())

    # Temperature medie previste prossimi 5 giorni
    temps = [f["main"]["temp"] for f in weather["forecast_data"]["list"][:40]]  # ~5 giorni
    avg_temp = sum(temps) / len(temps)

    # Umidità media
    hums = [f["main"]["humidity"] for f in weather["forecast_data"]["list"][:40]]
    avg_hum = sum(hums) / len(hums)

    # Logica avanzata
    index = 0
    if 12 <= avg_temp <= 18:
        index += 4
    elif 10 <= avg_temp < 12 or 18 < avg_temp <= 22:
        index += 2

    if avg_hum >= 75:
        index += 3
    elif avg_hum >= 60:
        index += 1

    if past_rain_total >= 15 and future_rain_total >= 5:
        index += 3
    elif past_rain_total >= 8:
        index += 1

    if alt >= 800:
        index += 1  # bonus altitudine montana

    # Normalizza
    index = min(index, 10)

    # Stima raccolto
    if index >= 8:
        raccolto = "6-10+ porcini"
    elif index >= 5:
        raccolto = "2-5 porcini"
    else:
        raccolto = "0-1 porcini"

    return index, raccolto, avg_temp, avg_hum, past_rain_total, future_rain_total

@app.get("/api/score")
def score(lat: float = Query(...), lon: float = Query(...), alt: float = Query(500)):
    weather = get_weather_data(lat, lon)
    index, raccolto, avg_temp, avg_hum, past_rain_total, future_rain_total = porcini_index(weather, alt)

    return {
        "indice_attuale": index,
        "stima_raccolto": raccolto,
        "piogge_passate": weather["past_rain"],
        "piogge_future": weather["future_rain"],
        "fattori": {
            "temperatura_media": round(avg_temp, 1),
            "umidita_media": round(avg_hum, 1),
            "pioggia_tot_passata_mm": past_rain_total,
            "pioggia_tot_futura_mm": future_rain_total,
            "altitudine": alt
        },
        "consigli": [
            "Cerca nei boschi misti di faggio e abete",
            "Evita le ore calde del giorno",
            "Controlla sotto felci e vicino a vecchi tronchi"
        ]
    }

@app.get("/api/health")
def health():
    return {"ok": True}






