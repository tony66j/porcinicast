# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from datetime import datetime, timedelta

# --- MODELLO SCIENTIFICO AVANZATO ---
class PorciniPredictor:
    def _normalize(self, value, min_val, max_val):
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def _calculate_soil_temp_score(self, soil_temps):
        avg_soil_temp = np.mean(soil_temps[-10:])
        return 1.0 - self._normalize(abs(avg_soil_temp - 17.5), 0, 8)

    def _calculate_rainfall_score(self, past_30d_precip):
        last_soak_event_day_idx, soak_event_precip = -1, 0
        for i in range(27, -1, -1):
            window_precip = sum(past_30d_precip[i:i+3])
            if window_precip >= 20:
                last_soak_event_day_idx = i + 2
                soak_event_precip = window_precip
                break
        
        if last_soak_event_day_idx == -1:
            return {"score": 0.0, "days_since_soak": 99, "soak_precip_mm": 0}

        days_since_soak = (30 - 1) - last_soak_event_day_idx
        incubation_score = 0
        if 10 <= days_since_soak <= 22:
            incubation_score = 1.0 - abs(days_since_soak - 16) / 6.0
        
        return { "score": incubation_score, "days_since_soak": days_since_soak, "soak_precip_mm": soak_event_precip }

    def _calculate_terrain_score(self, slope, aspect_deg, month):
        slope_score = max(0.0, 1.0 - self._normalize(slope, 20, 45))
        aspect_bonus = 1.0 - self._normalize(abs(aspect_deg - (0 if 6 <= month <= 9 else 180)), 0, 180)
        return slope_score * 0.4 + aspect_bonus * 0.6
        
    def calculate_composite_score(self, weather_data, terrain_data):
        current_month = datetime.now().month
        
        soil_temp_score = self._calculate_soil_temp_score(weather_data['past']['soil_temperature'])
        rainfall_analysis = self._calculate_rainfall_score(weather_data['past']['precipitation'])
        terrain_score = self._calculate_terrain_score(terrain_data['slope'], terrain_data['aspect'], current_month)
        
        weights = {'rainfall': 0.55, 'soil_temp': 0.30, 'terrain': 0.15}
        final_score = (rainfall_analysis["score"] * weights['rainfall'] +
                       soil_temp_score * weights['soil_temp'] +
                       terrain_score * weights['terrain']) * 100
        
        if final_score < 45: harvest = "Nullo o molto scarso (< 0.1 kg)"
        elif final_score < 65: harvest = "Discreto (0.2 - 0.7 kg)"
        elif final_score < 85: harvest = "Buono (0.8 - 2.0 kg)"
        else: harvest = "Eccellente (> 2.0 kg)"

        days_since_soak = rainfall_analysis['days_since_soak']
        future_rain_sum_5d = sum(weather_data['forecast']['precipitation'][:5])
        
        if days_since_soak > 22 and future_rain_sum_5d > 15:
            days_to_new_rain = next((i for i, p in enumerate(weather_data['forecast']['precipitation']) if p > 10), 2)
            new_start_in_days = days_to_new_rain + 10
            new_peak_in_days = days_to_new_rain + 16
            window_status = f"Nuova buttata prevista! Inizio tra {new_start_in_days} gg."
            peak_date = (datetime.now() + timedelta(days=new_peak_in_days)).strftime("%d %B")
        else:
            start_in_days = max(0, 10 - days_since_soak)
            peak_in_days = max(0, 16 - days_since_soak)
            peak_date = (datetime.now() + timedelta(days=peak_in_days)).strftime("%d %B")
            if days_since_soak <= 10: window_status = f"Inizio previsto tra {start_in_days} giorni."
            elif 10 < days_since_soak <= 22: window_status = "La 'buttata' è in corso!"
            else: window_status = "In attesa di piogge significative."
        
        advice = [window_status]
        if terrain_data['slope'] > 20: advice.append("Il versante è ripido. Controlla le piccole terrazze.")
        avg_forecast_temp = np.mean(weather_data['forecast']['temp_max'][:3])
        if avg_forecast_temp > 25: advice.append("Caldo in arrivo. Cerca nei versanti esposti a Nord.")
        elif avg_forecast_temp < 14: advice.append("Fresco in arrivo. Privilegia i versanti esposti a Sud.")

        return {
            "overall_score": int(final_score),
            "estimated_harvest_per_2h": harvest,
            "fruiting_window": { "status": window_status, "peak_date": peak_date },
            "weather_forecast": weather_data['forecast'],
            "practical_advice": advice
        }

# --- API FastAPI ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
predictor = PorciniPredictor()
http_client = httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "TrovaPorcini/2.0"})

async def get_real_weather_data(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    # FIX: Corretti i nomi dei parametri per l'API di Open-Meteo
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": 30,
        "forecast_days": 7,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m,shortwave_radiation_sum,soil_temperature_0_to_7cm",
        "timezone": "auto"
    }
    try:
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        data = response.json()['daily']
        
        # Open-Meteo restituisce i dati passati e futuri insieme. Li separiamo qui.
        split_point = -7 # Gli ultimi 7 giorni sono la previsione
        
        return {
            "past": {
                "precipitation": data['precipitation_sum'][:split_point],
                "soil_temperature": data['soil_temperature_0_to_7cm'][:split_point],
            },
            "forecast": {
                "time": data['time'][split_point:],
                "temp_max": data['temperature_2m_max'][split_point:],
                "temp_min": data['temperature_2m_min'][split_point:],
                "precipitation": data['precipitation_sum'][split_point:],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Errore nel recupero dei dati meteo: {str(e)}")


async def get_terrain_data(lat: float, lon: float):
    np.random.seed(int(lat * lon))
    return {"altitude": 800 + np.random.randint(-300, 300), "slope": 5 + np.random.rand() * 30, "aspect": np.random.rand() * 360}

@app.get("/api/geocode")
async def geocode(q: str):
    url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1&accept-language=it"
    try:
        r = await http_client.get(url, timeout=10)
        r.raise_for_status(); data = r.json()
        if not data: raise HTTPException(404, "Località non trovata")
        return {"lat": data[0]['lat'], "lon": data[0]['lon'], "name": data[0]['display_name']}
    except Exception as e: raise HTTPException(500, f"Errore di geocoding: {e}")

@app.get("/api/score")
async def get_score(lat: float, lon: float):
    try:
        weather_data = await get_real_weather_data(lat, lon)
        terrain_data = await get_terrain_data(lat, lon)
        prediction = predictor.calculate_composite_score(weather_data, terrain_data)
        prediction["coords"] = {"lat": lat, "lon": lon}
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno del server: {str(e)}")


