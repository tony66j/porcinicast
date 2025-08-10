# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from datetime import datetime, timedelta

# --- MODULO DI LOGICA SCIENTIFICA ---
class PorciniPredictor:
    def _normalize(self, value, min_val, max_val):
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def _calculate_soil_temp_score(self, soil_temp_5cm):
        return 1.0 - self._normalize(abs(soil_temp_5cm - 17), 0, 8)

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

    def _calculate_terrain_score(self, slope, aspect_deg):
        slope_score = max(0.0, 1.0 - self._normalize(slope, 20, 45))
        month = datetime.now().month
        aspect_bonus = 1.0 - self._normalize(abs(aspect_deg - (0 if 6 <= month <= 9 else 180)), 0, 180)
        return slope_score * 0.4 + aspect_bonus * 0.6
        
    def calculate_composite_score(self, weather_data, terrain_data):
        soil_temp_score = self._calculate_soil_temp_score(np.mean(weather_data['soil_temperature_5cm'][-10:]))
        rainfall_analysis = self._calculate_rainfall_score(weather_data['precipitation'])
        terrain_score = self._calculate_terrain_score(terrain_data['slope'], terrain_data['aspect'])
        
        weights = {'rainfall': 0.55, 'soil_temp': 0.30, 'terrain': 0.15}
        final_score = (rainfall_analysis["score"] * weights['rainfall'] +
                       soil_temp_score * weights['soil_temp'] +
                       terrain_score * weights['terrain']) * 100
        
        # Stima Raccolta (invariata)
        if final_score < 45: harvest = "Nullo o molto scarso (< 0.1 kg)"
        elif final_score < 65: harvest = "Discreto (0.2 - 0.7 kg)"
        elif final_score < 85: harvest = "Buono (0.8 - 2.0 kg)"
        else: harvest = "Eccellente (> 2.0 kg)"

        # Finestra di Uscita (invariata)
        days_since_soak = rainfall_analysis['days_since_soak']
        start_in_days = max(0, 10 - days_since_soak)
        peak_in_days = max(0, 16 - days_since_soak)
        peak_date = (datetime.now() + timedelta(days=peak_in_days)).strftime("%d %B")

        # MODELLO PREDITTIVO MIGLIORATO CON PIOGGE FUTURE
        # Controlla se una nuova "buttata" inizierà a causa delle piogge future
        future_rain_sum_5d = sum(weather_data['precipitation_forecast'][:5])
        if days_since_soak > 22 and future_rain_sum_5d > 15:
            # Una nuova pioggia significativa è prevista!
            days_to_new_rain = next((i for i, p in enumerate(weather_data['precipitation_forecast']) if p > 10), 2)
            new_start_in_days = days_to_new_rain + 10
            new_peak_in_days = days_to_new_rain + 16
            
            window_status = f"Nuova buttata prevista! Inizio tra {new_start_in_days} gg."
            peak_date = (datetime.now() + timedelta(days=new_peak_in_days)).strftime("%d %B")
        else:
            # Logica precedente se non ci sono nuove piogge previste
            if days_since_soak <= 10: window_status = f"Inizio previsto tra {start_in_days} giorni."
            elif 10 < days_since_soak <= 22: window_status = "La 'buttata' è in corso!"
            else: window_status = "In attesa di piogge significative."
        
        # Consigli Dinamici
        advice = [window_status]
        if terrain_data['slope'] > 20: advice.append("Il versante è ripido. Controlla le piccole terrazze.")
        current_temp = np.mean(weather_data['temperature_2m'][-3:])
        if current_temp > 24: advice.append("Fa caldo. Cerca nei versanti esposti a Nord.")
        elif current_temp < 12: advice.append("Fa fresco. Privilegia i versanti esposti a Sud.")

        return {
            "overall_score": int(final_score),
            "estimated_harvest_per_2h": harvest,
            "fruiting_window": {
                "status": window_status,
                "start_in_days": start_in_days,
                "peak_date": peak_date
            },
            "specifications": {
                "days_since_last_soak": days_since_soak,
                "last_soak_precipitation_mm": round(rainfall_analysis['soak_precip_mm']),
                "future_5d_weather": weather_data['precipitation_forecast'][:5], # Invia tutti i 5 giorni
                "altitude_m": int(terrain_data['altitude']),
                "slope_deg": int(terrain_data['slope']),
                "aspect_deg": int(terrain_data['aspect']),
            },
            "practical_advice": advice,
            "future_5d_temps": weather_data['temperature_forecast'][:5]
        }

# --- API FastAPI ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
predictor = PorciniPredictor()
http_client = httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "TrovaPorcini/1.1"})

async def get_external_data(lat, lon):
    np.random.seed(int(lat + lon))
    return {
        "precipitation": np.random.gamma(1.5, 3.0, 30).tolist(),
        "soil_temperature_5cm": (18 + np.random.randn(30) * 3).tolist(),
        "temperature_2m": (20 + np.random.randn(30) * 4).tolist(),
        "precipitation_forecast": np.random.gamma(0.8, 4.0, 10).tolist(), # Aumentata la variabilità per test
        "temperature_forecast": (19 + np.random.randn(10) * 4).tolist(),
    }
async def get_terrain_data(lat, lon):
    np.random.seed(int(lat * lon))
    return {"altitude": 800 + np.random.randint(-300, 300), "slope": 5 + np.random.rand() * 30, "aspect": np.random.rand() * 360}

@app.get("/api/geocode")
async def geocode(q: str):
    url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1&accept-language=it"
    try:
        r = await http_client.get(url, timeout=10)
        r.raise_for_status()
        data = r.json();
        if not data: raise HTTPException(404, "Località non trovata")
        return {"lat": data[0]['lat'], "lon": data[0]['lon'], "name": data[0]['display_name']}
    except Exception as e: raise HTTPException(500, f"Errore di geocoding: {e}")

@app.get("/api/score")
async def get_score(lat: float, lon: float):
    try:
        weather_data = await get_external_data(lat, lon)
        terrain_data = await get_terrain_data(lat, lon)
        prediction = predictor.calculate_composite_score(weather_data, terrain_data)
        prediction["coords"] = {"lat": lat, "lon": lon}
        return prediction
    except Exception as e: raise HTTPException(500, f"Errore nel calcolo: {e}")


