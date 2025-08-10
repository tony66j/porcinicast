# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from datetime import datetime, timedelta

# --- MODULO DI LOGICA SCIENTIFICA (Normalmente in utils.py) ---
# Per semplicità, è integrato qui.

class PorciniPredictor:
    """
    Modello avanzato per la previsione della fruttificazione di Boletus edulis.
    Basato su principi ecologici: shock termico/idrico, temperatura del suolo e habitat.
    """
    def _normalize(self, value, min_val, max_val):
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def _calculate_soil_temp_score(self, soil_temp_5cm):
        """Punteggio basato sulla temperatura del suolo. L'ottimo per il micelio è 15-19°C."""
        return 1.0 - self._normalize(abs(soil_temp_5cm - 17), 0, 8) # Tolleranza di 8°C dall'ottimo (17°C)

    def _calculate_rainfall_score(self, past_30d_precip, past_30d_radiation):
        """
        Analizza la pioggia degli ultimi 30 giorni.
        Cerca un evento di "reset" idrico (pioggia significativa) seguito da un periodo di incubazione.
        """
        # Trova il giorno dell'ultima pioggia significativa (>15mm in 2gg)
        last_soak_event_day = -1
        cumulative_precip = 0
        for i in range(29, 0, -1):
            if past_30d_precip[i] + past_30d_precip[i-1] > 15:
                last_soak_event_day = i
                break
        
        if last_soak_event_day == -1:
            return {"score": 0.0, "days_since_soak": 99, "total_precip_since_soak": 0}

        days_since_soak = 30 - last_soak_event_day
        
        # Periodo di incubazione ideale: 10-22 giorni
        incubation_score = 0
        if 10 <= days_since_soak <= 22:
            incubation_score = 1.0 - abs(days_since_soak - 16) / 6.0 # Ottimo a 16 giorni
        
        # Il suolo deve rimanere umido, ma non saturo. L'irraggiamento lo asciuga.
        precip_since_soak = sum(past_30d_precip[last_soak_event_day:])
        radiation_since_soak = sum(past_30d_radiation[last_soak_event_day:])
        
        moisture_retention_score = self._normalize(precip_since_soak / (radiation_since_soak * 0.05 + 1), 0.5, 3.0)

        return {
            "score": incubation_score * moisture_retention_score,
            "days_since_soak": days_since_soak,
            "total_precip_since_soak": precip_since_soak
        }

    def _calculate_terrain_score(self, slope, aspect_deg):
        """Valuta pendenza ed esposizione."""
        slope_score = max(0.0, 1.0 - self._normalize(slope, 20, 45)) # Penalizza pendenze > 20°
        
        # Bonus per esposizioni fresche (Nord) in estate, più soleggiate (Sud) in autunno
        month = datetime.now().month
        if 6 <= month <= 9: # Estate
            aspect_score = 1.0 - self._normalize(abs(aspect_deg - 0), 0, 180) # Ottimo a Nord (0/360°)
        else: # Autunno/Primavera
            aspect_score = 1.0 - self._normalize(abs(aspect_deg - 180), 0, 180) # Ottimo a Sud (180°)
            
        return slope_score * 0.4 + aspect_score * 0.6
        
    def calculate_composite_score(self, weather_data, terrain_data):
        """Calcola il punteggio finale e genera i dati di output."""
        
        # 1. Calcolo Punteggi Parziali
        soil_temp_score = self._calculate_soil_temp_score(np.mean(weather_data['soil_temperature_5cm'][-10:]))
        rainfall_analysis = self._calculate_rainfall_score(weather_data['precipitation'], weather_data['shortwave_radiation'])
        rainfall_score = rainfall_analysis["score"]
        terrain_score = self._calculate_terrain_score(terrain_data['slope'], terrain_data['aspect'])
        
        # 2. Pesi dei Fattori
        # La condizione idrica (pioggia e incubazione) è il fattore scatenante principale
        weights = {'rainfall': 0.50, 'soil_temp': 0.35, 'terrain': 0.15}
        
        final_score = (rainfall_score * weights['rainfall'] +
                       soil_temp_score * weights['soil_temp'] +
                       terrain_score * weights['terrain']) * 100
        
        # 3. Genera Previsione a 10 giorni (semplificata)
        forecast = [final_score]
        for i in range(1, 10):
            # Simula una variazione basata sul meteo futuro
            temp_factor = 1 - abs(weather_data['temperature_2m_forecast'][i] - 18) / 20 # Decadimento se troppo caldo/freddo
            rain_factor = 1 + weather_data['precipitation_forecast'][i] / 50 # Leggero bonus se piove
            next_day_score = forecast[-1] * 0.95 * temp_factor * rain_factor # 0.95 = decadimento naturale
            forecast.append(max(0, min(100, next_day_score)))
        
        # 4. Stima Raccolta
        if final_score < 45: harvest = "Nullo o molto scarso (< 0.1 kg)"
        elif final_score < 65: harvest = "Discreto (0.2 - 0.7 kg)"
        elif final_score < 85: harvest = "Buono (0.8 - 2.0 kg)"
        else: harvest = "Eccellente (> 2.0 kg)"

        # 5. Consigli Dinamici
        advice = []
        days_since_rain = rainfall_analysis['days_since_soak']
        if days_since_rain < 10: advice.append("Pioggia scatenante recente. La 'buttata' è imminente, ma il bosco è ancora bagnato. Attendi qualche giorno.")
        elif 10 <= days_since_rain <= 18: advice.append(f"PERIODO IDEALE! La 'buttata' dovrebbe essere in corso. Concentra la ricerca nelle zone con le caratteristiche di habitat migliori.")
        elif days_since_rain > 22: advice.append("È passato troppo tempo dall'ultima pioggia significativa. Cerca solo in zone molto umide o in ombra per trovare gli ultimi esemplari.")
        
        if terrain_data['slope'] > 20: advice.append("Il versante è ripido. Controlla le piccole terrazze e i punti dove si accumula il terriccio.")
        
        current_temp = np.mean(weather_data['temperature_2m'][-3:])
        if current_temp > 24: advice.append("Fa caldo. Cerca nei versanti esposti a Nord, più freschi e ombreggiati.")
        elif current_temp < 12: advice.append("Fa fresco. Privilegia i versanti esposti a Sud che ricevono più sole e calore durante il giorno.")

        return {
            "overall_score": int(final_score),
            "estimated_harvest_per_2h": harvest,
            "forecast_10_days": [int(s) for s in forecast],
            "past_30d_rain_mm": weather_data['precipitation'],
            "future_10d_weather": {
                "temperature": weather_data['temperature_2m_forecast'],
                "precipitation": weather_data['precipitation_forecast'],
            },
            "practical_advice": advice
        }

# --- API FastAPI ---

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = PorciniPredictor()
http_client = httpx.AsyncClient(timeout=20.0)

async def get_external_data(lat, lon):
    """Simula la chiamata a un servizio meteo avanzato (es. Open-Meteo)."""
    # In una app reale, qui faresti la chiamata HTTP
    # Per ora, generiamo dati realistici ma casuali.
    np.random.seed(int(lat + lon)) # Rende i dati "stabili" per la stessa località
    return {
        "precipitation": np.random.gamma(1.5, 3.0, 30).tolist(),
        "soil_temperature_5cm": (18 + np.random.randn(30) * 3).tolist(),
        "shortwave_radiation": (150 + np.random.rand(30) * 100).tolist(),
        "temperature_2m": (20 + np.random.randn(30) * 4).tolist(),
        "precipitation_forecast": np.random.gamma(0.8, 2.0, 10).tolist(),
        "temperature_2m_forecast": (19 + np.random.randn(10) * 3).tolist(),
    }

async def get_terrain_data(lat, lon):
    """Simula la chiamata a un servizio di elevazione."""
    np.random.seed(int(lat * lon))
    return {
        "altitude": 800 + np.random.randint(-300, 300),
        "slope": 5 + np.random.rand() * 30,
        "aspect": np.random.rand() * 360
    }

@app.get("/api/geocode")
async def geocode(q: str):
    url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1&accept-language=it"
    try:
        r = await http_client.get(url)
        r.raise_for_status()
        data = r.json()
        if not data:
            raise HTTPException(404, "Località non trovata")
        return {"lat": data[0]['lat'], "lon": data[0]['lon'], "name": data[0]['display_name']}
    except Exception as e:
        raise HTTPException(500, f"Errore di geocoding: {e}")

@app.get("/api/score")
async def get_score(lat: float, lon: float):
    try:
        weather_data = await get_external_data(lat, lon)
        terrain_data = await get_terrain_data(lat, lon)
        prediction = predictor.calculate_composite_score(weather_data, terrain_data)
        return prediction
    except Exception as e:
        raise HTTPException(500, f"Errore nel calcolo del punteggio: {e}")








