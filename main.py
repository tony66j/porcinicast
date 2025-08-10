# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from datetime import datetime, timedelta

# --- MODELLO SCIENTIFICO AVANZATO v2 ---
class PorciniPredictor:
    def _normalize(self, value, min_val, max_val):
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def _calculate_soil_temp_score(self, soil_temps_past_10d):
        avg_soil_temp = np.mean(soil_temps_past_10d)
        # L'ottimo per il micelio è 17.5°C. Il punteggio diminuisce linearmente fino a una tolleranza di 8°C.
        return 1.0 - self._normalize(abs(avg_soil_temp - 17.5), 0, 8)

    def _find_triggering_rain_event(self, past_30d_precip, past_30d_temps):
        # Un evento scatenante è definito da: >25mm di pioggia in 3 giorni, preceduto da 7 giorni relativamente secchi (<10mm totali)
        # e accompagnato da un calo termico (shock).
        for i in range(20, -1, -1): # Scansiona dal passato più recente
            seven_days_before = past_30d_precip[i-7:i] if i >= 7 else [0]*7
            three_days_of_rain = past_30d_precip[i:i+3]
            
            if sum(seven_days_before) < 10 and sum(three_days_of_rain) >= 25:
                # Abbiamo trovato un potenziale shock idrico. Verifichiamo lo shock termico.
                temp_before_rain = np.mean(past_30d_temps[i-3:i]) if i >= 3 else 20
                temp_during_rain = np.mean(past_30d_temps[i:i+3])
                if temp_before_rain - temp_during_rain > 2.0: # Calo di almeno 2 gradi
                    days_since_event_end = (30 - 1) - (i + 2)
                    return {"days_since": days_since_event_end, "precip_mm": sum(three_days_of_rain)}
        
        return {"days_since": 99, "precip_mm": 0}

    def _calculate_incubation_score(self, days_since_event, soil_temps_during_incubation, humidity_during_incubation):
        if not (10 <= days_since_event <= 24):
            return 0.0 # Fuori dalla finestra di incubazione

        # Il punteggio è massimo a 16 giorni dall'evento
        base_score = 1.0 - abs(days_since_event - 16) / 8.0

        # L'incubazione è modulata da temperatura e umidità
        avg_soil_temp = np.mean(soil_temps_during_incubation)
        avg_humidity = np.mean(humidity_during_incubation)
        
        temp_factor = 1.0 - self._normalize(abs(avg_soil_temp - 18.0), 0, 5) # Ottimo a 18°C durante incubazione
        humidity_factor = self._normalize(avg_humidity, 75, 95) # Ottimo con umidità alta
        
        return base_score * temp_factor * humidity_factor
        
    def calculate_score(self, weather_data):
        # 1. Analisi dello SHOCK IDRO-TERMICO
        rain_event = self._find_triggering_rain_event(weather_data['past']['precipitation'], weather_data['past']['temp_mean'])
        days_since_event = rain_event['days_since']

        # 2. Analisi FASE DI INCUBAZIONE
        start_incubation_idx = -days_since_event
        soil_temps_incubation = weather_data['past']['soil_temperature'][start_incubation_idx:]
        humidity_incubation = weather_data['past']['humidity'][start_incubation_idx:]
        
        incubation_score = self._calculate_incubation_score(days_since_event, soil_temps_incubation, humidity_incubation)
        
        # In questo modello avanzato, l'incubazione dopo uno shock è il fattore dominante (peso > 80%)
        # Altri fattori (terreno, ecc.) diventano moltiplicatori.
        final_score = incubation_score * 100
        
        # Logica per status e previsione (invariata ma ora basata su dati migliori)
        if final_score < 45: harvest = "Nullo o molto scarso (< 0.1 kg)"
        elif final_score < 70: harvest = "Discreto (0.2 - 0.7 kg)"
        elif final_score < 85: harvest = "Buono (0.8 - 2.0 kg)"
        else: harvest = "Eccellente (> 2.0 kg)"

        if days_since_event <= 10: window_status = f"Inizio buttata previsto tra {10 - days_since_event} giorni."
        elif 10 < days_since_event <= 24: window_status = "Buttata in corso! Questo è il momento."
        else: window_status = "In attesa di un evento di pioggia e calo termico."
        
        peak_date = (datetime.now() + timedelta(days=max(0, 16 - days_since_event))).strftime("%d %B")

        return {
            "overall_score": int(final_score),
            "estimated_harvest_per_2h": harvest,
            "fruiting_window": { "status": window_status, "peak_date": peak_date },
            "weather_forecast": weather_data['forecast']
        }

# --- API FastAPI ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
predictor = PorciniPredictor()
http_client = httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "TrovaPorcini/3.0"})

async def get_real_weather_data(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    # Aggiunti tutti i parametri necessari per il modello avanzato
    params = {
        "latitude": lat, "longitude": lon, "past_days": 30, "forecast_days": 7,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,shortwave_radiation_sum,soil_temperature_0_to_7cm",
        "timezone": "auto"
    }
    response = await http_client.get(url, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=503, detail=f"Errore API Meteo ({response.status_code}): {response.text}")
    data = response.json()['daily']
    
    split_point = -7
    return {
        "past": {
            "precipitation": data['precipitation_sum'][:split_point],
            "soil_temperature": data['soil_temperature_0_to_7cm'][:split_point],
            "temp_mean": data['temperature_2m_mean'][:split_point],
            "humidity": data['relative_humidity_2m_mean'][:split_point],
        },
        "forecast": {
            "time": data['time'][split_point:],
            "temp_max": data['temperature_2m_max'][split_point:],
            "temp_min": data['temperature_2m_min'][split_point:],
            "precipitation": data['precipitation_sum'][split_point:],
        }
    }

@app.get("/api/geocode")
async def geocode(q: str):
    url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1&accept-language=it"
    r = await http_client.get(url, timeout=10);
    if r.status_code != 200: raise HTTPException(500, "Errore Geocoding")
    data = r.json()
    if not data: raise HTTPException(404, "Località non trovata")
    return {"lat": data[0]['lat'], "lon": data[0]['lon'], "name": data[0]['display_name']}

@app.get("/api/score")
async def get_score(lat: float, lon: float):
    try:
        weather_data = await get_real_weather_data(lat, lon)
        prediction = predictor.calculate_score(weather_data)
        prediction["coords"] = {"lat": lat, "lon": lon}
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

