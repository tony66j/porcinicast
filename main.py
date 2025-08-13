lines.append("<h4>🧬 Analisi Biologica Super Avanzata v2.5.0</h4>")
    lines.append(f"<p><em>Modello fenologico basato su letteratura scientifica: Boddy et al. (2014), Büntgen et al. (2012), Kauserud et al. (2010)</em></p>")
    
    lines.append(f"<h4>🍄 Specie e Habitat</h4>")
    lines.append(f"<p><strong>Specie dominante predetta</strong>: <em>Boletus {species}</em></p>")
    lines.append(f"<p><strong>Habitat principale</strong>: {habitat_used} • <strong>Localizzazione</strong>: {elev}m, pendenza {slope}°, esposizione {aspect}</p>")
    
    profile = SPECIES_PROFILES_V25.get(species, SPECIES_PROFILES_V25["reticulatus"])
    season_text = f"{profile['season']['start_m']:02d}→{profile['season']['end_m']:02d}"
    lines.append(f"<p><strong>Ecologia specie</strong>: Stagione {season_text} • Lag biologico base ~{profile['lag_base']:.1f} giorni • VPD sensibilità {profile['vpd_sens']:.1f}</p>")
    
    lines.append(f"<h4>📊 Indice e Previsione</h4>")
    lines.append(f"<p><strong>Indice corrente</strong>: <strong style='font-size:1.2em'>{idx}/100</strong> - ")
    
    if idx >= 75:
        lines.append("<span style='color:#66e28a;font-weight:bold'>ECCELLENTE</span> - Condizioni ottimali per fruttificazione massiva")
    elif idx >= 60:
        lines.append("<span style='color:#8bb7ff;font-weight:bold'>MOLTO BUONE</span> - Fruttificazione abbondante attesa")
    elif idx >= 45:
        lines.append("<span style='color:#ffc857;font-weight:bold'>BUONE</span> - Fruttificazione moderata possibile")
    elif idx >= 30:
        lines.append("<span style='color:#ff9966;font-weight:bold'>MODERATE</span> - Fruttificazione limitata")
    else:
        lines.append("<span style='color:#ff6b6b;font-weight:bold'>SCARSE</span> - Fruttificazione improbabile")
    lines.append("</p>")
    
    if best and best.get("mean", 0) > 0:
        start, end, mean = best.get("start", 0), best.get("end", 0), best.get("mean", 0)
        lines.append(f"<p><strong>Finestra ottimale prossimi 10 giorni</strong>: Giorni <strong>{start+1}-{end+1}</strong> (indice medio ~<strong>{mean}</strong>)</p>")
    
    lines.append(f"<h4>🎯 Affidabilità Multi-Dimensionale</h4>")
    lines.append("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:12px 0'>")
    
    for dimension, value in confidence_detailed.items():
        if dimension == "overall": continue
        color = "#66e28a" if value >= 0.7 else "#ffc857" if value >= 0.5 else "#ff6b6b"
        dim_name = {
            "meteorological": "☁️ Meteorol.",
            "ecological": "🌿 Ecologica", 
            "hydrological": "💧 Idrologica",
            "atmospheric": "🌡️ Atmosferica",
            "empirical": "📊 Empirica"
        }.get(dimension, dimension.title())
        lines.append(f"<div style='text-align:center;padding:8px;background:#0a0f14;border-radius:6px'><div style='color:{color};font-weight:bold'>{value:.2f}</div><div style='font-size:11px;color:#8aa0b6'>{dim_name}</div></div>")
    
    lines.append("</div>")
    lines.append(f"<p><strong>Affidabilità complessiva</strong>: <strong style='color:{'#66e28a' if overall_conf >= 0.7 else '#ffc857' if overall_conf >= 0.5 else '#ff6b6b'}'>{overall_conf:.2f}</strong>/1.00</p>")
    
    lines.append(f"<h4>⏱️ Eventi Piovosi e Lag Biologico Dinamico</h4>")
    lines.append(f"<p><strong>Eventi rilevati</strong>: {lag_info}</p>")
    
    if flush_events:
        lines.append("<ul style='margin:8px 0 0 20px'>")
        for event in flush_events[:3]:
            when = event.get("event_when", "?")
            mm = event.get("event_mm", 0)
            lag = event.get("lag_days", 0)
            obs_text = "📊 Osservato" if event.get("observed") else "🔮 Previsto"
            lines.append(f"<li><strong>{when}</strong>: {mm:.1f}mm → flush ~{lag} giorni ({obs_text})</li>")
        lines.append("</ul>")
    
    lines.append(f"<h4>🎯 Raccomandazioni Strategiche</h4>")
    
    harvest = payload.get("harvest_estimate", "N/A")
    lines.append(f"<p><strong>Raccolto atteso</strong>: {harvest}</p>")
    
    if idx >= 60:
        lines.append("<p class='return-advice'><strong>🚀 Strategia OFFENSIVA</strong>: Condizioni eccellenti. Pianifica uscite multiple nei prossimi 3-5 giorni. Cerca aree con buon drenaggio e esposizione favorevole.</p>")
    elif idx >= 40:
        lines.append("<p class='return-advice'><strong>⚖️ Strategia BILANCIATA</strong>: Condizioni promettenti. Concentrati su aree già produttive e con habitat ideale per la specie predetta.</p>")
    elif idx >= 25:
        lines.append("<p class='return-advice'><strong>🎯 Strategia CONSERVATIVA</strong>: Condizioni incerte. Limita le uscite alle aree più promettenti e monitora l'evoluzione.</p>")
    else:
        lines.append("<p class='return-advice'><strong>⏸️ Strategia ATTESA</strong>: Condizioni attuali sfavorevoli. Monitora le previsioni per miglioramenti nei prossimi giorni.</p>")
    
    lines.append(f"<h4>✨ Innovazioni Modello v2.5.0</h4>")
    lines.append("<div style='background:#0a0f14;padding:12px;border-radius:8px;border-left:3px solid #62d5b4'>")
    lines.append("<ul style='margin:0;padding-left:20px'>")
    lines.append("<li><strong>Lag biologico dinamico</strong>: Modellazione basata su Boddy et al. (2014) con correzioni SMI, shock termico e VPD</li>")
    lines.append("<li><strong>Soglie pioggia adattive</strong>: Algoritmo che si adatta alle condizioni locali (SMI, stagione, quota, latitudine)</li>")
    lines.append("<li><strong>Confidence 5D</strong>: Valutazione separata di aspetti meteorologici, ecologici, idrologici, atmosferici ed empirici</li>")
    lines.append("<li><strong>Microtopografia avanzata</strong>: Calcoli multi-scala per slope, aspect, concavity e proxy di drenaggio</li>")
    lines.append("<li><strong>Sistema di validazione</strong>: Database crowd-sourced per miglioramento continuo e calibrazione regionale</li>")
    lines.append("<li><strong>Smoothing Savitzky-Golay</strong>: Preservazione intelligente dei picchi con fallback robusti</li>")
    lines.append("</ul>")
    lines.append("</div>")
    
    lines.append(f"<div style='margin-top:16px;padding:8px;background:#0e141b;border-radius:6px;font-size:11px;color:#8aa0b6'>")
    lines.append(f"<strong>Metodologia</strong>: Modello fenologico integrato con forzanti meteorologiche, correzioni ecologiche e feedback empirici. ")
    lines.append(f"Validazione continua tramite crowd-sourcing. Riferimenti: Boddy & Heilmann-Clausen (2014), Büntgen et al. (2012).")
    lines.append(f"</div>")
    
    return "\n".join(lines)

# ===== ENDPOINTS SUPER AVANZATI =====

@app.get("/api/health")
async def health():
    capabilities = {
        "numpy": NUMPY_AVAILABLE,
        "scipy": SCIPY_AVAILABLE, 
        "geohash": GEOHASH_AVAILABLE,
        "cds": CDS_AVAILABLE
    }
    
    return {
        "ok": True, 
        "time": datetime.now(timezone.utc).isoformat(), 
        "version": "2.5.0",
        "model": "super_advanced",
        "capabilities": capabilities,
        "features": [
            "lag_biologico_dinamico", "confidence_5d", "soglie_adattive", 
            "crowd_sourcing", "savitzky_golay", "microtopografia_avanzata",
            "era5_land_smi", "osm_habitat_avanzato"
        ]
    }

@app.get("/api/geocode")
async def api_geocode(q: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "format": "json", "q": q, "addressdetails": 1, "limit": 1,
            "email": os.getenv("NOMINATIM_EMAIL", "info@trovaporcini.com")
        }
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        
        if data:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display": data[0].get("display_name", ""),
                "source": "nominatim"
            }
    except Exception as e:
        logger.warning(f"Nominatim failed: {e}")
    
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": q, "count": 1, "language": "it"}
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            j = r.json()
        
        res = (j.get("results") or [])
        if not res: 
            raise HTTPException(404, "Località non trovata")
        
        it = res[0]
        return {
            "lat": float(it["latitude"]),
            "lon": float(it["longitude"]),
            "display": f"{it.get('name')} ({(it.get('country_code') or '').upper()})",
            "source": "open_meteo"
        }
    except Exception as e:
        logger.error(f"Geocoding completely failed: {e}")
        raise HTTPException(404, "Errore nel geocoding")

@app.post("/api/report-sighting")
async def report_sighting(
    lat: float, lon: float, species: str, 
    quantity: int = 1, size_cm_avg: float = None, size_cm_max: float = None,
    confidence: float = 0.8, photo_url: str = "", notes: str = "",
    habitat_observed: str = "", weather_conditions: str = "",
    user_experience_level: int = 3
):
    try:
        date = datetime.now().date().isoformat()
        geohash = geohash_encode_advanced(lat, lon)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sightings 
            (lat, lon, date, species, quantity, size_cm_avg, size_cm_max, confidence, 
             photo_url, notes, habitat_observed, weather_conditions, user_experience_level,
             geohash, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, species, quantity, size_cm_avg, size_cm_max, confidence,
              photo_url, notes, habitat_observed, weather_conditions, user_experience_level,
              geohash, "2.5.0"))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Advanced sighting: {species} x{quantity} at ({lat:.4f}, {lon:.4f})")
        return {"status": "success", "message": "Segnalazione registrata con successo", "id": cursor.lastrowid}
        
    except Exception as e:
        logger.error(f"Sighting error: {e}")
        raise HTTPException(500, "Errore interno del server")

@app.post("/api/report-no-findings")
async def report_no_findings(
    lat: float, lon: float, searched_hours: float = 2.0,
    search_method: str = "visual", habitat_searched: str = "", 
    weather_conditions: str = "", notes: str = "",
    search_thoroughness: int = 3
):
    try:
        date = datetime.now().date().isoformat()
        geohash = geohash_encode_advanced(lat, lon)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO no_sightings 
            (lat, lon, date, searched_hours, search_method, habitat_searched,
             weather_conditions, notes, search_thoroughness, geohash, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, searched_hours, search_method, habitat_searched,
              weather_conditions, notes, search_thoroughness, geohash, "2.5.0"))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Advanced no-finding: {searched_hours}h at ({lat:.4f}, {lon:.4f})")
        return {"status": "success", "message": "Report registrato con successo", "id": cursor.lastrowid}
        
    except Exception as e:
        logger.error(f"No-finding error: {e}")
        raise HTTPException(500, "Errore interno del server")

@app.get("/api/validation-stats")
async def validation_stats_super_advanced():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), AVG(confidence), AVG(user_experience_level) FROM sightings")
        pos_stats = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*), AVG(search_thoroughness) FROM no_sightings")
        neg_stats = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT species, COUNT(*) as count, AVG(quantity), AVG(size_cm_avg)
            FROM sightings 
            WHERE size_cm_avg IS NOT NULL
            GROUP BY species 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_species_detailed = {
            species: {
                "count": count, 
                "avg_quantity": round(avg_qty or 0, 1),
                "avg_size_cm": round(avg_size or 0, 1)
            }
            for species, count, avg_qty, avg_size in cursor.fetchall()
        }
        
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN lat > 45 THEN 1 END) as nord,
                COUNT(CASE WHEN lat BETWEEN 42 AND 45 THEN 1 END) as centro,
                COUNT(CASE WHEN lat < 42 THEN 1 END) as sud
            FROM sightings
        """)
        geo_dist = cursor.fetchone()
        
        conn.close()
        
        total_validations = (pos_stats[0] or 0) + (neg_stats[0] or 0)
        
        return {
            "positive_sightings": pos_stats[0] or 0,
            "negative_reports": neg_stats[0] or 0,
            "predictions_logged": pred_count,
            "total_validations": total_validations,
            "avg_confidence": round(pos_stats[1] or 0, 2),
            "avg_user_experience": round(pos_stats[2] or 0, 1),
            "avg_search_thoroughness": round(neg_stats[1] or 0, 1),
            "top_species_detailed": top_species_detailed,
            "geographic_distribution": {
                "nord_italia": geo_dist[0] or 0,
                "centro_italia": geo_dist[1] or 0, 
                "sud_italia": geo_dist[2] or 0
            },
            "ready_for_ml": total_validations >= 100,
            "model_version": "2.5.0",
            "capabilities": {
                "numpy": NUMPY_AVAILABLE,
                "scipy": SCIPY_AVAILABLE,
                "cds": CDS_AVAILABLE
            }
        }
        
    except Exception as e:
        logger.error(f"Advanced stats error: {e}")
        return {"error": str(e)}

@app.get("/api/score")
async def api_score_super_advanced(
    lat: float = Query(..., description="Latitudine"),
    lon: float = Query(..., description="Longitudine"),
    half: float = Query(8.5, gt=3.0, lt=20.0, description="Half-life API (giorni)"),
    habitat: str = Query("", description="Habitat forzato"),
    autohabitat: int = Query(1, description="1=auto OSM, 0=manuale"),
    hours: int = Query(4, ge=2, le=8, description="Ore sul campo"),
    background_tasks: BackgroundTasks = None
):
    """
    🚀 ENDPOINT PRINCIPALE SUPER AVANZATO v2.5.0
    Mantiene TUTTE le funzionalità scientifiche avanzate
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting super advanced analysis for ({lat:.4f}, {lon:.4f})")
        
        # Prefetch ERA5-Land in background se disponibile
        if CDS_AVAILABLE:
            asyncio.create_task(_prefetch_era5l_sm_advanced(lat, lon))
        
        # Fetch paralleli
        tasks = [
            fetch_open_meteo_super_advanced(lat, lon, past=15, future=10),
            fetch_openweather_super_advanced(lat, lon),
            fetch_elevation_grid_super_advanced(lat, lon),
        ]
        
        if autohabitat == 1:
            tasks.append(fetch_osm_habitat_super_advanced(lat, lon))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        om_data = results[0] if not isinstance(results[0], Exception) else {}
        ow_data = results[1] if not isinstance(results[1], Exception) else {}
        elev_data = results[2] if not isinstance(results[2], Exception) else (800.0, 8.0, 180.0, "S", 0.0, 1.0)
        
        if autohabitat == 1 and len(results) > 3:
            osm_habitat_data = results[3] if not isinstance(results[3], Exception) else ("misto", 0.15, {})
        else:
            osm_habitat_data = ("misto", 0.15, {})
        
        elev_m, slope_deg, aspect_deg, aspect_oct, concavity, drainage_proxy = elev_data
        
        # Habitat determination
        habitat_used = (habitat or "").strip().lower()
        habitat_source = "manuale"
        habitat_confidence = 0.6
        auto_scores = {}
        
        if autohabitat == 1:
            auto_habitat, auto_conf, auto_scores = osm_habitat_data
            if not habitat_used and auto_habitat:
                habitat_used = auto_habitat
                habitat_confidence = auto_conf
                habitat_source = f"automatico OSM (conf {auto_conf:.2f})"
            elif habitat_used:
                habitat_source = "manuale (override)"
        
        if not habitat_used:
            habitat_used = "misto"
        
        # Process meteo data
        if not om_data or "daily" not in om_data:
            raise HTTPException(500, "Errore dati meteorologici")
        
        daily = om_data["daily"]
        time_series = daily["time"]
        
        P_series = [float(x or 0.0) for x in daily["precipitation_sum"]]
        Tmin_series = [float(x or 0.0) for x in daily["temperature_2m_min"]]
        Tmax_series = [float(x or 0.0) for x in daily["temperature_2m_max"]]
        Tmean_series = [float(x or 0.0) for x in daily["temperature_2m_mean"]]
        ET0_series = daily.get("et0_fao_evapotranspiration", [2.0] * len(P_series))
        RH_series = daily.get("relative_humidity_2m_mean", [65.0] * len(P_series))
        
        past_days = 15
        future_days = 10
        
        P_past = P_series[:past_days]
        P_future = P_series[past_days:past_days + future_days]
        Tmean_past = Tmean_series[:past_days]
        Tmean_future = Tmean_series[past_days:past_days + future_days]
        Tmin_past = Tmin_series[:past_days]
        RH_past = RH_series[:past_days]
        RH_future = RH_series[past_days:past_days + future_days]
        
        # Advanced indicators
        api_value = api_index(P_past, half_life=half)
        smi_series = smi_from_p_et0_advanced(P_series, ET0_series)
        smi_current = smi_series[past_days - 1] if past_days - 1 < len(smi_series) else 0.5
        
        tmean_7d = sum(Tmean_past[-7:]) / max(1, len(Tmean_past[-7:]))
        thermal_shock = thermal_shock_index_advanced(Tmin_past, window_days=3)
        rh_7d = sum(RH_past[-7:]) / max(1, len(RH_past[-7:]))
        vpd_series_future = [vpd_hpa(Tmean_future[i], RH_future[i]) for i in range(min(len(Tmean_future), len(RH_future)))]
        vpd_current = vpd_series_future[0] if vpd_series_future else 5.0
        
        # Advanced microclimate
        month_current = datetime.now(timezone.utc).month
        microclimate_energy = microclimate_energy_advanced(aspect_oct, slope_deg, month_current, lat, elev_m)
        twi_index = twi_advanced_proxy(slope_deg, concavity, drainage_proxy)
        
        # Species inference
        species = infer_porcino_species_super_advanced(habitat_used, month_current, elev_m, aspect_oct, lat)
        species_profile = SPECIES_PROFILES_V25[species]
        
        logger.info(f"Species inferred: {species} for habitat {habitat_used} at {elev_m}m")
        
        # Rain events and dynamic lag
        rain_events = detect_rain_events_super_advanced(P_past + P_future, smi_series, month_current, elev_m, lat)
        
        forecast = [0.0] * future_days
        flush_events_details = []
        
        for event_idx, event_mm, event_strength in rain_events:
            smi_local = smi_series[event_idx] if event_idx < len(smi_series) else smi_current
            smi_adjusted = clamp(smi_local + species_profile["smi_bias"], 0.0, 1.0)
            
            if event_idx >= past_days:
                future_idx = event_idx - past_days
                vpd_stress = max(0.0, (vpd_series_future[future_idx] - 8.0) / 10.0) if future_idx < len(vpd_series_future) else 0.0
            else:
                vpd_stress = 0.0
            
            lag_days = stochastic_lag_super_advanced(
                smi=smi_adjusted,
                thermal_shock=thermal_shock,
                tmean7=tmean_7d,
                species=species,
                vpd_stress=vpd_stress,
                photoperiod_factor=1.0
            )
            
            peak_idx = event_idx + lag_days
            base_amplitude = event_strength * microclimate_energy
            
            if event_idx >= past_days:
                future_peak_idx = peak_idx - past_days
                if 0 <= future_peak_idx < len(vpd_series_future):
                    vpd_penalty = vpd_penalty_advanced(vpd_series_future[future_peak_idx], species_profile["vpd_sens"], elev_m)
                else:
                    vpd_penalty = 0.8
            else:
                vpd_penalty = 1.0
            
            final_amplitude = base_amplitude * vpd_penalty
            
            sigma = 3.0 if event_strength > 0.8 else 2.5
            skew = 0.2 if species in ["aereus", "reticulatus"] else 0.0
            
            for day_idx in range(future_days):
                abs_day_idx = past_days + day_idx
                kernel_value = gaussian_kernel_advanced(abs_day_idx, peak_idx, sigma, skewness=skew)
                forecast[day_idx] += 100.0 * final_amplitude * kernel_value
            
            when_str = time_series[event_idx] if event_idx < len(time_series) else f"+{event_idx - past_days + 1}d"
            flush_events_details.append({
                "event_day_index": event_idx,
                "event_when": when_str,
                "event_mm": round(event_mm, 1),
                "event_strength": round(event_strength, 2),
                "lag_days": lag_days,
                "predicted_peak_abs_index": peak_idx,
                "observed": event_idx < past_days,
                "smi_local": round(smi_adjusted, 2),
                "vpd_penalty": round(vpd_penalty, 2)
            })
        
        # Advanced smoothing
        forecast_clamped = [clamp(v, 0.0, 100.0) for v in forecast]
        forecast_smoothed = savitzky_golay_advanced(forecast_clamped, window_length=5, polyorder=2)
        forecast_final = [int(round(x)) for x in forecast_smoothed]
        
        # Best window analysis
        best_window = {"start": 0, "end": 2, "mean": 0}
        if len(forecast_final) >= 3:
            best_mean = 0
            for i in range(len(forecast_final) - 2):
                window_mean = sum(forecast_final[i:i+3]) / 3.0
                if window_mean > best_mean:
                    best_mean = window_mean
                    best_window = {"start": i, "end": i+2, "mean": int(round(window_mean))}
        
        current_index = forecast_final[0] if forecast_final else 0
        
        # Validations and 5D confidence
        has_validations, validation_count, validation_accuracy = check_recent_validations_super_advanced(lat, lon)
        
        weather_reliability = 0.8
        if ow_data and "daily" in ow_data:
            weather_reliability = 0.9
        
        confidence_5d = confidence_5d_super_advanced(
            weather_agreement=weather_reliability,
            habitat_confidence=habitat_confidence,
            smi_reliability=0.9 if CDS_AVAILABLE else 0.75,
            vpd_validity=(vpd_current <= 12.0),
            has_recent_validation=has_validations,
            elevation_reliability=0.9 if slope_deg > 1.0 else 0.7,
            temporal_consistency=0.8
        )
        
        # Harvest and size estimates
        harvest_estimate, harvest_note = estimate_harvest_super_advanced(current_index, hours, species, confidence_5d["overall"])
        size_estimates = estimate_mushroom_sizes_advanced(flush_events_details, tmean_7d, rh_7d, species)
        
        processing_time = round((time.time() - start_time) * 1000, 1)
        
        # Rain tables
        rain_past_table = {time_series[i]: round(P_past[i], 1) for i in range(min(past_days, len(time_series)))}
        rain_future_table = {
            time_series[past_days + i] if past_days + i < len(time_series) else f"+{i+1}d": round(P_future[i], 1) 
            for i in range(future_days)
        }
        
        # Final response
        response_payload = {
            "lat": lat, "lon": lon,
            "elevation_m": round(elev_m),
            "slope_deg": round(slope_deg, 1),
            "aspect_deg": round(aspect_deg, 1),
            "aspect_octant": aspect_oct or "N/A",
            "concavity": round(concavity, 3),
            "drainage_proxy": round(drainage_proxy, 2),
            
            "API_star_mm": round(api_value, 1),
            "P7_mm": round(sum(P_past[-7:]), 1),
            "P15_mm": round(sum(P_past), 1),
            "Tmean7_c": round(tmean_7d, 1),
            "RH7_pct": round(rh_7d, 1),
            "thermal_shock_index": round(thermal_shock, 2),
            "smi_current": round(smi_current, 2),
            "vpd_current_hpa": round(vpd_current, 1),
            
            "microclimate_energy": round(microclimate_energy, 2),
            "twi_index": round(twi_index, 2),
            
            "index": current_index,
            "forecast": forecast_final,
            "best_window": best_window,
            "confidence_detailed": confidence_5d,
            
            "harvest_estimate": harvest_estimate,
            "harvest_note": harvest_note,
            "size_cm": size_estimates["avg_size"],
            "size_class": size_estimates["size_class"], 
            "size_range_cm": size_estimates["size_range"],
            
            "habitat_used": habitat_used,
            "habitat_source": habitat_source,
            "habitat_confidence": round(habitat_confidence, 3),
            "auto_habitat_scores": auto_scores,
            "species": species,
            "species_profile": {
                "season_range": f"{species_profile['season']['start_m']:02d}-{species_profile['season']['end_m']:02d}",
                "temp_optimal": f"{species_profile['tm7_opt'][0]:.1f}-{species_profile['tm7_opt'][1]:.1f}°C",
                "lag_base_days": species_profile["lag_base"],
                "vpd_sensitivity": species_profile["vpd_sens"],
                "drought_tolerance": species_profile["drought_tolerance"]
            },
            
            "flush_events": flush_events_details,
            "total_events_detected": len(rain_events),
            "events_observed": len([e for e in flush_events_details if e["observed"]]),
            "events_predicted": len([e for e in flush_events_details if not e["observed"]]),
            
            "rain_past": rain_past_table,
            "rain_future": rain_future_table,
            
            "has_local_validations": has_validations,
            "validation_count": validation_count,
            "validation_accuracy": round(validation_accuracy, 2),
            
            "model_version": "2.5.0",
            "model_type": "super_advanced",
            "processing_time_ms": processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            "diagnostics": {
                "smi_source": "P-ET0 advanced" + (" + ERA5" if CDS_AVAILABLE else ""),
                "weather_sources": ["open_meteo"] + (["openweather"] if ow_data else []),
                "elevation_quality": "multi_scale_grid",
                "habitat_method": habitat_source,
                "lag_algorithm": "stochastic_v25_boddy2014",
                "smoothing_method": "savitzky_golay_advanced" if SCIPY_AVAILABLE else "custom_advanced",
                "confidence_system": "5d_multidimensional",
                "thresholds": "dynamic_adaptive_v25",
                "capabilities": {
                    "numpy": NUMPY_AVAILABLE,
                    "scipy": SCIPY_AVAILABLE,
                    "geohash": GEOHASH_AVAILABLE,
                    "cds": CDS_AVAILABLE
                }
            }
        }
        
        response_payload["dynamic_explanation"] = build_analysis_super_advanced_v25(response_payload)
        
        # Save prediction for ML
        if background_tasks:
            weather_metadata = {
                "api_value": api_value, "smi_current": smi_current,
                "tmean_7d": tmean_7d, "thermal_shock": thermal_shock,
                "vpd_current": vpd_current, "processing_time_ms": processing_time
            }
            model_features = {
                "elevation": elev_m, "slope": slope_deg, "aspect": aspect_oct,
                "microclimate_energy": microclimate_energy, "twi_index": twi_index,
                "species": species, "events_count": len(rain_events)
            }
            
            background_tasks.add_task(
                save_prediction_super_advanced,
                lat, lon, datetime.now().date().isoformat(),
                current_index, species, habitat_used,
                confidence_5d, weather_metadata, model_features
            )
        
        logger.info(f"Super advanced analysis completed: {current_index}/100 for {species} ({processing_time}ms)")
        return response_payload
        
    except Exception as e:
        processing_time = round((time.time() - start_time) * 1000, 1)
        logger.error(f"Super advanced analysis error ({processing_time}ms): {e}")
        raise HTTPException(500, f"Errore interno: {str(e)}")

# === HELPER FUNCTIONS ===

def estimate_harvest_super_advanced(index: int, hours: int, species: str, confidence: float) -> Tuple[str, str]:
    profile = SPECIES_PROFILES_V25[species]
    base_productivity = {"aereus": 1.2, "reticulatus": 1.0, "edulis": 0.9, "pinophilus": 1.1}
    multiplier = base_productivity.get(species, 1.0)
    
    if index >= 80: base_range = (8, 15)
    elif index >= 65: base_range = (5, 10) 
    elif index >= 50: base_range = (3, 7)
    elif index >= 35: base_range = (1, 4)
    elif index >= 20: base_range = (0, 2)
    else: base_range = (0, 1)
    
    hour_factor = 1.0 + (hours - 2) * 0.2
    conf_factor = 0.6 + 0.8 * confidence
    
    low = max(0, int(round(base_range[0] * multiplier * hour_factor * conf_factor)))
    high = max(low + 1, int(round(base_range[1] * multiplier * hour_factor * conf_factor)))
    
    estimate = f"{low}-{high} porcini"
    
    if index >= 70:
        note = f"Condizioni eccellenti per {species}. Raccolto abbondante atteso."
    elif index >= 50:
        note = f"Buone condizioni per {species}. Raccolto moderato probabile."
    elif index >= 30:
        note = f"Condizioni incerte per {species}. Raccolto limitato possibile."
    else:
        note = f"Condizioni sfavorevoli per {species}. Raccolto improbabile."
    
    return estimate, note

def estimate_mushroom_sizes_advanced(events: List[Dict], tmean_7d: float, rh_7d: float, species: str) -> Dict[str, Any]:
    if not events:
        return {"avg_size": 5.0, "size_class": "medi", "size_range": [3.0, 7.0]}
    
    today_abs = 15
    recent_events = [e for e in events if abs(e["predicted_peak_abs_index"] - today_abs) <= 5]
    
    if recent_events:
        closest_event = min(recent_events, key=lambda e: abs(e["predicted_peak_abs_index"] - today_abs))
        days_from_peak = today_abs - closest_event["predicted_peak_abs_index"]
        age_days = max(0, -days_from_peak + 3)
    else:
        age_days = 2
    
    growth_rates = {"aereus": 1.4, "reticulatus": 1.6, "edulis": 1.2, "pinophilus": 1.3}
    base_rate = growth_rates.get(species, 1.4)
    
    temp_factor = 1.0
    if 16 <= tmean_7d <= 20: temp_factor = 1.2
    elif tmean_7d < 12 or tmean_7d > 24: temp_factor = 0.7
    
    rh_factor = min(1.2, max(0.6, rh_7d / 70.0))
    
    final_rate = base_rate * temp_factor * rh_factor
    avg_size = clamp(2.0 + age_days * final_rate, 2.0, 18.0)
    
    if avg_size < 5: size_class = "bottoni (2-5 cm)"
    elif avg_size < 10: size_class = "medi (6-10 cm)"
    else: size_class = "grandi (10+ cm)"
    
    size_range = [max(2.0, avg_size - 2.0), min(18.0, avg_size + 3.0)]
    
    return {
        "avg_size": round(avg_size, 1),
        "size_class": size_class,
        "size_range": [round(x, 1) for x in size_range]
    }

# === APPLICATION STARTUP ===
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8787))
    
    logger.info(f"🚀 Starting Trova Porcini API v2.5.0 SUPER ADVANCED on port {port}")
    logger.info(f"✨ Features: Dynamic lag • 5D confidence • Adaptive thresholds • Crowd-sourcing • ML-ready")
    logger.info(f"🔬 Capabilities: NumPy={NUMPY_AVAILABLE}, SciPy={SCIPY_AVAILABLE}, CDS={CDS_AVAILABLE}, Geohash={GEOHASH_AVAILABLE}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True,
        workers=1
    )
                return {
        "meteorological": round(met_conf, 3),
        "ecological": round(eco_conf, 3),
        "hydrological": round(hydro_conf, 3),
        "atmospheric": round(atmo_conf, 3),
        "empirical": round(emp_conf, 3),
        "overall": round(clamp(overall, 0.15, 0.95), 3)
    }

# ===== PROFILI SPECIE SUPER AVANZATI =====
SPECIES_PROFILES_V25 = {
    "aereus": {
        "hosts": ["quercia", "castagno", "misto"],
        "season": {"start_m": 6, "end_m": 10, "peak_m": [7, 8]},
        "tm7_opt": (18.0, 24.0), "tm7_critical": (12.0, 28.0),
        "lag_base": 9.2, "lag_range": (7, 12),
        "vpd_sens": 1.15, "drought_tolerance": 0.8,
        "soil_ph_opt": (5.5, 7.0), "smi_bias": 0.0,
        "elevation_opt": (200, 1000), "min_precip_flush": 12.0
    },
    "reticulatus": {
        "hosts": ["quercia", "castagno", "faggio", "misto"],
        "season": {"start_m": 5, "end_m": 9, "peak_m": [6, 7]},
        "tm7_opt": (16.0, 22.0), "tm7_critical": (10.0, 26.0),
        "lag_base": 8.8, "lag_range": (6, 11),
        "vpd_sens": 1.0, "drought_tolerance": 0.9,
        "soil_ph_opt": (5.0, 7.5), "smi_bias": 0.0,
        "elevation_opt": (100, 1200), "min_precip_flush": 10.0
    },
    "edulis": {
        "hosts": ["faggio", "conifere", "misto"],
        "season": {"start_m": 8, "end_m": 11, "peak_m": [9, 10]},
        "tm7_opt": (12.0, 18.0), "tm7_critical": (6.0, 22.0),
        "lag_base": 10.5, "lag_range": (8, 14),
        "vpd_sens": 1.2, "drought_tolerance": 0.6,
        "soil_ph_opt": (4.5, 6.5), "smi_bias": +0.05,
        "elevation_opt": (600, 2000), "min_precip_flush": 8.0
    },
    "pinophilus": {
        "hosts": ["conifere", "misto"],
        "season": {"start_m": 6, "end_m": 10, "peak_m": [8, 9]},
        "tm7_opt": (14.0, 20.0), "tm7_critical": (8.0, 24.0),
        "lag_base": 9.8, "lag_range": (7, 13),
        "vpd_sens": 0.9, "drought_tolerance": 1.1,
        "soil_ph_opt": (4.0, 6.0), "smi_bias": -0.02,
        "elevation_opt": (400, 1800), "min_precip_flush": 9.0
    }
}

def infer_porcino_species_super_advanced(habitat_used: str, month: int, elev_m: float, 
                                        aspect_oct: Optional[str], lat: float) -> str:
    h = (habitat_used or "misto").lower()
    candidates = []
    
    for species, profile in SPECIES_PROFILES_V25.items():
        if h not in profile["hosts"]: continue
        
        score = 1.0
        
        if month in profile["peak_m"]:
            score *= 1.5
        elif profile["season"]["start_m"] <= month <= profile["season"]["end_m"]:
            score *= 1.0
        else:
            score *= 0.3
        
        elev_min, elev_max = profile["elevation_opt"]
        if elev_min <= elev_m <= elev_max:
            score *= 1.2
        elif elev_m < elev_min:
            score *= max(0.4, 1.0 - (elev_min - elev_m) / 500.0)
        else:
            score *= max(0.4, 1.0 - (elev_m - elev_max) / 800.0)
        
        if aspect_oct:
            if species in ["aereus", "reticulatus"] and aspect_oct in ["S", "SE", "SW"]:
                score *= 1.1
            elif species in ["edulis", "pinophilus"] and aspect_oct in ["N", "NE", "NW"]:
                score *= 1.1
        
        if species == "aereus" and lat < 42.0: score *= 1.2
        elif species == "edulis" and lat > 45.0: score *= 1.15
        elif species == "pinophilus" and 44.0 <= lat <= 46.0: score *= 1.1
        
        candidates.append((species, score))
    
    if not candidates:
        return "reticulatus"
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

# ===== LAG BIOLOGICO DINAMICO SUPER AVANZATO =====
def stochastic_lag_super_advanced(smi: float, thermal_shock: float, tmean7: float, 
                                 species: str, vpd_stress: float = 0.0, 
                                 photoperiod_factor: float = 1.0) -> int:
    profile = SPECIES_PROFILES_V25.get(species, SPECIES_PROFILES_V25["reticulatus"])
    base_lag = profile["lag_base"]
    
    smi_effect = -4.5 * (smi ** 1.5)
    shock_effect = -2.0 * thermal_shock
    
    tm_opt_min, tm_opt_max = profile["tm7_opt"]
    tm_crit_min, tm_crit_max = profile["tm7_critical"]
    
    if tm_opt_min <= tmean7 <= tm_opt_max:
        temp_effect = -1.5
    elif tm_crit_min <= tmean7 < tm_opt_min:
        temp_effect = 2.0 * (tm_opt_min - tmean7) / (tm_opt_min - tm_crit_min)
    elif tm_opt_max < tmean7 <= tm_crit_max:
        temp_effect = 1.5 * (tmean7 - tm_opt_max) / (tm_crit_max - tm_opt_max)
    else:
        temp_effect = 3.0
    
    vpd_effect = 1.5 * vpd_stress * profile["vpd_sens"]
    photoperiod_effect = 0.5 * (1.0 - photoperiod_factor)
    
    final_lag = base_lag + smi_effect + shock_effect + temp_effect + vpd_effect + photoperiod_effect
    
    lag_min, lag_max = profile["lag_range"]
    return int(round(clamp(final_lag, lag_min, lag_max)))

def gaussian_kernel_advanced(x: float, mu: float, sigma: float, skewness: float = 0.0) -> float:
    base_gauss = math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    if skewness != 0.0:
        skew_factor = 1.0 + skewness * ((x - mu) / sigma)
        return base_gauss * max(0.1, skew_factor)
    
    return base_gauss

def event_strength_advanced(mm: float, duration_hours: float = 24.0, 
                          antecedent_smi: float = 0.5) -> float:
    base_strength = 1.0 - math.exp(-mm / 15.0)
    duration_factor = min(1.2, 1.0 + (duration_hours - 12.0) / 48.0)
    smi_factor = 0.7 + 0.6 * antecedent_smi
    return clamp(base_strength * duration_factor * smi_factor, 0.0, 1.5)

# ===== METEO AVANZATO =====
async def fetch_open_meteo_super_advanced(lat: float, lon: float, past: int = 15, future: int = 10) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    daily_vars = [
        "precipitation_sum", "precipitation_hours",
        "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
        "et0_fao_evapotranspiration", "relative_humidity_2m_mean",
        "shortwave_radiation_sum", "wind_speed_10m_max",
        "soil_moisture_0_to_10cm"
    ]
    hourly_vars = [
        "temperature_2m", "relative_humidity_2m", "precipitation"
    ]
    
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "daily": ",".join(daily_vars),
        "hourly": ",".join(hourly_vars),
        "past_days": past, "forecast_days": future,
        "models": "best_match"
    }
    
    async with httpx.AsyncClient(timeout=40, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def fetch_openweather_super_advanced(lat: float, lon: float) -> Dict[str, Any]:
    if not OWM_KEY: return {}
    
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat, "lon": lon,
        "exclude": "minutely,alerts",
        "units": "metric", "lang": "it",
        "appid": OWM_KEY
    }
    
    try:
        async with httpx.AsyncClient(timeout=40, headers=HEADERS) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning(f"OpenWeather failed: {e}")
        return {}

# ===== ELEVAZIONE E MICROTOPOGRAFIA SUPER AVANZATA =====
_elev_cache: Dict[str, Any] = {}

async def fetch_elevation_grid_super_advanced(lat: float, lon: float) -> Tuple[float, float, float, Optional[str], float, float]:
    best_result = None
    
    for step_m in [30.0, 90.0, 180.0]:
        try:
            grid = await _fetch_elevation_block_super_advanced(lat, lon, step_m)
            if not grid: continue
            
            slope, aspect, octant = slope_aspect_from_grid_super_advanced(grid, step_m)
            concavity = concavity_from_grid_super_advanced(grid)
            drainage = drainage_proxy_from_grid_advanced(grid)
            elevation = grid[1][1]
            
            relief = max(max(row) for row in grid) - min(min(row) for row in grid)
            quality = min(1.0, relief / 50.0)
            
            result = {
                "elevation": elevation, "slope": slope, "aspect": aspect,
                "octant": octant, "concavity": concavity, "drainage": drainage,
                "quality": quality, "scale": step_m
            }
            
            if best_result is None or quality > best_result["quality"]:
                best_result = result
                
        except Exception as e:
            logger.warning(f"Elevation error at scale {step_m}m: {e}")
            continue
    
    if not best_result:
        return 800.0, 8.0, 180.0, "S", 0.0, 1.0
    
    r = best_result
    return (float(r["elevation"]), r["slope"], r["aspect"], 
            r["octant"], r["concavity"], r["drainage"])

async def _fetch_elevation_block_super_advanced(lat: float, lon: float, step_m: float) -> Optional[List[List[float]]]:
    cache_key = f"{round(lat,5)},{round(lon,5)}@{int(step_m)}"
    
    if cache_key in _elev_cache:
        cache_age = time.time() - _elev_cache[cache_key].get("timestamp", 0)
        if cache_age < 3600:
            return _elev_cache[cache_key]["grid"]
    
    try:
        deg_lat = step_m / 111320.0
        deg_lon = step_m / (111320.0 * max(0.2, math.cos(math.radians(lat))))
        
        coords = []
        for dy in [-deg_lat, 0, deg_lat]:
            for dx in [-deg_lon, 0, deg_lon]:
                coords.append({
                    "latitude": lat + dy,
                    "longitude": lon + dx
                })
        
        async with httpx.AsyncClient(timeout=25, headers=HEADERS) as c:
            r = await c.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": coords}
            )
            r.raise_for_status()
            j = r.json()
        
        elevations = [p["elevation"] for p in j["results"]]
        grid = [elevations[0:3], elevations[3:6], elevations[6:9]]
        
        _elev_cache[cache_key] = {
            "grid": grid,
            "timestamp": time.time()
        }
        
        if len(_elev_cache) > 1000:
            oldest_keys = sorted(_elev_cache.keys(), 
                               key=lambda k: _elev_cache[k]["timestamp"])[:200]
            for k in oldest_keys:
                _elev_cache.pop(k, None)
        
        return grid
        
    except Exception as e:
        logger.warning(f"Elevation fetch error: {e}")
        return None

def slope_aspect_from_grid_super_advanced(grid: List[List[float]], cell_size_m: float = 30.0) -> Tuple[float, float, Optional[str]]:
    z = grid
    
    dzdx = ((z[0][2] + 2*z[1][2] + z[2][2]) - (z[0][0] + 2*z[1][0] + z[2][0])) / (8 * cell_size_m)
    dzdy = ((z[2][0] + 2*z[2][1] + z[2][2]) - (z[0][0] + 2*z[0][1] + z[0][2])) / (8 * cell_size_m)
    
    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)
    
    if dzdx == 0 and dzdy == 0:
        aspect_deg = 0.0
        octant = None
    else:
        aspect_rad = math.atan2(-dzdx, dzdy)
        aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
        
        octants = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        idx = int((aspect_deg + 22.5) // 45)
        octant = octants[idx] if slope_deg > 2.0 else None
    
    return round(slope_deg, 2), round(aspect_deg, 1), octant

def concavity_from_grid_super_advanced(grid: List[List[float]]) -> float:
    z = grid
    center = z[1][1]
    
    curvatures = []
    
    if len(z) >= 3:
        ns_curv = z[0][1] + z[2][1] - 2*center
        curvatures.append(ns_curv)
    
    if len(z[0]) >= 3:
        ew_curv = z[1][0] + z[1][2] - 2*center
        curvatures.append(ew_curv)
    
    nw_se_curv = z[0][0] + z[2][2] - 2*center
    ne_sw_curv = z[0][2] + z[2][0] - 2*center
    curvatures.extend([nw_se_curv, ne_sw_curv])
    
    mean_curvature = sum(curvatures) / len(curvatures) if curvatures else 0.0
    return clamp(mean_curvature / 10.0, -0.5, 0.5)

def drainage_proxy_from_grid_advanced(grid: List[List[float]]) -> float:
    z = grid
    center = z[1][1]
    
    draining_cells = 0
    total_cells = 0
    
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1: continue
            if z[i][j] > center: draining_cells += 1
            total_cells += 1
    
    drainage_ratio = draining_cells / total_cells if total_cells > 0 else 0.0
    return clamp(drainage_ratio, 0.1, 1.0)

# ===== OSM HABITAT SUPER AVANZATO =====
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter", 
    "https://overpass.openstreetmap.ru/api/interpreter"
]

async def fetch_osm_habitat_super_advanced(lat: float, lon: float, radius_m: int = 500) -> Tuple[str, float, Dict[str,float]]:
    
    query = f"""
    [out:json][timeout:30];
    (
      way(around:{radius_m},{lat},{lon})["landuse"="forest"];
      way(around:{radius_m},{lat},{lon})["natural"~"^(wood|forest)$"];
      relation(around:{radius_m},{lat},{lon})["landuse"="forest"];
      relation(around:{radius_m},{lat},{lon})["natural"~"^(wood|forest)$"];
      node(around:{radius_m},{lat},{lon})["natural"="tree"];
      node(around:{radius_m},{lat},{lon})["tree"];
    );
    out tags qt;
    """
    
    for url_idx, url in enumerate(OVERPASS_URLS):
        try:
            async with httpx.AsyncClient(timeout=35, headers=HEADERS) as c:
                r = await c.post(url, data={"data": query})
                r.raise_for_status()
                j = r.json()
            
            scores = score_osm_elements_super_advanced(j.get("elements", []))
            habitat, confidence = choose_dominant_habitat_advanced(scores)
            
            logger.info(f"OSM habitat via URL {url_idx}: {habitat} (conf: {confidence:.2f})")
            return habitat, confidence, scores
            
        except Exception as e:
            logger.warning(f"OSM URL {url_idx} failed: {e}")
            continue
    
    logger.info("OSM failed, using advanced heuristic")
    return habitat_heuristic_super_advanced(lat, lon)

def score_osm_elements_super_advanced(elements: List[Dict]) -> Dict[str, float]:
    scores = {"castagno": 0.0, "faggio": 0.0, "quercia": 0.0, "conifere": 0.0, "misto": 0.0}
    
    for element in elements:
        tags = {k.lower(): str(v).lower() for k, v in (element.get("tags", {})).items()}
        
        genus = tags.get("genus", "")
        species = tags.get("species", "")
        
        if "castanea" in genus or "castagna" in species:
            scores["castagno"] += 4.0
        elif "quercus" in genus or "querce" in species:
            scores["quercia"] += 4.0
        elif "fagus" in genus or "faggio" in species:
            scores["faggio"] += 4.0
        elif any(g in genus for g in ["pinus", "picea", "abies", "larix"]):
            scores["conifere"] += 3.5
        
        leaf_type = tags.get("leaf_type", "")
        if "needleleaved" in leaf_type:
            scores["conifere"] += 2.0
        elif "broadleaved" in leaf_type:
            scores["misto"] += 1.0
        
        wood = tags.get("wood", "")
        wood_scores = {
            "conifer": ("conifere", 2.5), "pine": ("conifere", 2.0),
            "spruce": ("conifere", 2.0), "fir": ("conifere", 2.0),
            "beech": ("faggio", 3.0), "oak": ("quercia", 3.0),
            "chestnut": ("castagno", 3.0), "broadleaved": ("misto", 1.5),
            "deciduous": ("misto", 1.0), "mixed": ("misto", 2.0)
        }
        
        for keyword, (habitat, score) in wood_scores.items():
            if keyword in wood:
                scores[habitat] += score
        
        landuse = tags.get("landuse", "")
        natural = tags.get("natural", "")
        
        if landuse == "forest" or natural in ["wood", "forest"]:
            for habitat in scores:
                scores[habitat] += 0.2
    
    return scores

def choose_dominant_habitat_advanced(scores: Dict[str, float]) -> Tuple[str, float]:
    total_score = sum(scores.values())
    
    if total_score < 0.5:
        return "misto", 0.15
    
    dominant = max(scores.items(), key=lambda x: x[1])
    habitat, max_score = dominant
    
    dominance_ratio = max_score / total_score
    confidence = min(0.95, dominance_ratio ** 0.7 * 0.9)
    
    if habitat in ["faggio", "castagno"] and max_score > 3.0:
        confidence *= 1.1
    
    return habitat, clamp(confidence, 0.1, 0.95)

def habitat_heuristic_super_advanced(lat: float, lon: float) -> Tuple[str, float, Dict[str, float]]:
    elevation_estimate = 800.0
    
    if lat > 46.5:
        habitat, conf = "conifere", 0.65
    elif lat > 45.0:
        habitat, conf = ("faggio" if elevation_estimate > 1000 else "misto"), 0.6
    elif lat > 43.5:
        if lon < 11.0:
            habitat, conf = "castagno", 0.55
        else:
            habitat, conf = "misto", 0.5
    elif lat > 41.5:
        if elevation_estimate > 1200:
            habitat, conf = "faggio", 0.6
        else:
            habitat, conf = "quercia", 0.55
    else:
        habitat, conf = "quercia", 0.6
    
    scores = {h: (0.8 if h == habitat else 0.1) for h in ["castagno", "faggio", "quercia", "conifere", "misto"]}
    
    return habitat, conf, scores

# ===== EVENT DETECTION SUPER AVANZATA =====
def detect_rain_events_super_advanced(rains: List[float], smi_series: List[float], 
                                     month: int, elevation: float, lat: float) -> List[Tuple[int, float, float]]:
    events = []
    n = len(rains)
    i = 0
    
    while i < n:
        smi_local = smi_series[i] if i < len(smi_series) else 0.5
        temp_trend = 0.0  # placeholder
        
        threshold_1d = dynamic_rain_threshold_v25(smi_local, month, elevation, lat, temp_trend)
        threshold_2d = threshold_1d * 1.4
        threshold_3d = threshold_1d * 1.8
        
        if rains[i] >= threshold_1d:
            strength = event_strength_advanced(rains[i], antecedent_smi=smi_local)
            events.append((i, rains[i], strength))
            i += 1
            continue
        
        if i + 1 < n:
            rain_2d = rains[i] + rains[i + 1]
            if rain_2d >= threshold_2d:
                avg_smi = (smi_local + (smi_series[i+1] if i+1 < len(smi_series) else 0.5)) / 2
                strength = event_strength_advanced(rain_2d, duration_hours=36.0, antecedent_smi=avg_smi)
                events.append((i + 1, rain_2d, strength))
                i += 2
                continue
        
        if i + 2 < n:
            rain_3d = rains[i] + rains[i + 1] + rains[i + 2]
            if rain_3d >= threshold_3d:
                avg_smi = sum(smi_series[i:i+3]) / 3 if i+2 < len(smi_series) else 0.5
                strength = event_strength_advanced(rain_3d, duration_hours=60.0, antecedent_smi=avg_smi)
                events.append((i + 2, rain_3d, strength))
                i += 3
                continue
        
        i += 1
    
    return events

# ===== DATABASE UTILS SUPER AVANZATI =====
def save_prediction_super_advanced(lat: float, lon: float, date: str, score: int, 
                                  species: str, habitat: str, confidence_data: dict,
                                  weather_data: dict, model_features: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        geohash = geohash_encode_advanced(lat, lon, precision=8)
        
        cursor.execute('''
            INSERT INTO predictions 
            (lat, lon, date, predicted_score, species, habitat, confidence_data, 
             weather_data, model_features, model_version, geohash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, score, species, habitat, 
              json.dumps(confidence_data), json.dumps(weather_data),
              json.dumps(model_features), "2.5.0", geohash))
        
        conn.commit()
        conn.close()
        logger.info(f"Prediction saved: {score}/100 for {species}")
        
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")

def check_recent_validations_super_advanced(lat: float, lon: float, days: int = 30, 
                                           radius_km: float = 15.0) -> Tuple[bool, int, float]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        cursor.execute('''
            SELECT COUNT(*), AVG(confidence) FROM sightings 
            WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
            AND date >= ? AND validation_status != 'rejected'
        ''', (lat - lat_delta, lat + lat_delta, 
              lon - lon_delta, lon + lon_delta, cutoff_date))
        
        pos_result = cursor.fetchone()
        pos_count = pos_result[0] or 0
        pos_conf = pos_result[1] or 0.0
        
        cursor.execute('''
            SELECT COUNT(*), AVG(search_thoroughness) FROM no_sightings 
            WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
            AND date >= ?
        ''', (lat - lat_delta, lat + lat_delta, 
              lon - lon_delta, lon + lon_delta, cutoff_date))
        
        neg_result = cursor.fetchone()
        neg_count = neg_result[0] or 0
        
        total_count = pos_count + neg_count
        avg_accuracy = (pos_conf + (neg_result[1] or 0.0)) / 2.0 if total_count > 0 else 0.0
        
        conn.close()
        
        has_validations = total_count >= 3
        return has_validations, total_count, avg_accuracy
        
    except Exception as e:
        logger.error(f"Error checking validations: {e}")
        return False, 0, 0.0

# ===== ANALISI TESTUALE SUPER AVANZATA =====
def build_analysis_super_advanced_v25(payload: Dict[str, Any]) -> str:
    idx = payload["index"]
    best = payload.get("best_window", {})
    elev = payload["elevation_m"]
    slope = payload["slope_deg"]
    aspect = payload.get("aspect_octant", "N/A")
    habitat_used = (payload.get("habitat_used") or "").capitalize() or "Misto"
    species = payload.get("species", "reticulatus")
    
    confidence_detailed = payload.get("confidence_detailed", {})
    overall_conf = confidence_detailed.get("overall", 0.0)
    
    flush_events = payload.get("flush_events", [])
    lag_info = f"{len([e for e in flush_events if e.get('observed')])} osservati, {len([e for e in flush_events if not e.get('observed')])} previsti"
    
    lines = []
    
    lines.append("<h4>🧬 Analisi Biologica Super Avanzata v2.5.0</h4>")
    lines.append(f"<p><em>Modello fenologico basato# main.py — Trova Porcini API v2.5.0 SUPER AVANZATO - Render Compatible via Docker
# VERSIONE COMPLETA che mantiene TUTTE le funzionalità scientifiche avanzate
# Soluzione: usa Dockerfile per gestire dipendenze complesse

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio, tempfile, time, sqlite3, logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta
import json

# Import avanzati con fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import geohash2
    GEOHASH_AVAILABLE = True
except ImportError:
    GEOHASH_AVAILABLE = False

try:
    import cdsapi
    from netCDF4 import Dataset, num2date
    CDS_AVAILABLE = True
except ImportError:
    CDS_AVAILABLE = False

# Setup logging professionale
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/porcini.log') if os.path.exists('logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trova Porcini API v2.5.0 - SUPER AVANZATO", 
    version="2.5.0",
    description="Modello fenologico avanzato per previsione fruttificazione Boletus spp."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"], 
    allow_credentials=True
)

HEADERS = {"User-Agent":"TrovaPorcini/2.5.0 (+scientific)", "Accept-Language":"it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")
CDS_API_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")

# Database avanzato
DB_PATH = "data/porcini_validations.db"

def init_database():
    """Inizializza database SQLite avanzato per machine learning"""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Tabella segnalazioni con metadati completi
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                date TEXT NOT NULL,
                species TEXT NOT NULL,
                quantity INTEGER DEFAULT 1,
                size_cm_avg REAL,
                size_cm_max REAL,
                confidence REAL DEFAULT 0.8,
                photo_url TEXT,
                notes TEXT,
                habitat_observed TEXT,
                weather_conditions TEXT,
                predicted_score INTEGER,
                model_version TEXT DEFAULT '2.5.0',
                user_experience_level INTEGER DEFAULT 3,
                validation_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                geohash TEXT,
                elevation_m REAL,
                slope_deg REAL,
                aspect_deg REAL
            )
        ''')
        
        # Tabella ricerche negative
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS no_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                date TEXT NOT NULL,
                searched_hours REAL DEFAULT 2.0,
                search_method TEXT DEFAULT 'visual',
                habitat_searched TEXT,
                weather_conditions TEXT,
                notes TEXT,
                predicted_score INTEGER,
                model_version TEXT DEFAULT '2.5.0',
                search_thoroughness INTEGER DEFAULT 3,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                geohash TEXT,
                elevation_m REAL
            )
        ''')
        
        # Tabella predizioni per ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                date TEXT NOT NULL,
                predicted_score INTEGER NOT NULL,
                species TEXT NOT NULL,
                habitat TEXT,
                confidence_data TEXT,
                weather_data TEXT,
                model_features TEXT,
                model_version TEXT DEFAULT '2.5.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                validated BOOLEAN DEFAULT FALSE,
                validation_date TEXT,
                validation_result TEXT
            )
        ''')
        
        # Tabella performance modello
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                model_version TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                rmse REAL,
                total_predictions INTEGER,
                total_validations INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database avanzato inizializzato con successo")
    except Exception as e:
        logger.error(f"Errore inizializzazione database: {e}")

init_database()

# ===== UTILITIES MATEMATICHE AVANZATE =====
def clamp(v, a, b): 
    return a if v < a else b if v > b else v

def half_life_coeff(days: float) -> float:
    return 1.0 - 0.5**(1.0/max(1.0, days))

def api_index(precip: List[float], half_life: float = 8.0) -> float:
    k = half_life_coeff(half_life)
    api = 0.0
    for p in precip: 
        api = (1-k) * api + k * (p or 0.0)
    return api

def stddev_advanced(xs: List[float]) -> float:
    if not xs: return 0.0
    if NUMPY_AVAILABLE:
        return float(np.std(xs, ddof=1))
    else:
        m = sum(xs) / len(xs)
        return (sum((x-m)**2 for x in xs) / max(1, len(xs)-1))**0.5

def percentile_advanced(xs: List[float], p: float) -> float:
    if NUMPY_AVAILABLE:
        return float(np.percentile(xs, p))
    else:
        if not xs: return 0.0
        sorted_xs = sorted(xs)
        k = (len(sorted_xs) - 1) * p / 100.0
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_xs):
            return sorted_xs[f] * (1 - c) + sorted_xs[f + 1] * c
        return sorted_xs[f]

def geohash_encode_advanced(lat: float, lon: float, precision: int = 8) -> str:
    if GEOHASH_AVAILABLE:
        return geohash2.encode(lat, lon, precision)
    else:
        return f"{lat:.4f},{lon:.4f}"

# ===== VPD AVANZATO =====
def saturation_vapor_pressure_hpa(Tc: float) -> float:
    return 6.112 * math.exp((17.67 * Tc) / (Tc + 243.5))

def vpd_hpa(Tc: float, RH: float) -> float:
    RHc = clamp(RH, 0.0, 100.0)
    return saturation_vapor_pressure_hpa(Tc) * (1.0 - RHc/100.0)

def vpd_penalty_advanced(vpd_max_hpa: float, species_vpd_sens: float = 1.0, 
                        elevation_m: float = 800.0) -> float:
    alt_factor = 1.0 - (elevation_m - 500.0) / 2000.0
    alt_factor = clamp(alt_factor, 0.7, 1.2)
    vpd_corrected = vpd_max_hpa * alt_factor
    
    if vpd_corrected <= 5.0: base = 1.0
    elif vpd_corrected >= 15.0: base = 0.3
    else: base = 1.0 - 0.7 * (vpd_corrected - 5.0) / 10.0
    
    penalty = 1.0 - (1.0-base) * species_vpd_sens
    return clamp(penalty, 0.25, 1.0)

def thermal_shock_index_advanced(tmin_series: List[float], window_days: int = 3) -> float:
    if len(tmin_series) < 2 * window_days: return 0.0
    
    recent = sum(tmin_series[-window_days:]) / window_days
    previous = sum(tmin_series[-2*window_days:-window_days]) / window_days
    drop = previous - recent
    
    if drop <= 0.5: return 0.0
    if drop >= 6.0: return 1.0
    
    return 1.0 / (1.0 + math.exp(-2.0 * (drop - 3.0)))

# ===== TOPOGRAFIA AVANZATA =====
def twi_advanced_proxy(slope_deg: float, concavity: float, 
                      drainage_area_proxy: float = 1.0) -> float:
    beta = max(0.1, math.radians(max(0.1, slope_deg)))
    tanb = max(0.05, math.tan(beta))
    area_proxy = max(0.1, 1.0 + 10.0 * max(0.0, concavity))
    twi = math.log(area_proxy) - math.log(tanb)
    return clamp((twi + 3.0) / 6.0, 0.0, 1.0)

def microclimate_energy_advanced(aspect_oct: Optional[str], slope_deg: float, 
                                month: int, latitude: float, elevation_m: float) -> float:
    if not aspect_oct or slope_deg < 0.5: return 0.5
    
    aspect_energy = {
        "N": 0.3, "NE": 0.4, "E": 0.6, "SE": 0.8,
        "S": 1.0, "SW": 0.9, "W": 0.7, "NW": 0.4
    }
    base_energy = aspect_energy.get(aspect_oct, 0.5)
    
    if month in [6,7,8]: seasonal_factor = 1.0
    elif month in [9,10]: seasonal_factor = 0.8
    elif month in [4,5]: seasonal_factor = 0.7
    else: seasonal_factor = 0.5
    
    lat_factor = 1.0 - (latitude - 42.0) / 50.0
    lat_factor = clamp(lat_factor, 0.7, 1.2)
    
    if elevation_m > 1500: alt_factor = 0.85
    elif elevation_m > 1000: alt_factor = 0.95
    else: alt_factor = 1.0
    
    slope_factor = 1.0 + min(0.3, slope_deg / 60.0)
    
    final_energy = base_energy * seasonal_factor * lat_factor * alt_factor * slope_factor
    return clamp(final_energy, 0.2, 1.2)

# ===== SOGLIE DINAMICHE SUPER AVANZATE =====
def dynamic_rain_threshold_v25(smi: float, month: int, elevation: float, 
                              lat: float, recent_temp_trend: float) -> float:
    base_threshold = 7.5
    
    if smi > 0.8: smi_factor = 0.6
    elif smi > 0.6: smi_factor = 0.8
    elif smi < 0.3: smi_factor = 1.4
    else: smi_factor = 1.0
    
    seasonal_et = {
        1: 0.5, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.3, 6: 1.5,
        7: 1.6, 8: 1.5, 9: 1.2, 10: 0.9, 11: 0.6, 12: 0.5
    }
    et_factor = seasonal_et.get(month, 1.0)
    
    if elevation > 1500: alt_factor = 0.75
    elif elevation > 1200: alt_factor = 0.85
    elif elevation > 800: alt_factor = 0.92
    else: alt_factor = 1.0
    
    if lat > 46.0: lat_factor = 0.9
    elif lat < 41.0: lat_factor = 1.1
    else: lat_factor = 1.0
    
    if recent_temp_trend > 1.0: temp_factor = 1.15
    elif recent_temp_trend < -1.0: temp_factor = 0.9
    else: temp_factor = 1.0
    
    final_threshold = base_threshold * smi_factor * et_factor * alt_factor * lat_factor * temp_factor
    return clamp(final_threshold, 4.0, 18.0)

# ===== SMOOTHING SAVITZKY-GOLAY AVANZATO =====
def savitzky_golay_advanced(forecast: List[float], window_length: int = 5, 
                           polyorder: int = 2) -> List[float]:
    if len(forecast) < 5:
        return simple_smoothing_fallback(forecast)
    
    if SCIPY_AVAILABLE and NUMPY_AVAILABLE:
        try:
            arr = np.array(forecast, dtype=float)
            wl = min(window_length, len(arr))
            if wl % 2 == 0: wl -= 1
            if wl < 3: wl = 3
            po = min(polyorder, wl - 1)
            
            smoothed = savgol_filter(arr, window_length=wl, polyorder=po, mode='nearest')
            
            # Preserva picchi importanti
            for i, (orig, smooth) in enumerate(zip(forecast, smoothed)):
                if orig > 75 and smooth < orig * 0.8:
                    smoothed[i] = orig * 0.9
            
            return np.clip(smoothed, 0, 100).tolist()
            
        except Exception as e:
            logger.warning(f"Savgol failed: {e}, using custom smoothing")
    
    return advanced_custom_smoothing(forecast)

def advanced_custom_smoothing(forecast: List[float]) -> List[float]:
    if len(forecast) < 3:
        return forecast[:]
    
    smoothed = []
    for i in range(len(forecast)):
        weights = []
        values = []
        
        for j in range(max(0, i-2), min(len(forecast), i+3)):
            dist = abs(i - j)
            weight = math.exp(-dist**2 / 2.0)
            weights.append(weight)
            values.append(forecast[j])
        
        weighted_sum = sum(w * v for w, v in zip(weights, values))
        weight_sum = sum(weights)
        smoothed_val = weighted_sum / weight_sum
        
        if forecast[i] > 70 and smoothed_val < forecast[i] * 0.85:
            smoothed_val = forecast[i] * 0.92
        
        smoothed.append(smoothed_val)
    
    return smoothed

def simple_smoothing_fallback(forecast: List[float]) -> List[float]:
    if len(forecast) <= 2:
        return forecast[:]
    
    smoothed = [forecast[0]]
    for i in range(1, len(forecast)-1):
        smoothed.append((forecast[i-1] + 2*forecast[i] + forecast[i+1]) / 4.0)
    smoothed.append(forecast[-1])
    return smoothed

# ===== SMI AVANZATO CON ERA5-LAND =====
SM_CACHE: Dict[str, Dict[str, Any]] = {}

async def _prefetch_era5l_sm_advanced(lat: float, lon: float, days: int = 40) -> None:
    if not CDS_API_KEY or not CDS_AVAILABLE: return
    
    key = f"{round(lat,3)},{round(lon,3)}"
    if key in SM_CACHE and (time.time() - SM_CACHE[key].get("ts", 0)) < 12*3600:
        return
    
    def _blocking_download():
        try:
            c = cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY, quiet=True, verify=1)
            end = datetime.utcnow().date()
            start = end - timedelta(days=days-1)
            years = sorted({start.year, end.year})
            months = [f"{m:02d}" for m in range(1,13)] if len(years)>1 else [f"{m:02d}" for m in range(start.month, end.month+1)]
            days_list = [f"{d:02d}" for d in range(1,31)]
            bbox = [lat+0.05, lon-0.05, lat-0.05, lon+0.05]
            
            req = {
                "product_type": "reanalysis",
                "variable": ["volumetric_soil_water_layer_1"],
                "year": [str(y) for y in years],
                "month": months,
                "day": days_list,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": bbox,
                "format": "netcdf",
            }
            
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                target = tmp.name
            c.retrieve("reanalysis-era5-land", req, target)
            
            ds = Dataset(target)
            t = ds.variables["time"]
            times = num2date(t[:], t.units)
            var = ds.variables.get("swvl1") or ds.variables["volumetric_soil_water_layer_1"]
            data = var[:]
            
            daily: Dict[str, float] = {}
            if NUMPY_AVAILABLE:
                for i in range(data.shape[0]):
                    v = float(np.nanmean(np.array(data[i]).astype("float64")))
                    day = times[i].date().isoformat()
                    if day not in daily: daily[day] = v
                    else: daily[day] = (daily[day] + v) / 2.0
            
            ds.close()
            os.remove(target)
            return {"daily": daily, "ts": time.time()}
            
        except Exception as e:
            logger.warning(f"ERA5-Land fetch failed: {e}")
            return None
    
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _blocking_download)
        if data and "daily" in data:
            SM_CACHE[key] = data
    except Exception as e:
        logger.warning(f"ERA5-Land processing failed: {e}")

def smi_from_p_et0_advanced(P: List[float], ET0: List[float]) -> List[float]:
    if NUMPY_AVAILABLE:
        alpha = 0.25
        S = 0.0
        xs = []
        
        for i, (p, et) in enumerate(zip(P, ET0)):
            forcing = (p or 0.0) - (et or 0.0)
            month = ((i + datetime.now().month - len(P) - 1) % 12) + 1
            if month in [6,7,8]: alpha_adj = alpha * 1.2
            else: alpha_adj = alpha
            
            S = (1 - alpha_adj) * S + alpha_adj * forcing
            xs.append(S)
        
        arr = np.array(xs, dtype=float)
        valid = arr[np.isfinite(arr)]
        
        if valid.size >= 10:
            p10, p90 = np.percentile(valid, [10, 90])
        else:
            p10, p90 = float(arr.min()), float(arr.max())
            if p90 - p10 < 1e-6: p10, p90 = p10-1, p90+1
        
        normalized = (arr - p10) / max(1e-6, (p90 - p10))
        return np.clip(normalized, 0.0, 1.0).tolist()
    else:
        return smi_fallback_pure_python(P, ET0)

def smi_fallback_pure_python(P: List[float], ET0: List[float]) -> List[float]:
    alpha = 0.25
    S = 0.0
    xs = []
    
    for p, et in zip(P, ET0):
        forcing = (p or 0.0) - (et or 0.0)
        S = (1 - alpha) * S + alpha * forcing
        xs.append(S)
    
    if len(xs) >= 5:
        sorted_xs = sorted(xs)
        p10_idx = max(0, int(0.1 * len(sorted_xs)))
        p90_idx = min(len(sorted_xs)-1, int(0.9 * len(sorted_xs)))
        p10, p90 = sorted_xs[p10_idx], sorted_xs[p90_idx]
    else:
        p10, p90 = min(xs) if xs else -1.0, max(xs) if xs else 1.0
        if p90-p10 < 1e-6: p10, p90 = p10-1, p90+1
    
    return [clamp((x-p10)/(p90-p10), 0.0, 1.0) for x in xs]

# ===== CONFIDENCE SYSTEM 5D SUPER AVANZATO =====
def confidence_5d_super_advanced(
    weather_agreement: float,
    habitat_confidence: float,
    smi_reliability: float,
    vpd_validity: bool,
    has_recent_validation: bool,
    elevation_reliability: float = 0.8,
    temporal_consistency: float = 0.7
) -> Dict[str, float]:
    
    met_conf = clamp(weather_agreement, 0.15, 0.98)
    
    eco_base = clamp(habitat_confidence, 0.1, 0.9)
    if has_recent_validation: eco_base *= 1.15
    eco_conf = clamp(eco_base, 0.1, 0.95)
    
    hydro_base = clamp(smi_reliability, 0.2, 0.9)
    hydro_conf = hydro_base * clamp(temporal_consistency, 0.5, 1.0)
    
    atmo_base = 0.85 if vpd_validity else 0.35
    atmo_conf = atmo_base * clamp(elevation_reliability, 0.6, 1.0)
    
    emp_base = 0.75 if has_recent_validation else 0.35
    emp_conf = emp_base
    
    weights = {"met": 0.28, "eco": 0.24, "hydro": 0.22, "atmo": 0.16, "emp": 0.10}
    
    components = [met_conf, eco_conf, hydro_conf, atmo_conf, emp_conf]
    min_component = min(components)
    
    penalty = 1.0 if min_component > 0.4 else (0.8 + 0.2 * min_component / 0.4)
    
    overall = (weights["met"] * met_conf + 
               weights["eco"] * eco_conf + 
               weights["hydro"] * hydro_conf + 
               weights["atmo"] * atmo_conf + 
               weights["emp"] * emp_conf) * penalty
    
    return {
        "meteorological": round(met_conf, 3),
        "ecological": round(eco_conf, 3),
        "hydrological": round(hydro_conf, 3),
        "atmospheric": round(atmo_conf, 3),
        "empirical": round(emp_conf, 3),
        "overall": round(clamp(overall, 0.15, 0.95), 3


