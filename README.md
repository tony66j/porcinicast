# PorciniCast – MVP completo

Nowcasting/forecast dell'idoneità per porcini (Boletus edulis s.l.) dato un punto (lat/lon).
Questa MVP include:
- **Backend FastAPI** con endpoint per geocoding proxy, score (oggi + D+10), estrazione quota, stima **pendenza/aspect** (da campionamento locale di elevazione), e inferenza **boschi/essenze** via OSM (Overpass) con fallback euristico per quota/latitudine.
- **Frontend** statico che interroga il backend e visualizza indice, finestra migliore, specie e grafici.
- **CORS** abilitato per test locale.

> Nota: il backend chiama servizi pubblici (Nominatim, Open‑Meteo, Open‑Elevation, Overpass). È buona norma inserire una `User-Agent` personalizzata e rispettare i limiti (rate-limit) dei provider.

## Requisiti

- Python 3.10+
- pip


## Installazione

```bash
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Avvio Backend (sviluppo)

```bash
uvicorn backend.main:app --reload --port 8787
```

Il backend espone le API su `http://127.0.0.1:8787`.

## Avvio Frontend (sviluppo)

Apri `frontend/index.html` direttamente nel browser **oppure** usa un server statico:

```bash
python -m http.server 5500 -d frontend
```

Poi visita: `http://localhost:5500/`

> Se apri il file direttamente (file://), alcuni browser potrebbero limitare fetch; usa il piccolo server statico per un'esperienza uniforme.


## Endpoints principali

- `GET /api/geocode?q=...` → proxy Nominatim, restituisce `{lat, lon, display}`
- `GET /api/score?lat=..&lon=..` → calcola score **oggi** e **D+10** con breakdown e suggerimenti
- `GET /api/forest?lat=..&lon=..&alt=..` → inferisce essenze dominate (OSM + fallback altimetrico)

## Come funziona lo score (sintesi)

L'indice (0–100) combina:
- `P14` (pioggia ultimi 14 gg), `Tmean7`, finestra altimetrica/termica stagionale,
- pendenza/aspect locali, compatibilità bosco-porcini (ectomicorrize),
- proxy suolo/vegetazione (placeholder costanti nell’MVP).

Il backend restituisce anche un breakdown per trasparenza.

## TODO (post‑MVP)

- Raster DEM/land cover locali (Copernicus DEM, CLC) con PostGIS per calcolo massivo e tiles.
- Calcolo griglia heatmap reale e caching.
- Modello ML supervisionato (quando avremo labels di ritrovamento).
- Gestione quote/regole regionali (permesse raccolta, aree protette).

Buone prove! 🍄
