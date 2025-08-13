# Dockerfile per Trova Porcini API v2.5.0 SUPER AVANZATO
# Risolve problemi Render mantenendo TUTTE le funzionalità avanzate

FROM python:3.11-slim

# Variabili ambiente ottimizzate
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Aggiorna sistema e installa dipendenze di sistema per scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Aggiorna pip e setuptools alle versioni più recenti
RUN pip install --upgrade pip setuptools wheel

# Installa prima le dipendenze scientifiche con pre-compiled wheels
RUN pip install --only-binary=all numpy==1.24.4
RUN pip install --only-binary=all scipy==1.10.1

# Copia requirements e installa resto delle dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice
COPY main.py .

# Crea directory per database e logs
RUN mkdir -p data logs

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/api/health')" || exit 1

# Usa PORT dinamico di Render
EXPOSE $PORT

# Comando di avvio ottimizzato per produzione
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --loop uvloop --http httptools
