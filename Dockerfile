# Valluvan — Thirukkural Wisdom Assistant
# Single image shared by the `ingest` (one-shot) and `app` (Streamlit) services.
FROM python:3.13-slim

# Keep Python lean and unbuffered so container logs stream in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # fastembed downloads its ONNX models here; mounted as a volume so they
    # persist across container restarts (see docker-compose.yml).
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app

# Install dependencies first for better layer caching. Use the lean runtime
# requirements (dev/eval-only libs like pandas/jupyter are excluded).
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the application (code + the committed, normalized data/thirukkural.json).
COPY . .

EXPOSE 8501

# Default: run the Streamlit chat UI. The `ingest` service overrides this
# command in docker-compose.yml.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
