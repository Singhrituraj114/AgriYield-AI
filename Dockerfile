FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (no pkl — downloaded at runtime from HF model repo)
COPY backend/  ./backend/
COPY frontend/ ./frontend/
COPY crop_production.csv .

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Download model files from HF, then start API server
CMD ["sh", "-c", "python backend/download_models.py && uvicorn backend.main:app --host 0.0.0.0 --port 7860"]
