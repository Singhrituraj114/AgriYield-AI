FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/  ./backend/
COPY frontend/ ./frontend/
COPY crop_production.csv .

# Model files are copied by huggingface_hub download script at build time
# (uploaded directly to the HF Space repo via web UI — see README)

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Start FastAPI — serves frontend at / and API at /predict etc.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
