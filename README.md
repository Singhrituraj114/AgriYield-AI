# 🌾 AgriYield AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-2.0-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/ML-RandomForest-green?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Frontend-Three.js%20%7C%20GSAP%20%7C%20Leaflet-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

> **AI-powered crop yield prediction for Indian agriculture** — combining a trained Random Forest model with an interactive 3D web interface, real-time analytics, India choropleth maps, and SHAP explainability.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 **Yield Prediction** | Predicts crop yield (kg/ha) using a pre-trained Random Forest model |
| 📈 **5-Year Forecast** | Projects yield trends for the next 5 years |
| 🧠 **SHAP Explainability** | Shows which input features drive each prediction |
| 🗺️ **India Choropleth Map** | Interactive Leaflet.js map — hover states to see avg yield |
| 📊 **Analytics Dashboard** | Top crops, season distribution, yearly trends, state heatmaps |
| 🔍 **Contextual Analysis** | After each prediction, auto-renders analytics filtered to your chosen state & crop |
| 🌌 **3D Animated UI** | Three.js particle universe, GSAP scroll animations, glassmorphism design |
| ⚡ **REST API** | 13 FastAPI endpoints with full Swagger docs at `/docs` |

---

## 🏗️ Architecture

```
AgriYieldAI/
├── backend/
│   └── main.py              # FastAPI application (13 endpoints)
├── frontend/
│   └── index.html           # Single-page app (Three.js, GSAP, Chart.js, Leaflet)
├── app.py                   # Original Streamlit app (legacy)
├── agri_yield_model.pkl     # Trained RandomForestRegressor
├── state_encoder.pkl        # LabelEncoder for states
├── season_encoder.pkl       # LabelEncoder for seasons
├── crop_encoder.pkl         # LabelEncoder for crops
├── crop_production.csv      # Dataset: 246,093 rows, India crop data
└── requirements.txt
```

**Stack:**
- **Backend**: FastAPI, Uvicorn, scikit-learn, SHAP, pandas, joblib
- **Frontend**: Three.js r128, GSAP 3.12.2, Chart.js 4.4.0, Leaflet.js 1.9.4
- **ML Model**: RandomForestRegressor (pre-trained, no retraining needed)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repo

```bash
git clone https://github.com/Singhrituraj114/AgriYieldAI.git
cd AgriYieldAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### ⚠️ Model Files (not in repo — too large for GitHub)

The trained `.pkl` files are excluded from this repository. You need to either:

**Option A — Train the model yourself:**
```bash
python app.py   # original Streamlit app that trains and saves the model
```

**Option B — Download pre-trained files** and place them in the project root:
```
AgriYieldAI/
├── agri_yield_model.pkl
├── state_encoder.pkl
├── season_encoder.pkl
└── crop_encoder.pkl
```

### 3. Start the backend

```bash
uvicorn backend.main:app --reload --reload-dir backend --port 8000
```

### 4. Start the frontend server

```bash
cd frontend
python -m http.server 3000
```

### 5. Open in browser

| Service | URL |
|---|---|
| 🌐 Frontend | http://localhost:3000 |
| ⚡ API (Swagger UI) | http://localhost:8000/docs |
| 🔗 API Root | http://localhost:8000 |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/meta` | Available states, seasons, crops |
| GET | `/stats` | Dataset statistics |
| GET | `/data/sample` | Sample rows from dataset |
| POST | `/predict` | **Predict yield for given inputs** |
| POST | `/forecast` | 5-year yield forecast |
| POST | `/shap` | SHAP feature importance values |
| GET | `/analytics/top-crops` | Top crops by avg yield |
| GET | `/analytics/season-distribution` | Yield by season |
| GET | `/analytics/yearly-trend` | Year-over-year yield trend |
| GET | `/analytics/state-heatmap` | Avg yield per state |
| GET | `/analytics/state-geo` | GeoJSON-ready state yields (for map) |
| GET | `/analytics/by-state?state=X` | Full analytics for a specific state |
| GET | `/analytics/by-crop?crop=X&state=Y` | Full analytics for a specific crop |

### Example — Predict Yield

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Andhra Pradesh",
    "season": "Kharif     ",
    "crop": "Rice",
    "year": 2024,
    "area": 1000
  }'
```

**Response:**
```json
{
  "yield_per_hectare": 2134.56,
  "estimated_production": 2134560.0,
  "unit": "kg/ha"
}
```

---

## 📊 Dataset

`crop_production.csv` — India crop production data

| Column | Description |
|---|---|
| `State_Name` | Indian state |
| `District_Name` | District within state |
| `Crop_Year` | Year of harvest |
| `Season` | Kharif / Rabi / Whole Year / etc. |
| `Crop` | Crop variety |
| `Area` | Area under cultivation (hectares) |
| `Production` | Total production (kg) |

- **Rows**: 246,093
- **States**: 33
- **Crops**: 124+
- **Years**: 1997–2015

---

## 🧠 Model Details

- **Algorithm**: RandomForestRegressor
- **Target**: `Yield = Production / Area` (kg/ha)
- **Features**: State, Crop Year, Season, Crop, Area
- **Encoders**: LabelEncoders for categorical features (state, season, crop)
- **Explainability**: SHAP TreeExplainer

---

## 📸 UI Preview

The frontend features:
- **3D particle universe** background (Three.js, 4000 particles)
- **Animated stat counters** — total records, states, crops, avg yield
- **Prediction panel** with instant results
- **Analytics tabs** — Top Crops, Season Distribution, Yearly Trend, Heatmap, SHAP, India Map
- **Contextual Analysis** — auto-updates after every prediction with filtered charts

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Made with ❤️ for Indian agriculture by <a href="https://github.com/Singhrituraj114">Singhrituraj114</a></p>
