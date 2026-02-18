from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import shap
import os

app = FastAPI(title="AgriYield AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL     = joblib.load(os.path.join(BASE, "agri_yield_model.pkl"))
LE_STATE  = joblib.load(os.path.join(BASE, "state_encoder.pkl"))
LE_SEASON = joblib.load(os.path.join(BASE, "season_encoder.pkl"))
LE_CROP   = joblib.load(os.path.join(BASE, "crop_encoder.pkl"))

DF = pd.read_csv(os.path.join(BASE, "crop_production.csv"))
DF = DF.dropna(subset=["Area", "Production"])
DF = DF[DF["Area"] > 0]
DF["Yield"] = DF["Production"] / DF["Area"]

# No background data — prevents OOM/segfault on large RandomForest models
EXPLAINER = shap.TreeExplainer(MODEL)


# ── Schemas ──
class PredictReq(BaseModel):
    state: str
    season: str
    crop: str
    year: int
    area: float

class ForecastReq(BaseModel):
    state: str
    season: str
    crop: str
    area: float
    start_year: int
    n: int = 5


# ── Helper ──
FEATURE_COLS    = ["State_Name", "Crop_Year", "Season", "Crop", "Area"]
FEATURE_DISPLAY = ["State",      "Year",      "Season", "Crop", "Area"]

def encode(state, season, year, crop, area):
    return pd.DataFrame([[
        LE_STATE.transform([state])[0],
        year,
        LE_SEASON.transform([season])[0],
        LE_CROP.transform([crop])[0],
        area
    ]], columns=FEATURE_COLS)


# ────────────── ROUTES ──────────────

@app.get("/")
def root():
    return {"status": "AgriYield AI API running", "version": "2.0"}


@app.get("/meta")
def meta():
    return {
        "states":   sorted(LE_STATE.classes_.tolist()),
        "seasons":  sorted(LE_SEASON.classes_.tolist()),
        "crops":    sorted(LE_CROP.classes_.tolist()),
        "year_min": int(DF["Crop_Year"].min()),
        "year_max": 2035,
    }


@app.get("/stats")
def stats():
    return {
        "total_records": len(DF),
        "total_states":  int(DF["State_Name"].nunique()),
        "total_crops":   int(DF["Crop"].nunique()),
        "total_seasons": int(DF["Season"].nunique()),
        "year_range":    f"{int(DF['Crop_Year'].min())} \u2013 {int(DF['Crop_Year'].max())}",
        "avg_yield":     round(float(DF["Yield"].mean()), 4),
        "max_yield":     round(float(DF["Yield"].clip(upper=DF["Yield"].quantile(0.99)).max()), 4),
    }


@app.post("/predict")
def predict(req: PredictReq):
    try:
        X   = encode(req.state, req.season, req.year, req.crop, req.area)
        yld = float(MODEL.predict(X)[0])
        # Confidence interval from individual RF trees
        tree_preds = np.array([float(t.predict(X)[0]) for t in MODEL.estimators_])
        return {
            "yield":      round(yld, 4),
            "production": round(yld * req.area, 4),
            "conf_low":   round(float(np.percentile(tree_preds, 10)), 4),
            "conf_high":  round(float(np.percentile(tree_preds, 90)), 4),
            "conf_std":   round(float(np.std(tree_preds)), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/forecast")
def forecast(req: ForecastReq):
    try:
        years = list(range(req.start_year, req.start_year + req.n))
        X = pd.concat(
            [encode(req.state, req.season, y, req.crop, req.area) for y in years],
            ignore_index=True
        )
        preds = MODEL.predict(X).tolist()
        return {"years": years, "yields": [round(p, 4) for p in preds]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/shap")
def shap_explain(req: PredictReq):
    try:
        X  = encode(req.state, req.season, req.year, req.crop, req.area)
        sv = EXPLAINER.shap_values(X, check_additivity=False)
        # sv shape: (n_samples, n_features) — grab first row
        vals = sv[0] if hasattr(sv, 'shape') and len(sv.shape) == 2 else sv
        base = float(np.array(EXPLAINER.expected_value).ravel()[0])
        return {
            "shap_values":  [round(float(v), 6) for v in vals],
            "base_value":   round(base, 6),
            "input_values": X.values[0].tolist(),
            "features":     ["State", "Year", "Season", "Crop", "Area"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics/top-crops")
def top_crops(n: int = 10):
    data = DF.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(n)
    return {"crops": data.index.tolist(), "yields": [round(v, 4) for v in data.values]}


@app.get("/analytics/season-distribution")
def season_dist():
    data = DF.groupby("Season")["Yield"].agg(["mean", "median", "std"]).reset_index()
    data.columns = ["season", "mean", "median", "std"]
    return data.fillna(0).round(4).to_dict(orient="records")


@app.get("/analytics/yearly-trend")
def yearly_trend():
    data = DF.groupby("Crop_Year")["Yield"].mean().reset_index()
    return {
        "years":  data["Crop_Year"].tolist(),
        "yields": [round(v, 4) for v in data["Yield"].tolist()],
    }


@app.get("/analytics/state-heatmap")
def state_heatmap():
    data = DF.groupby("State_Name")["Yield"].mean().sort_values(ascending=False).reset_index()
    return {
        "states": data["State_Name"].tolist(),
        "yields": [round(v, 4) for v in data["Yield"].tolist()],
    }


@app.get("/data/sample")
def data_sample(n: int = Query(default=200, le=500)):
    sample = DF.sample(min(n, len(DF)), random_state=None).copy()
    sample = sample.replace({np.nan: None})
    return sample.to_dict(orient="records")


@app.get("/analytics/state-geo")
def state_geo():
    """Returns avg yield per state for choropleth map."""
    data = DF.groupby("State_Name")["Yield"].mean().round(4)
    return {str(k): float(v) for k, v in data.items()}


@app.get("/analytics/by-state")
def by_state(state: str = Query(...)):
    """Contextual analytics filtered by state."""
    sdf = DF[DF["State_Name"] == state]
    if sdf.empty:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found")

    top_crops = (sdf.groupby("Crop")["Yield"].mean()
                   .sort_values(ascending=False).head(8))
    season_data = (sdf.groupby("Season")["Yield"]
                     .agg(["mean", "median"]).reset_index()
                     .rename(columns={"Season": "season"}))
    yearly = sdf.groupby("Crop_Year")["Yield"].mean().reset_index()

    return {
        "top_crops": {
            "crops":  top_crops.index.tolist(),
            "yields": [round(v, 4) for v in top_crops.values],
        },
        "season_dist": season_data.fillna(0).round(4).to_dict(orient="records"),
        "yearly_trend": {
            "years":  yearly["Crop_Year"].tolist(),
            "yields": [round(v, 4) for v in yearly["Yield"].tolist()],
        },
        "total_records": len(sdf),
        "avg_yield":     round(float(sdf["Yield"].mean()), 4),
        "top_crop":      top_crops.index[0] if len(top_crops) else "N/A",
    }


@app.get("/analytics/by-crop")
def by_crop(crop: str = Query(...), state: str = Query(default=None)):
    """Contextual analytics filtered by crop (optionally scoped to state)."""
    cdf = DF[DF["Crop"] == crop]
    if state:
        cdf = cdf[cdf["State_Name"] == state]
    if cdf.empty:
        raise HTTPException(status_code=404, detail="No data for given crop/state combo")

    yearly = cdf.groupby("Crop_Year")["Yield"].mean().reset_index()
    season_data = (cdf.groupby("Season")["Yield"]
                      .agg(["mean", "median"]).reset_index()
                      .rename(columns={"Season": "season"}))
    top_states = (cdf.groupby("State_Name")["Yield"].mean()
                     .sort_values(ascending=False).head(8))

    return {
        "yearly_trend": {
            "years":  yearly["Crop_Year"].tolist(),
            "yields": [round(v, 4) for v in yearly["Yield"].tolist()],
        },
        "season_dist": season_data.fillna(0).round(4).to_dict(orient="records"),
        "top_states": {
            "states": top_states.index.tolist(),
            "yields": [round(v, 4) for v in top_states.values],
        },
        "avg_yield":     round(float(cdf["Yield"].mean()), 4),
        "total_records": len(cdf),
    }


@app.get("/recommend")
def recommend(
    state: str = Query(...),
    season: str = Query(...),
    year: int  = Query(...),
    area: float = Query(...),
    n: int = 8,
):
    """Batch-predict all crops at once and rank by yield — much faster than one-by-one."""
    rows, valid_crops = [], []
    for crop in LE_CROP.classes_:
        try:
            rows.append([
                LE_STATE.transform([state])[0],
                year,
                LE_SEASON.transform([season])[0],
                LE_CROP.transform([crop])[0],
                area,
            ])
            valid_crops.append(crop)
        except Exception:
            pass
    if not rows:
        raise HTTPException(status_code=400, detail="No valid crops found for given inputs")
    X_all  = pd.DataFrame(rows, columns=FEATURE_COLS)
    preds  = MODEL.predict(X_all)
    results = [
        {"crop": c.strip(), "yield": round(float(p), 4), "production": round(float(p) * area, 4)}
        for c, p in zip(valid_crops, preds)
    ]
    results.sort(key=lambda x: x["yield"], reverse=True)
    return {"recommendations": results[:n], "state": state, "season": season.strip(), "year": year}


class BatchPredictReq(BaseModel):
    inputs: List[PredictReq]


@app.post("/batch-predict")
def batch_predict(req: BatchPredictReq):
    """Predict yield for a list of inputs at once."""
    rows, meta = [], []
    for item in req.inputs:
        try:
            rows.append([
                LE_STATE.transform([item.state])[0],
                item.year,
                LE_SEASON.transform([item.season])[0],
                LE_CROP.transform([item.crop])[0],
                item.area,
            ])
            meta.append(item)
        except Exception as e:
            meta.append({"error": str(e)})
            rows.append([0, 0, 0, 0, 0])  # placeholder
    X_all = pd.DataFrame(rows, columns=FEATURE_COLS)
    preds = MODEL.predict(X_all)
    results = []
    for i, item in enumerate(meta):
        if isinstance(item, dict) and "error" in item:
            results.append(item)
        else:
            yld = float(preds[i])
            results.append({
                "state": item.state, "season": item.season.strip(),
                "crop": item.crop, "year": item.year, "area": item.area,
                "yield": round(yld, 4), "production": round(yld * item.area, 4),
            })
    return {"results": results, "count": len(results)}


@app.get("/crop-compare")
def crop_compare(
    state:  str   = Query(...),
    season: str   = Query(...),
    year:   int   = Query(...),
    area:   float = Query(...),
    crops:  str   = Query(...),   # comma-separated crop names
):
    """Compare predicted yield across multiple crops."""
    crop_list = [c.strip() for c in crops.split(",")]
    rows, valid, errors = [], [], []
    for crop in crop_list:
        matched = next((c for c in LE_CROP.classes_ if c.strip() == crop), None)
        if not matched:
            errors.append({"crop": crop, "error": f"'{crop}' not in model"})
            continue
        try:
            rows.append([
                LE_STATE.transform([state])[0],
                year,
                LE_SEASON.transform([season])[0],
                LE_CROP.transform([matched])[0],
                area,
            ])
            valid.append((crop, matched))
        except Exception as e:
            errors.append({"crop": crop, "error": str(e)})
    results = list(errors)
    if rows:
        X_all = pd.DataFrame(rows, columns=FEATURE_COLS)
        preds = MODEL.predict(X_all)
        for (crop_stripped, matched), yld in zip(valid, preds):
            hist = DF[DF["Crop"] == matched][DF["State_Name"] == state]["Yield"]
            results.append({
                "crop": crop_stripped,
                "yield": round(float(yld), 4),
                "production": round(float(yld) * area, 4),
                "historical_avg": round(float(hist.mean()), 4) if not hist.empty else None,
            })
    return {"state": state, "season": season.strip(), "year": year, "area": area, "comparison": results}
