import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ───────────────── PAGE CONFIG ─────────────────
st.set_page_config(
    page_title="AgriYield AI",
    page_icon="🌾",
    layout="wide"
)

# ───────────────── UI STYLE ────────────────────
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.glass {
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
}
</style>
""", unsafe_allow_html=True)

# ───────────────── LOAD ASSETS ─────────────────
@st.cache_resource
def load_assets():
    model = joblib.load("agri_yield_model.pkl")
    le_state = joblib.load("state_encoder.pkl")
    le_season = joblib.load("season_encoder.pkl")
    le_crop = joblib.load("crop_encoder.pkl")
    return model, le_state, le_season, le_crop

@st.cache_data
def load_data():
    df = pd.read_csv("crop_production.csv")
    df = df.dropna(subset=["Area", "Production"])
    df = df[df["Area"] > 0]
    df["Yield"] = df["Production"] / df["Area"]
    return df

MODEL, LE_STATE, LE_SEASON, LE_CROP = load_assets()
DF = load_data()

# ───────────────── ML CORE ─────────────────────
def encode(state, season, year, crop, area):
    return np.array([[
        LE_STATE.transform([state])[0],
        year,
        LE_SEASON.transform([season])[0],
        LE_CROP.transform([crop])[0],
        area
    ]])

def predict(state, season, year, crop, area):
    yld = MODEL.predict(encode(state, season, year, crop, area))[0]
    return yld, yld * area

def forecast(state, season, crop, area, start_year, n=5):
    years = np.arange(start_year, start_year + n)
    X = np.vstack([
        encode(state, season, y, crop, area)[0]
        for y in years
    ])
    return years, MODEL.predict(X)

# ───────────────── SHAP SETUP ──────────────────
@st.cache_resource
def load_explainer(model, df):
    X_bg = df[["State_Name", "Crop_Year", "Season", "Crop", "Area"]].sample(
        300, random_state=42
    )
    X_bg["State_Name"] = LE_STATE.transform(X_bg["State_Name"])
    X_bg["Season"] = LE_SEASON.transform(X_bg["Season"])
    X_bg["Crop"] = LE_CROP.transform(X_bg["Crop"])
    return shap.TreeExplainer(model, X_bg)

EXPLAINER = load_explainer(MODEL, DF)

# ───────────────── HEADER ──────────────────────
st.title("🌾 AgriYield AI")
st.caption("Interactive Crop Yield Prediction with Explainable AI (SHAP)")

# ───────────────── SIDEBAR INPUT ───────────────
st.sidebar.header("🌱 Input Parameters")

state = st.sidebar.selectbox("State", LE_STATE.classes_)
season = st.sidebar.selectbox("Season", LE_SEASON.classes_)
crop = st.sidebar.selectbox("Crop", LE_CROP.classes_)
year = st.sidebar.slider("Crop Year", 2000, 2035, 2025)
area = st.sidebar.slider("Area (Hectares)", 0.1, 1000.0, 10.0)

run = st.sidebar.button("🔮 Predict")

# ───────────────── PREDICTION SECTION ──────────
if run:
    yld, prod = predict(state, season, year, crop, area)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.metric("🌾 Yield (T/Ha)", f"{yld:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.metric("📦 Production (Tonnes)", f"{prod:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    years, preds = forecast(state, season, crop, area, year)

    fig = go.Figure(go.Scatter(
        x=years, y=preds,
        mode="lines+markers",
        line=dict(width=3)
    ))
    fig.update_layout(
        height=300,
        xaxis_title="Year",
        yaxis_title="Yield (T/Ha)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ───────────────── EXPLAINABILITY ──────────────
st.divider()
st.subheader("🧠 Model Explainability (SHAP)")

X_single = encode(state, season, year, crop, area)
shap_values = EXPLAINER.shap_values(X_single)

fig, ax = plt.subplots()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=EXPLAINER.expected_value,
        data=X_single[0],
        feature_names=["State", "Year", "Season", "Crop", "Area"]
    ),
    show=False
)
st.pyplot(fig)

# ───────────────── GLOBAL ANALYTICS ────────────
st.divider()
st.subheader("📊 Global Analytics")

col1, col2 = st.columns(2)

with col1:
    top_crops = DF.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10)
    fig = px.bar(
        top_crops,
        orientation="h",
        labels={"value": "Yield (T/Ha)", "index": "Crop"},
        title="Top Crops by Average Yield"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        DF,
        x="Season",
        y="Yield",
        title="Yield Distribution by Season"
    )
    st.plotly_chart(fig, use_container_width=True)

# ───────────────── DATA VIEW ───────────────────
with st.expander("🔍 View Sample Data"):
    st.dataframe(DF.sample(100), use_container_width=True)

# ───────────────── FOOTER ──────────────────────
st.caption(
    "Model: RandomForestRegressor | Target: Yield (T/Ha) | Explainability: SHAP"
)
