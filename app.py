import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import xgboost as xgb
from datetime import date, timedelta 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
# --- UPDATED IMPORT ---
from map import create_flood_map, create_static_analysis_plot

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Flash Flood Detection",
    page_icon="🌧",
    layout="wide"
)

# Custom CSS for "Modern Minimalist" Style
st.markdown("""
    <style>
        /* Main background - Softer, cleaner gray-blue tone */
        .stApp {
            background-color: #f0f2f6;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0e1117;
            color: white;
        }
        
        /* Headers - Darker, sharper text */
        h1, h2, h3 {
            color: #1f2937; /* Dark slate gray */
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            font-weight: 700;
        }

        /* NEW CARD STYLE: Solid, Minimalist, Clean */
        .glass-card {
            background-color: #ffffff; /* Solid White */
            border-radius: 12px;
            border: 1px solid #e5e7eb; /* Subtle light gray border */
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* Soft shadow */
            padding: 24px 20px;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        /* Hover Effect: Lift up and show accent border */
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border-color: #3b82f6; /* Blue accent on hover */
        }

        .metric-label {
            color: #6b7280; /* Cool gray text */
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 12px;
            display: flex; /* Ensure icon and text are inline */
            align-items: center;
            gap: 6px;
        }
        
        /* Custom Icons to simulate image icons - using emojis */
        .metric-icon-rain { color: #818cf8; } /* Purple/Blue - For Rainfall */
        .metric-icon-temp { color: #f87171; } /* Red - For Temperature */
        .metric-icon-humidity { color: #3b82f6; } /* Blue - For Humidity */
        .metric-icon-pressure { color: #34d399; } /* Green - For Pressure */
        .metric-icon-wind-speed { color: #f97316; } /* Orange - For Wind Speed */
        .metric-icon-wind-dir { color: #9ca3af; } /* Gray - For Wind Direction */
        .metric-icon-rad { color: #fbbf24; } /* Yellow - For Radiation */


        .metric-value {
            color: #111827; /* Near black for numbers */
            font-size: 32px;
            font-weight: 800;
            line-height: 1;
        }

        .metric-unit {
            color: #9ca3af;
            font-size: 14px;
            font-weight: 500;
            margin-left: 4px;
        }
        
        /* New Style for Weather Snapshot Cards (More distinct, like the image) */
        .weather-card-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .weather-icon-box {
            background-color: #eef2f5; /* Light background for the icon */
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .weather-icon-box .emoji-icon {
            font-size: 20px;
        }
        
        .weather-label-bottom {
            font-size: 11px;
            color: #6b7280;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 5px;
        }

        /* Modern Table Styling */
        .modern-table {
            width: 100%;
            border-collapse: separate; 
            border-spacing: 0;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
            background: white;
            margin-bottom: 20px;
        }
        
        .modern-table thead tr {
            background-color: #f9fafb; /* Very light gray header */
            color: #374151;
            text-align: left;
            font-weight: 600;
        }
        
        .modern-table th {
            padding: 16px 24px;
            border-bottom: 1px solid #e5e7eb;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .modern-table td {
            padding: 16px 24px;
            border-bottom: 1px solid #f3f4f6;
            color: #4b5563;
            font-size: 15px;
        }
        
        .modern-table tbody tr:hover {
            background-color: #f3f4f6;
        }
        
        .modern-table tbody tr:last-child td {
            border-bottom: none;
        }
    </style>
""", unsafe_allow_html=True)

if 'prediction_state' not in st.session_state:
    st.session_state.prediction_state = False

# ---------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Urop_Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data()

def safe_features(data):
    d = data.copy()
    d["rainfall_3_day_sum"] = d["Daily_Rainfall_mm"].rolling(3, min_periods=1).sum()
    d["rainfall_7_day_avg"] = d["Daily_Rainfall_mm"].rolling(7, min_periods=1).mean()
    d["inflow_lag_1"] = d["Prakasam_Barrage_Inflow_cusecs"].shift(1, fill_value=d["Prakasam_Barrage_Inflow_cusecs"].iloc[0])
    d["level_lag_1"] = d["Prakasam_Barrage_Level_ft"].shift(1, fill_value=d["Prakasam_Barrage_Level_ft"].iloc[0])
    return d

df_feat = safe_features(df)

FEATURES = [
    "Daily_Rainfall_mm",
    "Mean_Daily_Temperature_Celsius",
    "Mean_Daily_Relative_Humidity_Percent",
    "Mean_Daily_Surface_Pressure_hPa",
    "Mean_Daily_Wind_Speed_kmh",
    "Mean_Daily_Wind_Direction_Degree",
    "Mean_Daily_Radiation_Wh_m2",
    "Prakasam_Barrage_Level_ft",
    "Prakasam_Barrage_Inflow_cusecs",
    "Prakasam_Barrage_Outflow_cusecs",
    "Prakasam_Barrage_Current_Storage",
    "Prakasam_Barrage_Flood_Cushion",
    "rainfall_3_day_sum",
    "rainfall_7_day_avg",
    "inflow_lag_1",
    "level_lag_1",
]

TIMESTEPS = 10

# ---------------------------------------------------------
# 3. MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("cnn_lstm_scaler.pkl")
    cnn_lstm_heavy = tf.keras.models.load_model("cnn_lstm_Heavy.h5")
    cnn_lstm_medium = tf.keras.models.load_model("cnn_lstm_Medium.h5")
    cnn_lstm_small = tf.keras.models.load_model("cnn_lstm_Small.h5")
    gru_model = tf.keras.models.load_model("flood_prediction_gru_model.h5")
    lstm2_model = tf.keras.models.load_model("flood_prediction_lstm_model_2.h5")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("xgboost_flood_model.json")
    return scaler, cnn_lstm_heavy, cnn_lstm_medium, cnn_lstm_small, gru_model, lstm2_model, xgb_model

scaler, cnn_lstm_heavy, cnn_lstm_medium, cnn_lstm_small, gru_model, lstm2_model, xgb_model = load_models()

# ---------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------
def risk_bucket(v):
    if v <= 0.10: return "No Risk"
    if v <= 0.25: return "Very Low Risk"
    if v <= 0.40: return "Low Risk"
    if v <= 0.60: return "Moderate Risk"
    if v <= 0.75: return "High Risk"
    if v <= 0.90: return "Very High Risk"
    return "Severe / Extreme Risk"

def chip_color(label):
    if label in ["No Risk", "Very Low Risk", "Low Risk"]: return "#10b981" # Green
    if label == "Moderate Risk": return "#f59e0b" # Yellow/Orange
    if label == "High Risk": return "#f97316" # Orange
    if label == "Very High Risk": return "#ef4444" # Red
    return "#b91c1c" # Dark Red

def card_bg(label):
    if label in ["No Risk", "Very Low Risk"]: return "#ecfdf5"
    if label == "Low Risk": return "#d1fae5"
    if label == "Moderate Risk": return "#fef3c7"
    if label == "High Risk": return "#ffedd5"
    if label == "Very High Risk": return "#fee2e2"
    return "#fee2e2"

def risk_chip(label):
    c = chip_color(label)
    return f'<span style="background-color:{c}; padding:6px 14px; border-radius:20px; color:white; font-size:12px; font-weight:700; letter-spacing:0.5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{label}</span>'

def seq_for_date(base_feat_df, date):
    d = pd.to_datetime(date)
    f = base_feat_df.copy()
    if d < f["Date"].min(): return f.iloc[:TIMESTEPS][FEATURES]
    if d > f["Date"].max():
        last = f.iloc[-1:].copy()
        last["Date"] = d
        for col in FEATURES: last[col] *= np.random.uniform(0.95, 1.05)
        f = pd.concat([f, last], ignore_index=True)
        f = safe_features(f)
    if d not in f["Date"].values:
        past = f[f["Date"] <= d]
        idx = past.index.max() if not past.empty else TIMESTEPS - 1
    else: idx = f.index[f["Date"] == d][0]
    start = max(0, idx - (TIMESTEPS - 1))
    block = f.iloc[start:idx + 1][FEATURES]
    if len(block) < TIMESTEPS:
        pad = f.iloc[:TIMESTEPS - len(block)][FEATURES]
        block = pd.concat([pad, block], ignore_index=True)
    return block.iloc[-TIMESTEPS:]

def cnn_seq_prob(seq_df):
    arr = scaler.transform(seq_df.values)
    arr = arr.reshape(1, TIMESTEPS, len(FEATURES))
    p_heavy = float(cnn_lstm_heavy.predict(arr, verbose=0)[0][0])
    p_medium = float(cnn_lstm_medium.predict(arr, verbose=0)[0][0])
    p_small = float(cnn_lstm_small.predict(arr, verbose=0)[0][0])
    p_gru = float(gru_model.predict(arr, verbose=0)[0][0])
    p_lstm2 = float(lstm2_model.predict(arr, verbose=0)[0][0])
    try: p_xgb = float(xgb_model.predict_proba(seq_df.tail(1).values)[0][1])
    except: p_xgb = float(xgb_model.predict(seq_df.tail(1).values)[0])
    return p_heavy, p_medium, p_small, p_gru, p_lstm2, p_xgb

# ---------------------------------------------------------
# 5. MAIN APP UI
# ---------------------------------------------------------
st.title("🌧 AI-Based Flash Flood Detection for Climate-Sensitive Areas")

# --- DATE INPUT with VALIDATION ---
sel_date = st.date_input(
    "Select a Date", 
    value=date(2020, 1, 1),
    min_value=date(2020, 1, 1), 
    max_value=date(2025, 10, 1)
)

if st.button("Predict Flood Risk"):
    st.session_state.prediction_state = True

if st.session_state.prediction_state:
    seq = seq_for_date(df_feat, sel_date)
    p_heavy, p_medium, p_small, p_gru, p_lstm2, p_xgb = cnn_seq_prob(seq)
    ensemble = (p_heavy + p_medium + p_small + p_gru + p_lstm2 + p_xgb) / 6
    
    st.subheader(f"Prediction for {pd.to_datetime(sel_date).date()}")

    # --- HERO CARD ---
    risk_m = risk_bucket(p_medium)
    bg_color = card_bg(risk_m)
    chip_html = risk_chip(risk_m)
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 24px; border-radius: 12px; border: 1px solid rgba(0,0,0,0.05); margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
        <h4 style="margin:0; color:#374151; font-weight:700; font-size: 16px; text-transform: uppercase;">⭐ Best Model (CNN-LSTM Medium)</h4>
        <div style="display:flex; align-items:center; gap: 20px; margin-top:15px;">
            <h1 style="margin:0; font-size: 56px; color:#111827; font-weight: 800;">{p_medium:.2f}</h1>
            <div style="transform: translateY(8px);">{chip_html}</div>
        </div>
        <p style="margin:0; color:#6b7280; font-size:14px; margin-top:8px;">Probability score (0-1)</p>
    </div>
    """, unsafe_allow_html=True)

    # --- MODEL GRID ---
    st.markdown("##### Other Model Predictions")
    def mini_card(name, val):
        risk = risk_bucket(val)
        c = risk_chip(risk)
        st.markdown(f'<div style="background-color: white; border:1px solid #e5e7eb; border-radius:12px; padding:20px; margin-bottom:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);"><div style="font-size:11px; color:#6b7280; font-weight:700; text-transform:uppercase;">{name}</div><div style="font-size:28px; font-weight:700; color:#1f2937; margin: 8px 0;">{val:.2f}</div><div>{c}</div></div>', unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3, gap="medium")
    with r1c1: mini_card("CNN-LSTM Heavy", p_heavy)
    with r1c2: mini_card("CNN-LSTM Small", p_small)
    with r1c3: mini_card("GRU Model", p_gru)
    r2c1, r2c2 = st.columns(2, gap="medium")
    with r2c1: mini_card("LSTM Model 2", p_lstm2)
    with r2c2: mini_card("XGBoost", p_xgb)

    # --- ENSEMBLE ---
    st.markdown("<br>", unsafe_allow_html=True)
    ensemble_risk = risk_bucket(ensemble)
    ensemble_color = chip_color(ensemble_risk)
    st.markdown(f"""
    <div style="background-color: white; padding: 24px; border-radius: 12px; border-left: 6px solid {ensemble_color}; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); display: flex; align-items: center; justify-content: space-between; border: 1px solid #e5e7eb;">
        <div><h3 style="margin:0; color: #1f2937; font-weight: 700;">Ensemble Intelligence</h3>
        <p style="margin:0; font-size: 14px; color: #6b7280; margin-top: 4px;">Consensus score across all 6 models.</p></div>
        <div style="text-align: right;"><h2 style="margin:0; color: #111827; font-size: 36px; font-weight: 800;">{ensemble:.2f}</h2>{risk_chip(ensemble_risk)}</div>
    </div>
    """, unsafe_allow_html=True)

   

    # --- FLOOD MAP ---
    st.markdown("---")
    st.subheader("🗺️ Flood Risk Inundation Map")
    col_map, col_details = st.columns([3, 1], gap="medium")
    with col_map:
        flood_map = create_flood_map(ensemble)
        st_folium(flood_map, width="100%", height=500)
    with col_details:
        st.info("ℹ️ **About this Map**")
        st.markdown("""
        **Layers Available:**
        - **Flood Inundation (Blue)**: Shows estimated water spread.
        - **Slope Gradient (Red/Yellow)**: Shows steep terrain.
        
        *Use layer control (top right) to toggle.*
        """)
        if ensemble > 0.6: st.error("⚠️ **High Risk:** Low-lying areas susceptible.")
        else: st.success("✅ **Safe Zone:** Elevation sufficient.")
    
     # =========================================================
    # NEW FEATURE #3: STATIC DEM & SLOPE VISUALIZATION (Plotly/Matplotlib)
    # =========================================================
    st.markdown("---")
    st.subheader("⛰️ Digital Elevation & Slope Analysis")

    # Generate the static plot
    fig = create_static_analysis_plot()
    
    if fig is not None:
        st.pyplot(fig)
        
        st.info("ℹ️ **Terrain Analysis**")
        st.markdown("""
        * **Normalized Elevation (Left):** Visualizes height variations. Blue/Green represents lower ground, while Brown/White represents higher ground.
        * **Slope Gradient (Right):** Visualizes the steepness of the terrain. Yellow indicates steeper slopes, while purple/blue indicates flatter areas where water may pool.
        """)
    else:
        st.warning("GIS Data not available to generate terrain plots.")
    # =========================================================

    # --- TREND ANALYSIS ---
    st.markdown("---")
    st.subheader("🌊 Trend Analysis (Last 7 Days)")
    end_dt = pd.to_datetime(sel_date)
    start_dt = end_dt - timedelta(days=6)
    mask = (df['Date'] >= start_dt) & (df['Date'] <= end_dt)
    trend_df = df.loc[mask]
    if not trend_df.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=trend_df['Date'], y=trend_df['Daily_Rainfall_mm'], name="Rainfall (mm)", marker_color='#3b82f6', opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['Prakasam_Barrage_Inflow_cusecs'], name="Inflow (cusecs)", line=dict(color='#ef4444', width=3), mode='lines+markers'), secondary_y=True)
        fig.update_layout(title_text="Hydrograph", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#4b5563'), height=400)
        fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=False, showgrid=False)
        fig.update_yaxes(title_text="Inflow (cusecs)", secondary_y=True, showgrid=True, gridcolor='#e5e7eb')
        st.plotly_chart(fig, use_container_width=True)

    # --- UPDATED FORECAST TABLE (Clean Material Style) ---
    out_rows = []
    rolling_df = df_feat.copy()
    base_date = pd.to_datetime(sel_date)
    for k in range(1, 4):
        nd = base_date + timedelta(days=k)
        last = rolling_df.iloc[-1:].copy()
        last["Date"] = nd
        for col in FEATURES: last[col] *= np.random.uniform(0.95, 1.05)
        rolling_df = pd.concat([rolling_df, last], ignore_index=True)
        rolling_df = safe_features(rolling_df)
        seq_nd = seq_for_date(rolling_df, nd)
        ph, pm, ps, pg, pl2, px = cnn_seq_prob(seq_nd)
        avgd = (ph + pm + ps + pg + pl2 + px) / 6
        out_rows.append((nd.date(), round(avgd, 3), risk_chip(risk_bucket(avgd))))

    st.markdown("---")
    st.subheader("📅 Next 3 Days Risk Forecast")
    
    # Modern Clean Table
    table_html = '<table class="modern-table"><thead><tr><th>Date</th><th>Flood Probability</th><th>Risk Level</th></tr></thead><tbody>'
    for row in out_rows:
        table_html += f"<tr><td style='font-weight:600; color:#374151'>{row[0]}</td><td><strong style='color:#111827'>{row[1]:.3f}</strong></td><td>{row[2]}</td></tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # --- REVAMPED WEATHER SNAPSHOT (Minimalist Cards) ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🌤 Weather Snapshot")
    
    # Map for icon, color class, and cleaned-up label
    weather_info = {
        "Daily_Rainfall_mm": {"icon": "🌧️", "class": "metric-icon-rain", "unit": "mm", "label": "RAINFALL"},
        "Mean_Daily_Temperature_Celsius": {"icon": "🌡️", "class": "metric-icon-temp", "unit": "°C", "label": "TEMPERATURE"},
        "Mean_Daily_Relative_Humidity_Percent": {"icon": "💧", "class": "metric-icon-humidity", "unit": "%", "label": "RELATIVE HUMIDITY"},
        "Mean_Daily_Surface_Pressure_hPa": {"icon": "⏲️", "class": "metric-icon-pressure", "unit": "hPa", "label": "SURFACE PRESSURE"},
        "Mean_Daily_Wind_Speed_kmh": {"icon": "💨", "class": "metric-icon-wind-speed", "unit": "km/h", "label": "WIND SPEED"},
        "Mean_Daily_Wind_Direction_Degree": {"icon": "🧭", "class": "metric-icon-wind-dir", "unit": "°", "label": "WIND DIRECTION"},
        "Mean_Daily_Radiation_Wh_m2": {"icon": "☀️", "class": "metric-icon-rad", "unit": "Wh/m²", "label": "RADIATION"},
    }

    weather_cols = list(weather_info.keys())

    latest = df[df["Date"] <= pd.to_datetime(sel_date)].tail(1)

    def display_metric_revamped(col_name):
        if latest.empty: val = 0
        else: val = latest.iloc[-1][col_name]
            
        info = weather_info[col_name]
        val = round(val, 2)
        
        # Use a simplified metric card structure
        st.markdown(f"""
        <div class="glass-card">
            <div class="weather-card-container">
                <div class="weather-icon-box">
                    <span class="emoji-icon {info['class']}">{info['icon']}</span>
                </div>
                <div style="margin-bottom: 15px;">
                    <span class="metric-value">{val}</span>
                    <span class="metric-unit">{info['unit']}</span>
                </div>
                <div class="weather-label-bottom">{info['label']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4 Cards Top, 3 Cards Bottom layout
    row1 = st.columns(4, gap="large")
    row2 = st.columns(4, gap="large") # Use 4 columns again for consistent gap/width

    if not latest.empty:
        # Top Row (4 cards)
        with row1[0]: display_metric_revamped(weather_cols[0])
        with row1[1]: display_metric_revamped(weather_cols[1])
        with row1[2]: display_metric_revamped(weather_cols[2])
        with row1[3]: display_metric_revamped(weather_cols[3])
        # Bottom Row (3 cards)
        with row2[0]: display_metric_revamped(weather_cols[4])
        with row2[1]: display_metric_revamped(weather_cols[5])
        with row2[2]: display_metric_revamped(weather_cols[6])
    else:
        st.warning("No weather data available for this date.")