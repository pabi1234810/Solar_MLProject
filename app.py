import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

model  = pickle.load(open('models/best_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl',     'rb'))

st.title("Solar Collector Efficiency Predictor")
st.write("Enter operating conditions to predict thermal efficiency.")

# ── Sliders ──────────────────────────────────────────────
irradiance = st.slider("Solar Irradiance (W/m²)", 200, 1100, 700)
T_inlet    = st.slider("Inlet Temperature (°C)",   25,   85,  50)
T_ambient  = st.slider("Ambient Temperature (°C)", 10,   45,  30)
wind_speed = st.slider("Wind Speed (m/s)",           0,    8,   2)
tilt_angle = st.slider("Collector Tilt Angle (°)",  10,   60,  35)

if st.button("Predict Efficiency"):
    inp   = np.array([[irradiance, T_inlet, T_ambient, wind_speed, tilt_angle]])
    inp_s = scaler.transform(inp)
    eta   = model.predict(inp_s)[0]
    st.success(f"Predicted Efficiency: **{eta:.4f}** ({eta*100:.1f}%)")

# ── Sample Data Table ─────────────────────────────────────
st.markdown("---")
st.subheader("Sample Data with Efficiency")

sample = pd.DataFrame({
    'Irradiance (W/m²)': [800, 600, 1000, 400, 900, 750, 500, 1050, 350, 650],
    'Inlet Temp (°C)':   [ 45,  60,   35,  70,  40,  55,  65,   30,  75,  50],
    'Ambient Temp (°C)': [ 30,  25,   32,  20,  35,  28,  22,   38,  18,  26],
    'Wind Speed (m/s)':  [2.0, 3.5,  1.0, 4.0, 1.5, 2.5, 3.0,  0.5, 5.0, 2.0],
    'Tilt Angle (°)':    [ 35,  40,   30,  45,  35,  38,  42,   28,  50,  36],
})

scaled   = scaler.transform(sample.values)
eff_vals = model.predict(scaled)
sample['Predicted Efficiency'] = [f"{e:.4f} ({e*100:.1f}%)" for e in eff_vals]

st.dataframe(sample, use_container_width=True)

# ── Feature Importance Chart ──────────────────────────────
st.markdown("---")
st.subheader("Feature Importance")

if os.path.exists('outputs/feature_importance.png'):
    st.image('outputs/feature_importance.png', use_container_width=True)
else:
    st.info("Feature importance chart not available in deployed version. Run feature_importance.py locally to generate it.")