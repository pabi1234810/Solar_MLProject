import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ── Train and cache model if not present ─────────────────
@st.cache_resource
def load_or_train_model():
    if os.path.exists('models/best_model.pkl') and os.path.exists('models/scaler.pkl'):
        model  = pickle.load(open('models/best_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl',     'rb'))
    else:
        # Generate data on the fly
        np.random.seed(42)
        n = 2000

        irradiance = np.random.uniform(200, 1100, n)
        T_inlet    = np.random.uniform(25,  85,   n)
        T_ambient  = np.random.uniform(10,  45,   n)
        wind_speed = np.random.uniform(0,   8,    n)
        tilt_angle = np.random.uniform(10,  60,   n)

        FR        = 0.82
        tau_alpha = 0.72
        UL        = 6.0

        eta = FR * (tau_alpha - UL * (T_inlet - T_ambient) / irradiance)
        eta = np.clip(eta + np.random.normal(0, 0.012, n), 0, 0.85)

        X = np.column_stack([irradiance, T_inlet, T_ambient, wind_speed, tilt_angle])
        y = eta

        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    return model, scaler

model, scaler = load_or_train_model()

# ── App UI ────────────────────────────────────────────────
st.title("Solar Collector Efficiency Predictor")
st.write("Enter operating conditions to predict thermal efficiency.")

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

# ── Feature Importance ────────────────────────────────────
st.markdown("---")
st.subheader("Feature Importance")

features   = ['Irradiance', 'Inlet Temp', 'Ambient Temp', 'Wind Speed', 'Tilt Angle']
importance = model.feature_importances_
fi_df      = pd.DataFrame({'Feature': features, 'Importance': importance})
fi_df      = fi_df.sort_values('Importance', ascending=True)

st.bar_chart(fi_df.set_index('Feature'))