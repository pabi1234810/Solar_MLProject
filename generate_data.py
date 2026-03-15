import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

irradiance = np.random.uniform(200, 1100, n)
T_inlet    = np.random.uniform(25, 85, n)
T_ambient  = np.random.uniform(10, 45, n)
wind_speed = np.random.uniform(0, 8, n)
tilt_angle = np.random.uniform(10, 60, n)

FR        = 0.82
tau_alpha = 0.72
UL        = 6.0

eta = FR * (tau_alpha - UL * (T_inlet - T_ambient) / irradiance)
eta = np.clip(eta + np.random.normal(0, 0.012, n), 0, 0.85)

df = pd.DataFrame({
    'irradiance': irradiance,
    'T_inlet':    T_inlet,
    'T_ambient':  T_ambient,
    'wind_speed': wind_speed,
    'tilt_angle': tilt_angle,
    'efficiency': np.round(eta, 4)
})

df.to_csv('data/solar_data.csv', index=False)
print(f"Done. {len(df)} rows saved.")