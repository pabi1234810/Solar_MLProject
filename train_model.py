import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# Load data
df = pd.read_csv('data/solar_data.csv')
X = df[['irradiance','T_inlet','T_ambient','wind_speed','tilt_angle']]
y = df['efficiency']

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train and compare models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost':           XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
}

best_r2    = -1
best_model = None
best_name  = ''

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2   = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:22s} → R²: {r2:.4f}  RMSE: {rmse:.4f}")
    if r2 > best_r2:
        best_r2    = r2
        best_model = model
        best_name  = name

# Save best model and scaler
pickle.dump(best_model, open('models/best_model.pkl', 'wb'))
pickle.dump(scaler,     open('models/scaler.pkl',     'wb'))
print(f"\nBest model: {best_name} (R²={best_r2:.4f}) — saved.")