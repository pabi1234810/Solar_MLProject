import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv('data/solar_data.csv')
X  = df[['irradiance','T_inlet','T_ambient','wind_speed','tilt_angle']]
y  = df['efficiency']

scaler   = pickle.load(open('models/scaler.pkl', 'rb'))
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── Tune Random Forest ─────────────────────────────────────
print("Tuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [5, 10, 20, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, cv=5, scoring='r2', n_jobs=-1, verbose=1
)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_r2   = r2_score(y_test, rf_pred)
print(f"Best RF Params : {rf_grid.best_params_}")
print(f"Best RF R²     : {rf_r2:.4f}")

# ── Tune XGBoost ───────────────────────────────────────────
print("\nTuning XGBoost...")
xgb_params = {
    'n_estimators':  [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth':     [3, 5, 7]
}
xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    xgb_params, cv=5, scoring='r2', n_jobs=-1, verbose=1
)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)
xgb_r2   = r2_score(y_test, xgb_pred)
print(f"Best XGB Params: {xgb_grid.best_params_}")
print(f"Best XGB R²    : {xgb_r2:.4f}")

# ── Save best of the two ───────────────────────────────────
if rf_r2 >= xgb_r2:
    pickle.dump(rf_best,  open('models/best_model.pkl', 'wb'))
    print(f"\nSaved: Tuned Random Forest (R²={rf_r2:.4f})")
else:
    pickle.dump(xgb_best, open('models/best_model.pkl', 'wb'))
    print(f"\nSaved: Tuned XGBoost (R²={xgb_r2:.4f})")