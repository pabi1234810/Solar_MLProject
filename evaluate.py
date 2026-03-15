import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

df     = pd.read_csv('data/solar_data.csv')
X      = df[['irradiance','T_inlet','T_ambient','wind_speed','tilt_angle']]
y      = df['efficiency']
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
model  = pickle.load(open('models/best_model.pkl', 'rb'))

X_scaled = scaler.transform(X)
_, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)

print(f"R²  : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue', edgecolors='none')
plt.plot([0, 0.85], [0, 0.85], 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel("Actual Efficiency")
plt.ylabel("Predicted Efficiency")
plt.title("Actual vs Predicted — Solar Collector Efficiency")
plt.legend()
plt.tight_layout()
plt.savefig('outputs/actual_vs_predicted.png', dpi=150)
plt.show()
print("Plot saved to outputs/")