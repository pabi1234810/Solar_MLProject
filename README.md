# Solar Collector Efficiency Predictor

A machine learning web app that predicts the thermal efficiency of a solar collector based on operating conditions and environmental parameters.

---

## Live Demo
[Click here to open the app](https://pabi1234810-solar-mlproject-app.streamlit.app)

---

## Project Overview

This project uses the **Hottel-Whillier-Bliss (HWB)** physical model to generate a realistic dataset and trains multiple ML models to predict solar collector efficiency.

---

## Features

- Predicts thermal efficiency from 5 input parameters
- Compares Linear Regression, Random Forest and XGBoost
- Hyperparameter tuning using GridSearchCV
- Feature importance visualization
- Interactive Streamlit web app with sample data table

---

## Input Parameters

| Parameter | Range | Unit |
|-----------|-------|------|
| Solar Irradiance | 200 — 1100 | W/m² |
| Inlet Temperature | 25 — 85 | °C |
| Ambient Temperature | 10 — 45 | °C |
| Wind Speed | 0 — 8 | m/s |
| Collector Tilt Angle | 10 — 60 | ° |

---

## Output

| Condition | Efficiency Range |
|-----------|-----------------|
| Best case | 0.55 — 0.85 |
| Typical | 0.30 — 0.55 |
| Poor | 0.00 — 0.30 |

---

## Project Structure
```
SolarMLProject/
├── data/
│   └── solar_data.csv       ← generated dataset (2000 rows)
├── models/
│   ├── best_model.pkl       ← saved best ML model
│   └── scaler.pkl           ← saved MinMaxScaler
├── outputs/
│   └── feature_importance.png
├── generate_data.py         ← generates dataset using HWB model
├── train_model.py           ← trains and compares ML models
├── tune_model.py            ← hyperparameter tuning
├── evaluate.py              ← evaluation plots
├── feature_importance.py    ← feature importance chart
├── app.py                   ← Streamlit web app
└── requirements.txt
```

---

## ML Models Used

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted trees |

Best model selected automatically based on R² score.

---

## Physics Background

Efficiency calculated using the **Hottel-Whillier-Bliss equation:**
```
η = FR × [τα - UL × (Ti - Ta) / I]
```

Where:
- FR = Heat removal factor
- τα = Transmittance-absorptance product
- UL = Overall heat loss coefficient
- Ti = Inlet fluid temperature
- Ta = Ambient temperature
- I  = Solar irradiance

---

## Installation
```bash
git clone https://github.com/pabi1234810/Solar_MLProject.git
cd Solar_MLProject
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
```

---

## Run Locally
```bash
# Step 1 - Generate data
python generate_data.py

# Step 2 - Train models
python train_model.py

# Step 3 - Tune models
python tune_model.py

# Step 4 - Evaluate
python evaluate.py

# Step 5 - Launch app
streamlit run app.py
```

---

## Tech Stack

- Python 3.11
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Matplotlib
- Streamlit

---

## Author

**Pabitra Chakraborty**  
Mechanical Engineering, Jadavpur University (2023-2027)