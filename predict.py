# predict.py
import os
from pathlib import Path
import joblib
import pandas as pd

DEFAULT_MODEL_NAMES = [
    "churn_pipeline.joblib",
    "churn_rf.joblib",
    "model.joblib"
]

def find_model_path():
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    # try current folder, parent outputs folder, or known names
    candidates = [Path(p) for p in DEFAULT_MODEL_NAMES] + [Path("outputs") / p for p in DEFAULT_MODEL_NAMES]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Model file not found. Set environment variable MODEL_PATH or place model in outputs/. Searched: " + ", ".join(str(x) for x in candidates))

def load_model():
    model_path = find_model_path()
    model = joblib.load(model_path)
    return model

def predict_single(model, input_dict):
    # expects dict mapping of features (same names as training X columns)
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0,1] if hasattr(model, "predict_proba") else None
    return {'prediction': int(pred), 'probability': float(proba) if proba is not None else None}

if __name__ == "__main__":
    model = load_model()
    sample = {
        "Age": 40,
        "Gender": "Female",
        "Tenure": 20,
        "Usage Frequency": 10,
        "Support Calls": 3,
        "Payment Delay": 5,
        "Subscription Type": "Standard",
        "Contract Length": "Monthly",
        "Total Spend": 400,
        "Last Interaction": 5
    }
    print(predict_single(model, sample))
