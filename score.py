# score.py
import json
import os
import joblib
import pandas as pd

MODEL_ENV_NAME = "AZUREML_MODEL_PATH"  # optional env var
DEFAULT_MODEL_FILE = "churn_pipeline.joblib"

def init():
    global model
    model_path = os.environ.get(MODEL_ENV_NAME, DEFAULT_MODEL_FILE)
    if not os.path.exists(model_path):
        # Azure might mount model into a folder; try relative lookups
        possible = [
            model_path,
            os.path.join("model", model_path),
            os.path.join("/var/azureml-app/azureml", model_path),
            os.path.join(os.getcwd(), model_path)
        ]
        found = None
        for p in possible:
            if os.path.exists(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Model file not found. Checked: {possible}")
        model_path = found
    model = joblib.load(model_path)

def run(raw_data):
    """
    Azure will send JSON body. Accepts:
     - list of dicts: [{"Age":..., ...}, ...]
     - single dict: {"Age":...}
    Returns JSON serializable dict.
    """
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        if isinstance(data, dict):
            # single example
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            # maybe Azure sends {"data": [...]}
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return {"error": "Unsupported input format."}

        preds = model.predict(df).tolist()
        proba = model.predict_proba(df)[:, 1].tolist() if hasattr(model, "predict_proba") else None

        result = {"predictions": preds}
        if proba is not None:
            result["probabilities"] = proba
        return result
    except Exception as e:
        return {"error": str(e)}
