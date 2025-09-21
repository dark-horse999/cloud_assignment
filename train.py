# train.py
import argparse
import json
import os
from pathlib import Path
import joblib
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from data_utils import load_data, prepare_xy, train_val_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_pipeline(X: pd.DataFrame, random_state=42):
    """
    Minimal preprocessing pipeline:
     - numeric features -> StandardScaler
     - categorical features -> OneHotEncoder (drop='if_binary' is optional)
    Update this to match your actual dataset and desired transforms.
    """
    # detect dtypes
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    return pipeline, numeric_cols, cat_cols

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help="Path to training CSV (local path inside job).")
    p.add_argument("--output_dir", type=str, default="outputs", help="Where to write model and metrics (outputs/).")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Loading data from %s", args.data_path)
    df = load_data(args.data_path)
    X, y = prepare_xy(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=args.test_size, random_state=args.random_state)

    pipeline, numeric_cols, cat_cols = build_pipeline(X_train, random_state=args.random_state)
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating on validation set...")
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "roc_auc": float(roc_auc_score(y_val, y_proba)) if y_proba is not None else None
    }
    logger.info("Metrics: %s", metrics)

    # Save pipeline (preprocessor + model), column names, and metrics
    model_path = Path(args.output_dir) / "churn_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Saved model pipeline to %s", model_path)

    with open(Path(args.output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature lists for debugging
    with open(Path(args.output_dir) / "columns.json", "w") as f:
        json.dump({"numeric": numeric_cols, "categorical": cat_cols, "training_columns": X.columns.tolist()}, f, indent=2)

if __name__ == "__main__":
    main()
