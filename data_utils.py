# data_utils.py (safe improvements)
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV from a path. `path` can be:
     - local path (./data/train.csv)
     - path provided by AzureML job (mounted local path).
    """
    p = Path(path)
    # If it's a folder containing a single CSV, try to find it
    if p.is_dir():
        csvs = list(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in folder: {path}")
        return pd.read_csv(csvs[0])
    # else assume file path
    return pd.read_csv(path)

def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # drop customer id
    df = df.copy()
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    # target
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
