# ml_playground/core/datasets.py
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn import datasets

def list_datasets():
    return ["Iris", "Breast Cancer", "Wine"]

def load_dataset(name: str) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    if name == "Iris":
        d = datasets.load_iris(as_frame=True)
    elif name == "Breast Cancer":
        d = datasets.load_breast_cancer(as_frame=True)
    elif name == "Wine":
        d = datasets.load_wine(as_frame=True)
    else:
        raise ValueError(f"Unbekanntes Dataset: {name}")
    X = d.data
    y = d.target.values if "target" in d else None
    return X, y

def split_features_target(df: pd.DataFrame, target: str):
    df = df.copy()
    if target not in df.columns:
        raise ValueError(f"Target '{target}' nicht in Spalten.")
    y = df[target].values
    X = df.drop(columns=[target])
    # nur numerische Features
    X = X.select_dtypes(include=[np.number])
    # fehlende entfernen
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y[: len(X)]
    return X, y

def load_csv(file, target: Optional[str] = None):
    df = pd.read_csv(file)
    if target is None:
        # nur numerische Spalten verwenden
        X = df.select_dtypes(include=[np.number]).copy()
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        return X, None
    else:
        return split_features_target(df, target)
