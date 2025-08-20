# ml_playground/core/pipeline.py
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Clustering metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# --------- Classification ---------
def train_and_eval(X: pd.DataFrame, y: np.ndarray, model) -> Dict[str, Any]:
    Xtr, Xte, ytr, yte = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    acc = float(accuracy_score(yte, yhat))
    return {"model": model, "accuracy": acc, "y_pred": yhat, "X_test": Xte, "y_test": yte}

# --------- Clustering ---------
def _last_estimator_labels(pipeline_or_estimator, X):
    # holt labels_ aus letzter Stufe; fallback auf predict
    est = pipeline_or_estimator
    if hasattr(est, "named_steps"):
        for step in est.named_steps.values():
            if hasattr(step, "labels_"):
                return step.labels_
        est = list(est.named_steps.values())[-1]
    return getattr(est, "labels_", None) or est.fit_predict(X)

def cluster_and_eval(X: pd.DataFrame, model, y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
    Xv = X.values
    labels = _last_estimator_labels(model.fit(Xv), Xv)

    # Anzahl Cluster (Noise -1 bei DBSCAN ignorieren)
    uniq = np.unique(labels)
    n_clusters = int(np.sum(uniq >= 0))

    # Metriken
    sil, ch, db = None, None, None
    valid_for_internal = len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels)
    if valid_for_internal:
        # metriken besser mit skalierten Features
        Xs = StandardScaler().fit_transform(Xv)
        try:
            sil = float(silhouette_score(Xs, labels))
        except Exception:
            sil = None
        try:
            ch = float(calinski_harabasz_score(Xs, labels))
        except Exception:
            ch = None
        try:
            db = float(davies_bouldin_score(Xs, labels))
        except Exception:
            db = None

    ari = nmi = None
    if y_true is not None and len(y_true) == len(labels):
        try:
            ari = float(adjusted_rand_score(y_true, labels))
            nmi = float(normalized_mutual_info_score(y_true, labels))
        except Exception:
            pass

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "ari": ari,
        "nmi": nmi,
    }

# --------- PCA ---------
def pca_project(X: pd.DataFrame, n_components: int = 2) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(Xs)
    return Z
