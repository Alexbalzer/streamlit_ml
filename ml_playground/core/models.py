# ml_playground/core/models.py
from typing import Dict, Any
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# ---------------- Classification ----------------
_CLASSIFIERS: Dict[str, Dict[str, Any]] = {
    "KNN": {
        "builder": lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=p["K"])),
        "space": {
            "K": {"type": "int", "min": 1, "max": 25, "default": 5, "step": 1, "label": "K"},
        },
    },
    "SVM": {
        "builder": lambda p: make_pipeline(StandardScaler(), SVC(C=p["C"], kernel="rbf")),
        "space": {
            "C": {"type": "float", "min": 0.001, "max": 10.0, "default": 1.0, "label": "C"},
        },
    },
    "Random Forest": {
        "builder": lambda p: RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=42, n_jobs=-1
        ),
        "space": {
            "max_depth": {"type": "int", "min": 2, "max": 30, "default": 10, "step": 1, "label": "max_depth"},
            "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 200, "step": 10, "label": "n_estimators"},
        },
    },
}

def list_models():
    return list(_CLASSIFIERS.keys())

def get_param_space(name: str):
    return _CLASSIFIERS[name]["space"]

def build_model(name: str, params: Dict[str, Any]):
    return _CLASSIFIERS[name]["builder"](params)

# ---------------- Clustering ----------------
_CLUSTERERS: Dict[str, Dict[str, Any]] = {
    "KMeans": {
        "builder": lambda p: make_pipeline(
            StandardScaler(),
            KMeans(n_clusters=p["n_clusters"], n_init=p["n_init"], init=p["init"], random_state=42)
        ),
        "space": {
            "n_clusters": {"type": "int", "min": 2, "max": 15, "default": 3, "step": 1, "label": "n_clusters"},
            "init": {"type": "select", "choices": ["k-means++", "random"], "index": 0, "label": "init"},
            "n_init": {"type": "int", "min": 1, "max": 50, "default": 10, "step": 1, "label": "n_init"},
        },
    },
    "DBSCAN": {
        "builder": lambda p: make_pipeline(
            StandardScaler(),
            DBSCAN(eps=p["eps"], min_samples=p["min_samples"])
        ),
        "space": {
            "eps": {"type": "float", "min": 0.1, "max": 5.0, "default": 0.8, "label": "eps"},
            "min_samples": {"type": "int", "min": 3, "max": 100, "default": 5, "step": 1, "label": "min_samples"},
        },
    },
    "Agglomerative": {
        "builder": lambda p: make_pipeline(
            StandardScaler(),
            AgglomerativeClustering(n_clusters=p["n_clusters"], linkage=p["linkage"])
        ),
        "space": {
            "n_clusters": {"type": "int", "min": 2, "max": 15, "default": 3, "step": 1, "label": "n_clusters"},
            "linkage": {"type": "select", "choices": ["ward", "complete", "average", "single"], "index": 0, "label": "linkage"},
        },
    },
}

def list_clusterers():
    return list(_CLUSTERERS.keys())

def get_cluster_param_space(name: str):
    return _CLUSTERERS[name]["space"]

def build_clusterer(name: str, params: Dict[str, Any]):
    return _CLUSTERERS[name]["builder"](params)
