# ml_playground/core/__init__.py
from .datasets import list_datasets, load_dataset, load_csv, split_features_target
from .models import (
    list_models, get_param_space, build_model,
    list_clusterers, get_cluster_param_space, build_clusterer
)
from .pipeline import train_and_eval, cluster_and_eval, pca_project

__all__ = [
    # datasets
    "list_datasets", "load_dataset", "load_csv", "split_features_target",
    # models
    "list_models", "get_param_space", "build_model",
    "list_clusterers", "get_cluster_param_space", "build_clusterer",
    # pipeline
    "train_and_eval", "cluster_and_eval", "pca_project",
]
