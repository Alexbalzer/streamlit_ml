import os
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml_playground.core.pipeline import _last_estimator_labels


def test_last_estimator_labels_returns_expected_array():
    X = np.array([[0, 0], [0, 1], [9, 8], [8, 9]])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=2, random_state=0)),
    ])

    pipe.fit(X)

    labels = _last_estimator_labels(pipe, X)
    expected = np.array([1, 1, 0, 0])

    assert isinstance(labels, np.ndarray)
    assert np.array_equal(labels, expected)
