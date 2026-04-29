import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression


@pytest.fixture(scope="session")
def iris_data():
    X, y = load_iris(return_X_y=True)
    y = y.reshape(-1, 1)
    return X, y


@pytest.fixture(scope="session")
def synthetic_regression():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    y = y.reshape(-1, 1)
    return X, y


@pytest.fixture(scope="session")
def synthetic_binary():
    X, y = make_classification(
        n_samples=200, n_features=5, n_redundant=0, n_informative=3,
        random_state=42, n_clusters_per_class=1
    )
    y = y.reshape(-1, 1)
    return X, y


@pytest.fixture(scope="session")
def simple_blobs():
    """Two linearly separable blobs for SVM testing."""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.8, random_state=42)
    y_ = y.copy().astype(float)
    y_[y_ == 0] = -1
    return X, y_


@pytest.fixture(scope="session")
def simple_kmeans_data():
    """Simple 3-cluster data."""
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.6, random_state=42)
    return X
