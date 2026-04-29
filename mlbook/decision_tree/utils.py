from typing import Optional, Tuple

import numpy as np


def feature_split(
    X: np.ndarray, feature_i: int, threshold: float | str
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary feature split. Numeric features use <= threshold, categorical use ==."""
    if isinstance(threshold, (int, float)):
        split_func = lambda sample: sample[feature_i] <= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_left = np.array([sample for sample in X if split_func(sample)])
    X_right = np.array([sample for sample in X if not split_func(sample)])

    return X_left, X_right


def calculate_gini(y: np.ndarray) -> float:
    """Calculate Gini impurity of labels."""
    y_list = y.tolist()
    probs = [y_list.count(i) / len(y_list) for i in np.unique(y_list)]
    gini = sum(p * (1 - p) for p in probs)
    return gini


def data_shuffle(
    X: np.ndarray, y: np.ndarray, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle data with optional random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def cat_label_convert(
    y: np.ndarray, n_col: Optional[int] = None
) -> np.ndarray:
    """Convert categorical labels to one-hot encoding."""
    if n_col is None:
        n_col = int(np.amax(y)) + 1
    one_hot = np.zeros((y.shape[0], n_col))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot
