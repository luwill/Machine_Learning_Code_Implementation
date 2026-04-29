import numpy as np
from mlbook.decision_tree.utils import (
    calculate_gini,
    cat_label_convert,
    data_shuffle,
    feature_split,
)


class TestFeatureSplit:
    def test_numeric_threshold(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        left, right = feature_split(X, 0, 3.0)
        assert len(left) == 2
        assert len(right) == 1
        assert np.array_equal(left, np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert np.array_equal(right, np.array([[5.0, 6.0]]))

    def test_categorical_threshold(self):
        X = np.array([["a", 1], ["b", 2], ["a", 3]])
        left, right = feature_split(X, 0, "a")
        assert len(left) == 2
        assert len(right) == 1

    def test_all_same(self):
        X = np.array([[1.0], [1.0], [1.0]])
        left, right = feature_split(X, 0, 1.0)
        assert len(left) == 3
        assert len(right) == 0


class TestCalculateGini:
    def test_pure_node(self):
        y = np.array([0, 0, 0])
        assert calculate_gini(y) == 0.0

    def test_balanced_binary(self):
        y = np.array([0, 1, 0, 1])
        assert calculate_gini(y) == 0.5

    def test_three_class(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        gini = calculate_gini(y)
        assert abs(gini - 2 / 3) < 1e-10


class TestDataShuffle:
    def test_preserves_shape(self):
        X = np.random.rand(100, 5)
        y = np.random.rand(100, 1)
        X_s, y_s = data_shuffle(X, y, seed=42)
        assert X_s.shape == X.shape
        assert y_s.shape == y.shape

    def test_deterministic(self):
        X = np.random.rand(50, 3)
        y = np.random.rand(50, 1)
        X1, y1 = data_shuffle(X, y, seed=42)
        X2, y2 = data_shuffle(X, y, seed=42)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_same_unique_values(self):
        X = np.arange(20).reshape(10, 2)
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        X_s, y_s = data_shuffle(X, y, seed=7)
        assert set(np.unique(y_s)) == {0, 1}


class TestCatLabelConvert:
    def test_three_class(self):
        y = np.array([0, 1, 2, 0])
        result = cat_label_convert(y)
        assert result.shape == (4, 3)
        assert np.array_equal(result[0], [1, 0, 0])
        assert np.array_equal(result[1], [0, 1, 0])

    def test_explicit_n_col(self):
        y = np.array([0, 1])
        result = cat_label_convert(y, n_col=4)
        assert result.shape == (2, 4)
