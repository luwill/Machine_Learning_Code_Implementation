import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error

from mlbook.decision_tree.cart import (
    BinaryDecisionTree,
    ClassificationTree,
    RegressionTree,
    TreeNode,
)


class TestTreeNode:
    def test_leaf_node(self):
        node = TreeNode(leaf_value=1.0)
        assert node.leaf_value == 1.0
        assert node.feature_i is None
        assert node.left_branch is None

    def test_internal_node(self):
        left = TreeNode(leaf_value=0.0)
        right = TreeNode(leaf_value=1.0)
        node = TreeNode(feature_i=2, threshold=0.5, left_branch=left, right_branch=right)
        assert node.feature_i == 2
        assert node.threshold == 0.5
        assert node.leaf_value is None


class TestClassificationTree:
    def test_fit_predict_iris(self, iris_data):
        X, y = iris_data
        tree = ClassificationTree(min_samples_split=2, max_depth=5)
        tree.fit(X, y)
        y_pred = tree.predict(X)
        acc = accuracy_score(y, y_pred)
        assert acc > 0.9

    def test_predict_returns_ndarray(self, iris_data):
        X, y = iris_data
        tree = ClassificationTree(max_depth=3)
        tree.fit(X, y)
        y_pred = tree.predict(X[:10])
        assert isinstance(y_pred, np.ndarray)

    def test_max_depth_enforcement(self, iris_data):
        X, y = iris_data
        tree = ClassificationTree(max_depth=1)
        tree.fit(X, y)
        y_pred = tree.predict(X)
        # With depth=1, accuracy won't be 100% but should still run
        assert len(y_pred) == len(y)

    def test_min_samples_split(self, synthetic_binary):
        X, y = synthetic_binary
        tree = ClassificationTree(min_samples_split=100)
        tree.fit(X, y)
        y_pred = tree.predict(X)
        assert len(y_pred) == len(y)


class TestRegressionTree:
    def test_fit_predict(self, synthetic_regression):
        X, y = synthetic_regression
        tree = RegressionTree(min_samples_split=5, max_depth=5)
        tree.fit(X, y)
        y_pred = tree.predict(X)
        mse = mean_squared_error(y, y_pred)
        mean_pred_mse = mean_squared_error(y, np.full_like(y, np.mean(y)))
        assert mse < mean_pred_mse

    def test_predict_returns_ndarray(self, synthetic_regression):
        X, y = synthetic_regression
        tree = RegressionTree(max_depth=3)
        tree.fit(X, y)
        y_pred = tree.predict(X[:10])
        assert isinstance(y_pred, np.ndarray)


class TestBinaryDecisionTree:
    def test_base_class_raises_on_direct_fit(self):
        tree = BinaryDecisionTree()
        X = np.random.rand(10, 3)
        y = np.random.rand(10, 1)
        with pytest.raises(AttributeError):
            tree.fit(X, y)
