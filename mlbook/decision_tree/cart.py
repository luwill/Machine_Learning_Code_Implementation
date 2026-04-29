from typing import Optional

import numpy as np

from mlbook.decision_tree.utils import calculate_gini, feature_split


class TreeNode:
    def __init__(
        self,
        feature_i: Optional[int] = None,
        threshold: Optional[float] = None,
        leaf_value: Optional[float] = None,
        left_branch: Optional["TreeNode"] = None,
        right_branch: Optional["TreeNode"] = None,
    ) -> None:
        self.feature_i = feature_i
        self.threshold = threshold
        self.leaf_value = leaf_value
        self.left_branch = left_branch
        self.right_branch = right_branch


class BinaryDecisionTree:
    def __init__(
        self,
        min_samples_split: int = 2,
        min_gini_impurity: float = float("inf"),
        max_depth: float = float("inf"),
        loss: Optional[object] = None,
    ) -> None:
        self.root: Optional[TreeNode] = None
        self.min_samples_split = min_samples_split
        self.min_gini_impurity = min_gini_impurity
        self.max_depth = max_depth
        self.gini_impurity_calculation = None
        self._leaf_value_calculation = None
        self.loss = loss

    def fit(self, X: np.ndarray, y: np.ndarray, loss: Optional[object] = None) -> None:
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, current_depth: int = 0
    ) -> TreeNode:
        init_gini_impurity = float("inf")
        best_criteria: Optional[dict] = None
        best_sets: Optional[dict] = None

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    Xy1, Xy2 = feature_split(Xy, feature_i, threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        impurity = self.impurity_calculation(y, y1, y2)

                        if impurity < init_gini_impurity:
                            init_gini_impurity = impurity
                            best_criteria = {
                                "feature_i": feature_i,
                                "threshold": threshold,
                            }
                            best_sets = {
                                "leftX": Xy1[:, :n_features],
                                "lefty": Xy1[:, n_features:],
                                "rightX": Xy2[:, :n_features],
                                "righty": Xy2[:, n_features:],
                            }

        if (
            best_criteria is not None
            and init_gini_impurity < self.min_gini_impurity
        ):
            left_branch = self._build_tree(
                best_sets["leftX"], best_sets["lefty"], current_depth + 1
            )
            right_branch = self._build_tree(
                best_sets["rightX"], best_sets["righty"], current_depth + 1
            )
            return TreeNode(
                feature_i=best_criteria["feature_i"],
                threshold=best_criteria["threshold"],
                left_branch=left_branch,
                right_branch=right_branch,
            )

        leaf_value = self._leaf_value_calculation(y)
        return TreeNode(leaf_value=leaf_value)

    def predict_value(self, x: np.ndarray, tree: Optional[TreeNode] = None) -> float:
        if tree is None:
            tree = self.root

        if tree.leaf_value is not None:
            return tree.leaf_value

        feature_value = x[tree.feature_i]

        branch = tree.right_branch
        if isinstance(feature_value, (int, float)):
            if feature_value <= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch

        return self.predict_value(x, branch)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_value(sample) for sample in X])


class ClassificationTree(BinaryDecisionTree):
    def _calculate_gini_impurity(
        self, y: np.ndarray, y1: np.ndarray, y2: np.ndarray
    ) -> float:
        p = len(y1) / len(y)
        gini_impurity = p * calculate_gini(y1) + (1 - p) * calculate_gini(y2)
        return gini_impurity

    def _majority_vote(self, y: np.ndarray) -> float:
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.impurity_calculation = self._calculate_gini_impurity
        self._leaf_value_calculation = self._majority_vote
        self.gini_impurity_calculation = calculate_gini
        super().fit(X, y)


class RegressionTree(BinaryDecisionTree):
    def _calculate_variance_reduction(
        self, y: np.ndarray, y1: np.ndarray, y2: np.ndarray
    ) -> float:
        var_tot = np.var(y, axis=0)
        var_y1 = np.var(y1, axis=0)
        var_y2 = np.var(y2, axis=0)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        variance_reduction = var_tot - (frac_1 * var_y1 + frac_2 * var_y2)
        return float(sum(variance_reduction))

    def _mean_of_y(self, y: np.ndarray) -> float:
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        self.gini_impurity_calculation = lambda y: np.var(y, axis=0)
        super().fit(X, y)
