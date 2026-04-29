import importlib.util
import os

import numpy as np
from sklearn.datasets import make_blobs

# Import the Perceptron from the chapter directory (not a package, use importlib)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_perceptron_path = os.path.join(_project_root, "charpter8_neural_networks", "perceptron.py")
_spec = importlib.util.spec_from_file_location("perceptron", _perceptron_path)
_perceptron_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_perceptron_module)
Perceptron = _perceptron_module.Perceptron


class TestPerceptron:
    def test_initialize_with_zeros(self):
        p = Perceptron()
        w, b = p.initialize_with_zeros(5)
        assert w.shape == (5,)
        assert np.all(w == 0.0)
        assert b == 0.0

    def test_sign(self):
        p = Perceptron()
        x = np.array([1.0, 2.0])
        w = np.array([0.5, -0.3])
        b = 0.1
        result = p.sign(x, w, b)
        assert isinstance(result, (float, np.floating))

    def test_train_linearly_separable(self):
        X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=0.5, random_state=42)
        y_ = np.array([1 if label == 1 else -1 for label in y])

        p = Perceptron()
        params = p.train(X, y_, learning_rate=0.1)

        w, b = params['w'], params['b']
        y_pred = np.array([1 if p.sign(x, w, b) > 0 else -1 for x in X])
        assert np.array_equal(y_pred, y_)
