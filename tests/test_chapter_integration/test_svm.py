import numpy as np
import sys
import os

# Import the SVM class from the chapter notebook
# Since SVM classes are defined in notebooks, we test using cvxopt directly
import pytest

cvxopt = pytest.importorskip("cvxopt")
from cvxopt import matrix, solvers


class TestHardMarginSVM:
    """Test Hard Margin SVM using cvxopt QP solver directly."""

    def test_qp_solver_runs(self, simple_blobs):
        X, y = simple_blobs
        m, n = X.shape

        P = matrix(np.identity(n + 1, dtype=np.float64))
        q = matrix(np.zeros((n + 1,), dtype=np.float64))
        G = matrix(np.zeros((m, n + 1), dtype=np.float64))
        h = -matrix(np.ones((m,), dtype=np.float64))

        P[0, 0] = 0
        for i in range(m):
            G[i, 0] = -y[i]
            G[i, 1:] = -X[i, :] * y[i]

        sol = solvers.qp(P, q, G, h)
        w = np.zeros(n)
        b = sol['x'][0]
        for i in range(1, n + 1):
            w[i - 1] = sol['x'][i]

        y_pred = np.sign(np.dot(w, X.T) + b)
        accuracy = np.mean(y_pred == y)
        assert accuracy > 0.95
