import numpy as np


class TestKMeans:
    """Test a simple KMeans implementation matching the chapter's approach."""

    def test_kmeans_runs(self, simple_kmeans_data):
        X = simple_kmeans_data
        k = 3
        n_samples = X.shape[0]

        # Initialize centroids by random sample
        rng = np.random.RandomState(42)
        centroids = X[rng.choice(n_samples, k, replace=False)]

        # Run a few iterations of Lloyd's algorithm
        for _ in range(10):
            # Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # Verify each cluster has at least one point
        for i in range(k):
            assert np.sum(labels == i) > 0

        # Verify all points are assigned
        assert len(labels) == n_samples
        # Verify only valid labels
        assert set(np.unique(labels)) <= {0, 1, 2}
