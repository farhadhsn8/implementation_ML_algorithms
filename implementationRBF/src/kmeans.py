"""A small K-means clustering implementation used by the RBF networks.

The cluster centers discovered by K-means become the RBF basis-function
centers, and the pairwise distance between centers is used to derive a
sensible Gaussian width (``std``).
"""

from __future__ import annotations

import numpy as np

CONVERGENCE_EPSILON = 1e-5


class KMeans:
    """K-means clustering on a feature matrix.

    Parameters
    ----------
    features : np.ndarray
        Data matrix of shape ``(n_samples, n_features)``.
    k : int
        Number of clusters (and therefore RBF centers).
    seed : int | None, default=None
        Optional random seed for reproducible center initialisation.
    """

    def __init__(self, features: np.ndarray, k: int, seed: int | None = None):
        self.features = np.asarray(features, dtype=float)
        self.k = k
        self.seed = seed

    def fit(self) -> tuple[np.ndarray, np.ndarray]:
        """Run Lloyd's algorithm until convergence.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(cluster_assignments, centers)``.
        """
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(self.features.shape[0], self.k, replace=False)
        centers = self.features[indices].copy()
        clusters = np.zeros(self.features.shape[0], dtype=int)

        while True:
            for i, point in enumerate(self.features):
                clusters[i] = int(
                    np.linalg.norm(point - centers, axis=1).argmin()
                )
            previous = centers.copy()
            for j in range(self.k):
                members = self.features[clusters == j]
                if members.shape[0] > 0:
                    centers[j] = members.mean(axis=0)
            if np.abs(centers - previous).sum() < CONVERGENCE_EPSILON:
                break

        return clusters, centers
