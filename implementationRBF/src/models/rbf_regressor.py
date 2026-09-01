"""Radial Basis Function network for regression.

A Gaussian basis-function layer projects the input into a higher
dimensional space, and a single linear output neuron combines the basis
responses with a learnable weight vector. The network supports both
auto-discovered centers (K-means) and user-provided centers / widths.

This is a faithful, production-friendly port of ``RBFNetRegressor`` from
``RBF.ipynb``.
"""

from __future__ import annotations

import numpy as np

from ..kmeans import KMeans


class RBFRegressor:
    """RBF network that performs scalar regression."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        number_of_gaussians: int,
        eta: float = 0.01,
        centers: np.ndarray | None = None,
        stds: float | np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.features = np.asarray(features, dtype=float)
        self.targets = np.asarray(targets, dtype=float)
        self.number_of_gaussians = number_of_gaussians
        self.eta = eta
        self.seed = seed

        self.clusters, self.centers = self._set_centers(centers)
        self.stds = self.calculate_std(stds)
        rng = np.random.default_rng(seed)
        self.bias = float(rng.random())
        self.weights = rng.random(number_of_gaussians)

    def _set_centers(self, centers: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if centers is None:
            return KMeans(self.features, self.number_of_gaussians, seed=self.seed).fit()
        return np.zeros(self.features.shape[0], dtype=int), np.asarray(
            centers, dtype=float
        )

    # -- Layers ----------------------------------------------------------

    def gaussian_layer(self, x: np.ndarray) -> np.ndarray:
        """Activation of every Gaussian basis function for input ``x``."""
        distances = np.linalg.norm(self.centers - x, axis=1)
        return np.exp(-((distances / self.stds) ** 2))

    def calculate_output(self, g_layer: np.ndarray) -> float:
        """Linear output neuron combining the basis responses."""
        return float(np.dot(g_layer, self.weights) + self.bias)

    def calculate_std(self, stds: float | np.ndarray | None = None) -> float | np.ndarray:
        """Gaussian width: explicit value or derived from center spread."""
        if stds is not None:
            return stds
        max_distance = 0.0
        for i, center1 in enumerate(self.centers):
            for center2 in self.centers[i + 1 :]:
                max_distance = max(max_distance, np.linalg.norm(center1 - center2))
        if max_distance == 0.0:
            max_distance = 1.0
        return max_distance / (2 * self.number_of_gaussians) ** 0.5

    # -- Training / inference ----------------------------------------------

    def training(self, epochs: int = 5) -> None:
        """Train the output weights with online gradient descent."""
        for _ in range(epochs):
            for i in range(self.features.shape[0]):
                g_layer = self.gaussian_layer(self.features[i])
                y_pred = self.calculate_output(g_layer)
                delta = self.targets[i].item() - y_pred
                self.weights = self.weights + self.eta * delta * g_layer
                self.bias = self.bias + self.eta * delta

    def predict(self, new_data: np.ndarray) -> float:
        """Predicted scalar value for a single sample."""
        return self.calculate_output(self.gaussian_layer(new_data))

    def predict_batch(self, samples: np.ndarray) -> np.ndarray:
        """Predicted values for a batch of samples."""
        return np.asarray([self.predict(sample) for sample in samples])
