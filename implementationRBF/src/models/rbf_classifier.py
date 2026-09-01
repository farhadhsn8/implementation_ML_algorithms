"""Radial Basis Function network for binary classification.

A Gaussian basis-function layer projects the input into a higher
dimensional space (one basis per cluster center), then a single sigmoid
output neuron combines the basis responses with a learnable weight vector.
Weights are trained online with gradient descent (the delta rule).

This is a faithful, production-friendly port of ``RBFNetClassifer`` from
``RBF.ipynb``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from ..kmeans import KMeans


class RBFClassifier:
    """RBF network that classifies into one of two classes (binary)."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        number_of_gaussians: int,
        eta: float = 0.01,
        seed: int | None = None,
    ):
        self.features = np.asarray(features, dtype=float)
        self.targets = np.asarray(targets, dtype=float)
        self.number_of_gaussians = number_of_gaussians
        self.eta = eta
        self.seed = seed

        kmeans = KMeans(self.features, number_of_gaussians, seed=seed)
        self.clusters, self.centers = kmeans.fit()
        self.stds = self.calculate_std()
        rng = np.random.default_rng(seed)
        self.bias = float(rng.random())
        self.weights = rng.random(number_of_gaussians)

    # -- Layers ----------------------------------------------------------

    def gaussian_layer(self, x: np.ndarray) -> np.ndarray:
        """Activation of every Gaussian basis function for input ``x``."""
        distances = np.linalg.norm(self.centers - x, axis=1)
        return np.exp(-((distances / self.stds) ** 2))

    def calculate_output(self, g_layer: np.ndarray) -> float:
        """Sigmoid output neuron combining the basis responses."""
        return float(expit(np.dot(g_layer, self.weights) + self.bias))

    def calculate_std(self) -> float:
        """Gaussian width from the maximum distance between centers."""
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
                delta = y_pred * (1 - y_pred) * (self.targets[i].item() - y_pred)
                self.weights = self.weights + self.eta * delta * g_layer
                self.bias = self.bias + self.eta * delta

    def predict(self, new_data: np.ndarray) -> float:
        """Continuous output in ``[0, 1]`` for a single sample."""
        return self.calculate_output(self.gaussian_layer(new_data))

    def predict_label(self, new_data: np.ndarray, threshold: float = 0.5) -> int:
        """Hard binary prediction for a single sample."""
        return int(self.predict(new_data) > threshold)

    def predict_batch(self, samples: np.ndarray, threshold: float = 0.5) -> list[int]:
        """Hard binary predictions for a batch of samples."""
        return [self.predict_label(sample, threshold) for sample in samples]
