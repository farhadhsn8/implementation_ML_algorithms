"""Gaussian Naive Bayes classifier implemented from scratch.

The classifier assumes feature independence and models each
feature / class pair with a univariate Gaussian distribution.
During prediction it uses log-likelihoods (equivalent to computing
the full product, but numerically stable).
"""

from __future__ import annotations

import math

import numpy as np


class Gaussian:
    """A univariate Gaussian distribution fitted on one feature."""

    def __init__(self, feature_target: np.ndarray):
        self.mean = float(np.mean(feature_target, axis=0)[0])
        self.std = float(np.std(feature_target, axis=0)[0])

    def compute_gaussian_func(self, x: float) -> float:
        """Evaluate the Gaussian probability density at ``x``."""
        return (
            math.exp((((x - self.mean) / self.std) ** 2) / -2)
            / (self.std * math.sqrt(2 * math.pi))
        )


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier.

    Parameters
    ----------
    features : np.ndarray
        Training feature matrix of shape ``(n_samples, n_features)``.
    targets : np.ndarray
        One-hot encoded training labels of shape ``(n_samples, n_classes)``.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets
        self.gaussians = [
            [0 for _ in range(targets.shape[1])] for _ in range(features.shape[1])
        ]

    def fit_model(self) -> None:
        """Fit one Gaussian per (feature, class) combination."""
        for i in range(len(self.gaussians)):
            for j in range(len(self.gaussians[0])):
                feature_target = np.hstack(
                    (self.features[:, i].reshape(-1, 1), self.targets[:, j].reshape(-1, 1))
                )
                self.gaussians[i][j] = Gaussian(
                    feature_target[feature_target[:, 1] == 1, :]
                )

    def predict(self, query_data: np.ndarray) -> tuple[int, np.ndarray]:
        """Predict the class of a sample.

        Returns ``(argmax_class, log_likelihood_vector)``.
        """
        likelihood = np.zeros(self.targets.shape[1])
        for i in range(likelihood.shape[0]):
            prior = np.count_nonzero(self.targets[:, i] == 1) / self.targets.shape[0]
            likelihood[i] = np.log(prior)
            for j in range(self.features.shape[1]):
                likelihood[i] += np.log(
                    self.gaussians[j][i].compute_gaussian_func(query_data[j])
                )
        return int(np.argmax(likelihood)), likelihood
