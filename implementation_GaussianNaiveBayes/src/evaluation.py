"""Evaluation utilities for the Gaussian Naive Bayes project."""

from __future__ import annotations

import numpy as np

from .models import GaussianNaiveBayes


def evaluate_model(model: GaussianNaiveBayes, features: np.ndarray) -> list[int]:
    """Classify every sample and return the predicted label vector."""
    return [model.predict(row)[0] for row in features]


def accuracy(y_true: np.ndarray, y_pred: list[int]) -> float:
    """Fraction of correctly classified samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sum(y_true == y_pred) / len(y_true))
