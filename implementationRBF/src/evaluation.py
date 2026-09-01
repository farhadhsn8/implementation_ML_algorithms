"""Evaluation utilities for the RBF network project."""

from __future__ import annotations

import numpy as np

from .models import RBFClassifier, RBFRegressor


def accuracy(y_true: np.ndarray, y_pred: list[int]) -> float:
    """Fraction of correctly classified samples."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.sum(y_true == y_pred) / len(y_true))


def classifier_predictions(
    model: RBFClassifier, features: np.ndarray, threshold: float = 0.5
) -> list[int]:
    """Binary predictions for a batch of samples."""
    return model.predict_batch(features, threshold)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between true and predicted values."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def regressor_predictions(model: RBFRegressor, features: np.ndarray) -> np.ndarray:
    """Predictions for a batch of samples."""
    return model.predict_batch(features)
