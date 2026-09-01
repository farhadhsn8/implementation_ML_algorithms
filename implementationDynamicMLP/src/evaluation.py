"""Evaluation utilities for the dynamic MLP project."""

from __future__ import annotations

import numpy as np

from .models import MLP


def predict_labels(model: MLP, features: np.ndarray) -> list[int]:
    """Argmax class predictions for a batch of samples."""
    return [int(np.argmax(model.predict_row(row))) for row in features]


def accuracy(y_true: np.ndarray, y_pred: list[int]) -> float:
    """Fraction of correctly classified samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sum(y_true == y_pred) / len(y_true))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between true and predicted values."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))
