"""Evaluation utilities: cross-validation and accuracy."""

from __future__ import annotations

import numpy as np

from .models import KNN


def leave_one_out_cross_validation(
    features: np.ndarray,
    targets: np.ndarray,
    k: int,
    distance_type: int = 0,
) -> list[int]:
    """Run leave-one-out cross-validation for the KNN classifier.

    For every sample the model is re-trained on all the remaining samples
    and the held-out sample is classified. Returns the predicted labels.
    """
    predictions: list[int] = []
    for i in range(features.shape[0]):
        train_mask = np.arange(len(features)) != i
        model = KNN(
            features[train_mask],
            targets[train_mask],
            features[i],
            k=k,
            distance_type=distance_type,
        )
        result = model.predict_class_of_query_data()
        predictions.append(result["popular_class"])
    return predictions


def accuracy(y_true: np.ndarray, y_pred: list[int]) -> float:
    """Fraction of correctly classified samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sum(y_true == y_pred) / len(y_true))
