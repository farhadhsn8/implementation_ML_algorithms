"""Evaluation utilities for the Random Forest project."""

from __future__ import annotations

import numpy as np

from .models import DecisionTree, RandomForest


def accuracy(y_true: np.ndarray, y_pred: list[int]) -> float:
    """Fraction of correctly classified samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sum(y_true == y_pred) / len(y_true))


def decision_tree_predictions(
    tree: DecisionTree, features: np.ndarray
) -> list[int]:
    """Classify every sample with a single decision tree."""
    return tree.predict_batch(features)


def random_forest_predictions(
    forest: RandomForest, features: np.ndarray
) -> list[int]:
    """Classify every sample with a trained random forest."""
    return forest.predict_batch(features)
