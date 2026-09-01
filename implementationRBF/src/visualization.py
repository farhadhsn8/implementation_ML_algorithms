"""Plotting helpers for data exploration and results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_clusters(
    features: np.ndarray, cluster_assignments: np.ndarray, title: str = "K-means clusters"
) -> None:
    """Scatter the data colored by cluster assignment."""
    plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=cluster_assignments)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")


def plot_confusion_matrix(
    cm: "np.ndarray", class_names: list[str] = ["0", "1"], cmap: str = "Blues"
) -> None:
    """Render a confusion matrix as a labeled heatmap."""
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, cmap=cmap)


def plot_regression_fit(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "RBF regression",
) -> None:
    """Overlay predicted values against the ground truth."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_true, c="red", label="desired")
    plt.scatter(x, y_pred, c="green", label="predict")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
