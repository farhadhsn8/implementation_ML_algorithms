"""Plotting helpers for data exploration and results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_regression_fit(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xlabel: str = "YearsExperience",
    ylabel: str = "Salary",
) -> None:
    """Overlay predicted values against the ground truth."""
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 4))
    plt.plot(x, y_true, marker="*", label="desired")
    plt.plot(x, y_pred, marker="*", label="predict")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid()
