"""Plotting helpers for data exploration and results."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

IRIS_CLASS_NAMES = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]


def plot_pairgrid(data_frame: pd.DataFrame, hue: str = "species") -> None:
    """Exploratory pair plot of the Iris dataset."""
    grid = sns.PairGrid(data_frame, hue=hue)
    grid.map_diag(sns.histplot)
    grid.map_offdiag(sns.scatterplot)
    grid.add_legend()


def plot_confusion_matrix(
    cm: "np.ndarray",
    class_names: list[str] = IRIS_CLASS_NAMES,
    cmap: str = "Blues",
) -> None:
    """Render a confusion matrix as a labeled heatmap."""
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, cmap=cmap)
