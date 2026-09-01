"""End-to-end KNN experiment on the Iris dataset.

Reproduces the notebook workflow:
1. Load and shuffle the dataset.
2. Exploratory pair plot.
3. Leave-one-out cross-validation (single K + K sweep).
4. Confusion matrix and accuracy plots.

Usage:
    python scripts/train_evaluate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data_loader import load_iris, load_iris_frame
from src.evaluation import accuracy, leave_one_out_cross_validation
from src.visualization import (
    plot_accuracy_vs_k,
    plot_confusion_matrix,
    plot_pairgrid,
)


def main() -> None:
    dataset = load_iris()
    features = dataset[:, :4]
    targets = dataset[:, 4]

    plot_pairgrid(load_iris_frame())
    plt.show()

    # -- Single K evaluation with leave-one-out cross-validation ----------
    y_pred = leave_one_out_cross_validation(features, targets, k=20, distance_type=1)
    cm = confusion_matrix(targets, y_pred)
    plot_confusion_matrix(cm)
    print(f"Leave-one-out accuracy (K=20, Manhattan): {accuracy(targets, y_pred) * 100:.2f} %")
    plt.show()

    # -- Sweep over K ------------------------------------------------------
    k_values = list(range(1, 120, 5))
    accuracies = []
    for k in k_values:
        predictions = leave_one_out_cross_validation(
            features, targets, k=k, distance_type=1
        )
        accuracies.append(accuracy(targets, predictions))

    plot_accuracy_vs_k(k_values, accuracies)
    print("K sweep complete — plotting accuracy vs. K.")
    plt.show()


if __name__ == "__main__":
    main()
