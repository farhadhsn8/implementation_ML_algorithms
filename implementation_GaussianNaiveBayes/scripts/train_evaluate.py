"""End-to-end Gaussian Naive Bayes experiment on the Iris dataset.

Reproduces the notebook workflow:
1. Load, shuffle and split the Iris dataset (120 train / 30 test).
2. Exploratory pair plot.
3. Fit the Gaussian NB classifier on one-hot labels.
4. Confusion matrices and accuracy on train and test splits.

Usage:
    python scripts/train_evaluate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data_loader import load_iris_frame, load_iris_one_hot
from src.evaluation import accuracy, evaluate_model
from src.models import GaussianNaiveBayes
from src.visualization import plot_confusion_matrix, plot_pairgrid

SEED = 42


def main() -> None:
    dataset = load_iris_one_hot(seed=SEED)
    train, test = dataset[:120], dataset[120:]

    plot_pairgrid(load_iris_frame())
    plt.show()

    model = GaussianNaiveBayes(train[:, 0:4], train[:, 4:7])
    model.fit_model()

    for split, name in ((train, "train"), (test, "test")):
        y_pred = evaluate_model(model, split[:, 0:4])
        y_true = split[:, 7].astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm)
        plt.title(f"Gaussian Naive Bayes — {name} split")
        print(f"{name} accuracy: {accuracy(y_true, y_pred) * 100:.2f} %")
        plt.show()


if __name__ == "__main__":
    main()
