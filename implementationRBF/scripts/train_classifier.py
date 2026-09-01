"""Train an RBF network on the Q4 binary classification task.

Reproduces the notebook workflow:
1. Load and split ``Q4.csv`` (800 train / 200 test).
2. Visualize the K-means cluster assignments (basis centers).
3. Train the RBF classifier with gradient descent.
4. Report confusion matrices and accuracy on train and test splits.

Usage:
    python scripts/train_classifier.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data_loader import load_q4_classification
from src.evaluation import accuracy, classifier_predictions
from src.models import RBFClassifier
from src.visualization import plot_clusters, plot_confusion_matrix

SEED = 42
N_GAUSSIANS = 50
LEARNING_RATE = 0.1
EPOCHS = 40


def main() -> None:
    train, test = load_q4_classification(seed=SEED)
    train_features, train_targets = train[:, :2], train[:, 2]
    test_features, test_targets = test[:, :2], test[:, 2]

    model = RBFClassifier(
        train_features,
        train_targets.reshape(-1, 1),
        number_of_gaussians=N_GAUSSIANS,
        eta=LEARNING_RATE,
        seed=SEED,
    )

    plot_clusters(train_features, model.clusters, title="K-means centers (train)")
    plt.show()

    model.training(EPOCHS)

    for features, targets, name in (
        (train_features, train_targets, "train"),
        (test_features, test_targets, "test"),
    ):
        preds = classifier_predictions(model, features)
        cm = confusion_matrix(targets, preds)
        plot_confusion_matrix(cm, cmap="Greens" if name == "train" else "Blues")
        plt.title(f"RBF classifier — {name} split")
        print(f"{name} accuracy: {accuracy(targets, preds) * 100:.2f} %")
        plt.show()


if __name__ == "__main__":
    main()
