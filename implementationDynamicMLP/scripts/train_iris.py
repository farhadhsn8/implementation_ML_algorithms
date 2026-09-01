"""Train a dynamic MLP on the Iris classification task.

Reproduces the notebook workflow:
1. Load, shuffle and split the Iris dataset (120 train / 30 test).
2. Build a configurable MLP (input -> 5 hidden -> 3 output).
3. Train with backpropagation for 100 epochs.
4. Report per-sample predictions and test accuracy.

Usage:
    python scripts/train_iris.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_iris_one_hot
from src.evaluation import accuracy, predict_labels
from src.models import MLP

SEED = 42

PARAMS = {
    "LEARNING_RATE": 0.01,
    "CODE_OF_ACTIVATION_FUNCTIONS": [4, 1, 3],  # linear -> sigmoid -> relu
    "NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS": [5],
}


def main() -> None:
    dataset = load_iris_one_hot(seed=SEED)
    train, test = dataset[:120], dataset[120:]

    model = MLP(
        train[:, 0:4],
        train[:, 4:7],
        PARAMS,
    )
    model.train(epochs=100)

    y_pred = predict_labels(model, test[:, 0:4])
    y_true = test[:, 7].astype(int)

    for i, row in enumerate(test):
        estimate = model.predict_row(row[0:4])
        print(estimate, row[4:7])

    print(f"\nCorrect / total: {sum(1 for p, t in zip(y_pred, y_true) if p == t)}/{len(y_true)}")
    print(f"Test accuracy: {accuracy(y_true, y_pred) * 100:.2f} %")


if __name__ == "__main__":
    main()
