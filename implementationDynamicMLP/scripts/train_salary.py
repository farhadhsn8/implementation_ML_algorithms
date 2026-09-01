"""Train a single-layer MLP (linear regression) on the Salary dataset.

Reproduces the notebook workflow:
1. Load, shuffle and split ``Salary_Data.csv`` (24 train / 6 test).
2. Train a 2-layer linear MLP with backpropagation for 100 epochs.
3. Plot predicted vs. desired salary for train and test splits.

Usage:
    python scripts/train_salary.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from src.data_loader import load_salary_data
from src.evaluation import mean_squared_error
from src.models import MLP
from src.visualization import plot_regression_fit

SEED = 42

PARAMS = {
    "LEARNING_RATE": 0.01,
    "CODE_OF_ACTIVATION_FUNCTIONS": [4, 4],  # linear -> linear
    "NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS": [],
}


def main() -> None:
    data = load_salary_data(seed=SEED)
    train, test = data[:24], data[24:]

    model = MLP(
        train[:, 0].reshape(-1, 1),
        train[:, 1].reshape(-1, 1),
        PARAMS,
    )
    model.train(epochs=100)

    for split, name in ((train, "train"), (test, "test")):
        x = split[:, 0]
        y_true = split[:, 1]
        y_pred = model.predict(x.reshape(-1, 1)).ravel()
        plot_regression_fit(x, y_true, y_pred)
        plt.title(f"Salary regression — {name} split")
        print(f"{name} MSE: {mean_squared_error(y_true, y_pred):.2f}")
        plt.show()


if __name__ == "__main__":
    main()
