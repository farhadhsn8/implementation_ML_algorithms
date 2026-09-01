"""Train an RBF network on the Q5 regression task.

Reproduces the notebook workflow:
1. Load ``Q5.csv`` (x -> y samples).
2. Train RBF regressors with auto-discovered centers and with a
   user-provided center set / width.
3. Plot predicted vs. desired values and report MSE.

Usage:
    python scripts/train_regressor.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from src.data_loader import load_q5_regression
from src.evaluation import mean_squared_error
from src.models import RBFRegressor
from src.visualization import plot_regression_fit

SEED = 42
LEARNING_RATE = 0.1
EPOCHS = 50


def _evaluate(model: RBFRegressor, data: "np.ndarray", name: str) -> None:
    model.training(EPOCHS)
    x, y_true = data[:, 0], data[:, 1]
    y_pred = model.predict_batch(x.reshape(-1, 1))
    plot_regression_fit(x, y_true, y_pred, title=f"RBF regression — {name}")
    print(f"{name}: MSE = {mean_squared_error(y_true, y_pred):.6f}")
    plt.show()


def main() -> None:
    data = load_q5_regression()

    # Auto-discovered centers via K-means.
    auto = RBFRegressor(
        data[:, 0].reshape(-1, 1),
        data[:, 1].reshape(-1, 1),
        number_of_gaussians=20,
        eta=LEARNING_RATE,
        seed=SEED,
    )
    _evaluate(auto, data, "auto centers (K-means, 20)")

    # User-provided centers + explicit width.
    manual = RBFRegressor(
        data[:, 0].reshape(-1, 1),
        data[:, 1].reshape(-1, 1),
        number_of_gaussians=5,
        eta=LEARNING_RATE,
        centers=data[::9, 0].reshape(-1, 1),
        stds=2.0,
        seed=SEED,
    )
    _evaluate(manual, data, "manual centers + std=2.0")


if __name__ == "__main__":
    main()
