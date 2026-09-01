"""Dataset loading helpers shared across the project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_q4_classification(
    test_size: float = 0.2, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``Q4.csv`` (2D points + binary label) as train/test arrays.

    Returns ``(train, test)`` where each array holds ``[x, y, label]``.
    """
    path = PROJECT_ROOT / "Q4.csv"
    df = pd.read_csv(path)
    data = df[["x", "y", "D"]].to_numpy()
    return train_test_split(data, test_size=test_size, random_state=seed)


def load_q5_regression() -> np.ndarray:
    """Load ``Q5.csv`` (x -> y samples) as a ``(x, y)`` array."""
    path = PROJECT_ROOT / "Q5.csv"
    df = pd.read_csv(path)
    return df[["x", "y"]].to_numpy()
