"""Dataset loading helpers shared across the project."""

from __future__ import annotations

import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_iris(seed: int | None = None) -> np.ndarray:
    """Load the Iris dataset and return ``[features | label]``.

    The label column is appended as the last column so the whole array can
    be split / shuffled in one operation, exactly like the notebook.
    """
    iris = datasets.load_iris()
    features = iris.data
    targets = iris.target.reshape(-1, 1)
    dataset = np.hstack((features, targets))
    return shuffle(dataset, random_state=seed)


def train_test_split_dataset(
    dataset: np.ndarray, test_size: float = 0.2, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``[features | label]`` array into train and test sets.

    The label column (last column) is kept untouched so every row stays
    aligned, mirroring the notebook's manual slicing.
    """
    train, test = train_test_split(
        dataset, test_size=test_size, shuffle=False, random_state=seed
    )
    return train, test


def load_iris_frame() -> "sns.load_dataset":
    """Return the Iris dataset as a Pandas DataFrame (for plotting)."""
    return sns.load_dataset("iris")
