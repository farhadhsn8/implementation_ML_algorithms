"""Dataset loading helpers shared across the project."""

from __future__ import annotations

import numpy as np
import seaborn as sns
from sklearn import datasets
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


def load_iris_frame() -> "sns.load_dataset":
    """Return the Iris dataset as a Pandas DataFrame (for plotting)."""
    return sns.load_dataset("iris")
