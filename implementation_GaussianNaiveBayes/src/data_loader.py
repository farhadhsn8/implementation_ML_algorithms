"""Dataset loading helpers shared across the project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.utils import shuffle


def load_iris_one_hot(seed: int | None = None) -> np.ndarray:
    """Load the Iris dataset with one-hot encoded labels.

    Returns ``[features | one_hot_labels | numeric_label]`` so the array can
    be split / shuffled in one operation, exactly like the notebook.
    """
    iris = datasets.load_iris()
    features = iris.data
    one_hot = pd.get_dummies(iris.target).to_numpy()
    labels = iris.target.reshape(-1, 1)
    dataset = np.hstack((features, one_hot, labels))
    return shuffle(dataset, random_state=seed)


def load_iris_frame() -> "sns.load_dataset":
    """Return the Iris dataset as a Pandas DataFrame (for plotting)."""
    return sns.load_dataset("iris")
