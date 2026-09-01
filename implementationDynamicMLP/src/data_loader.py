"""Dataset loading helpers shared across the project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_iris_one_hot(seed: int | None = None) -> np.ndarray:
    """Load Iris with one-hot labels: ``[features | one_hot | numeric_label]``."""
    iris = datasets.load_iris()
    features = iris.data
    one_hot = pd.get_dummies(iris.target).to_numpy()
    labels = iris.target.reshape(-1, 1)
    dataset = np.hstack((features, one_hot, labels))
    return shuffle(dataset, random_state=seed)


def load_salary_data(seed: int | None = None) -> np.ndarray:
    """Load ``Salary_Data.csv`` as a shuffled ``(years_experience, salary)`` array."""
    path = PROJECT_ROOT / "data" / "Salary_Data.csv"
    df = pd.read_csv(path)
    values = df[["YearsExperience", "Salary"]].to_numpy()
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(values)
    return values
