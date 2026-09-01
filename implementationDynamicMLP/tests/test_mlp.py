"""Smoke tests for the dynamic MLP implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_iris_one_hot, load_salary_data
from src.evaluation import accuracy, predict_labels
from src.models import MLP

IRIS_PARAMS = {
    "LEARNING_RATE": 0.01,
    "CODE_OF_ACTIVATION_FUNCTIONS": [4, 1, 3],
    "NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS": [5],
}

LINEAR_PARAMS = {
    "LEARNING_RATE": 0.01,
    "CODE_OF_ACTIVATION_FUNCTIONS": [4, 4],
    "NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS": [],
}


def test_mlp_requires_parameters():
    with pytest.raises(KeyError):
        MLP(np.zeros((5, 2)), np.zeros((5, 1)), {})


def test_mlp_learns_iris():
    data = load_iris_one_hot(seed=42)
    train, test = data[:120], data[120:]
    model = MLP(train[:, 0:4], train[:, 4:7], IRIS_PARAMS, seed=42)
    model.train(epochs=100)
    y_pred = predict_labels(model, test[:, 0:4])
    y_true = test[:, 7].astype(int)
    assert accuracy(y_true, y_pred) >= 0.9


def test_mlp_output_shape():
    data = load_iris_one_hot(seed=42)
    train = data[:120]
    model = MLP(train[:, 0:4], train[:, 4:7], IRIS_PARAMS, seed=42)
    out = model.predict_row(train[0, 0:4])
    assert out.shape == (3,)


def test_mlp_linear_regression_fit():
    data = load_salary_data(seed=42)
    train = data[:24]
    model = MLP(
        train[:, 0].reshape(-1, 1), train[:, 1].reshape(-1, 1), LINEAR_PARAMS, seed=42
    )
    model.train(epochs=100)
    pred = model.predict(train[:, 0].reshape(-1, 1)).ravel()
    # The linear single-layer MLP should approximate the training targets.
    assert np.corrcoef(pred, train[:, 1])[0, 1] > 0.95
