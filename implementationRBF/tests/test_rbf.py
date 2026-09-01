"""Smoke tests for the RBF network implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_q4_classification, load_q5_regression
from src.evaluation import accuracy, mean_squared_error
from src.kmeans import KMeans
from src.models import RBFClassifier, RBFRegressor


def test_kmeans_returns_expected_shape():
    features = np.random.default_rng(0).random((50, 2))
    clusters, centers = KMeans(features, k=5, seed=42).fit()
    assert clusters.shape == (50,)
    assert centers.shape == (5, 2)


def test_rbf_classifier_predicts_binary():
    train, test = load_q4_classification(seed=42)
    model = RBFClassifier(
        train[:, :2],
        train[:, 2].reshape(-1, 1),
        number_of_gaussians=20,
        eta=0.1,
        seed=42,
    )
    model.training(10)
    preds = model.predict_batch(test[:, :2])
    assert all(p in (0, 1) for p in preds)
    assert accuracy(test[:, 2], preds) >= 0.7


def test_rbf_classifier_output_in_unit_range():
    train, _ = load_q4_classification(seed=42)
    model = RBFClassifier(train[:, :2], train[:, 2].reshape(-1, 1), 10, eta=0.1, seed=42)
    model.training(5)
    out = model.predict(train[0, :2])
    assert 0.0 <= out <= 1.0


def test_rbf_regressor_reduces_error_after_training():
    data = load_q5_regression()
    model = RBFRegressor(
        data[:, 0].reshape(-1, 1),
        data[:, 1].reshape(-1, 1),
        number_of_gaussians=20,
        eta=0.1,
        seed=42,
    )
    before = mean_squared_error(
        data[:, 1], model.predict_batch(data[:, 0].reshape(-1, 1))
    )
    model.training(50)
    after = mean_squared_error(
        data[:, 1], model.predict_batch(data[:, 0].reshape(-1, 1))
    )
    assert after < before


def test_rbf_regressor_with_manual_centers():
    data = load_q5_regression()
    model = RBFRegressor(
        data[:, 0].reshape(-1, 1),
        data[:, 1].reshape(-1, 1),
        number_of_gaussians=5,
        eta=0.1,
        centers=data[::9, 0].reshape(-1, 1),
        stds=2.0,
        seed=42,
    )
    model.training(50)
    assert np.isfinite(model.predict_batch(data[:, 0].reshape(-1, 1))).all()
