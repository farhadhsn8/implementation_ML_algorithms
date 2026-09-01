"""Smoke tests for the Gaussian Naive Bayes implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_iris_one_hot
from src.evaluation import accuracy, evaluate_model
from src.models import Gaussian, GaussianNaiveBayes


def test_gaussian_pdf_integrates_to_one():
    sample = np.random.default_rng(0).normal(loc=5, scale=1, size=(2000, 1))
    gauss = Gaussian(sample)
    xs = np.linspace(sample.min(), sample.max(), 2000)
    integral = np.trapz([gauss.compute_gaussian_func(x) for x in xs], xs)
    assert abs(integral - 1.0) < 0.02


def test_gnb_high_accuracy_on_iris():
    data = load_iris_one_hot(seed=42)
    train, test = data[:120], data[120:]
    model = GaussianNaiveBayes(train[:, 0:4], train[:, 4:7])
    model.fit_model()
    y_pred = evaluate_model(model, test[:, 0:4])
    y_true = test[:, 7].astype(int)
    assert accuracy(y_true, y_pred) >= 0.9


def test_gnb_predict_returns_class_and_likelihoods():
    data = load_iris_one_hot(seed=42)
    train = data[:120]
    model = GaussianNaiveBayes(train[:, 0:4], train[:, 4:7])
    model.fit_model()
    cls, likelihood = model.predict(train[0, 0:4])
    assert 0 <= cls <= 2
    assert likelihood.shape == (3,)
