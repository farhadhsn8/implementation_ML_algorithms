"""Smoke tests for the K-Nearest Neighbors implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_iris
from src.evaluation import accuracy, leave_one_out_cross_validation
from src.models import COSINE, MANHATTAN, KNN


def test_knn_returns_popular_class():
    data = load_iris(seed=42)
    model = KNN(data[:100, :4], data[:100, 4], data[0, :4], k=3)
    result = model.predict_class_of_query_data()
    assert "popular_class" in result
    assert 0 <= result["popular_class"] <= 2


def test_knn_leave_one_out_high_accuracy():
    data = load_iris(seed=42)
    y_pred = leave_one_out_cross_validation(data[:, :4], data[:, 4], k=20, distance_type=MANHATTAN)
    assert accuracy(data[:, 4], y_pred) >= 0.9


def test_knn_supports_multiple_distance_metrics():
    data = load_iris(seed=42)
    for metric in (0, MANHATTAN, COSINE):
        model = KNN(data[:50, :4], data[:50, 4], data[0, :4], k=1, distance_type=metric)
        assert model.predict_class_of_query_data()["popular_class"] in (0, 1, 2)


def test_knn_invalid_distance_raises():
    data = load_iris(seed=42)
    model = KNN(data[:50, :4], data[:50, 4], data[0, :4], k=1, distance_type=99)
    with pytest.raises(ValueError):
        model.predict_class_of_query_data()
