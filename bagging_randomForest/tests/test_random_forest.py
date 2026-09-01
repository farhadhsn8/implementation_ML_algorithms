"""Smoke tests for the Bagging Random Forest implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_iris
from src.evaluation import accuracy, decision_tree_predictions
from src.models import C45, DecisionTree, ID3, RandomForest


@pytest.fixture
def dataset():
    data = load_iris(seed=42)
    train, test = data[:120], data[120:]
    return train[:, :4], train[:, 4], test[:, :4], test[:, 4]


def test_decision_tree_predicts_all_labels(dataset):
    train_features, train_targets, test_features, test_targets = dataset
    tree = DecisionTree(train_features, train_targets, purity_threshold=1.0, method=ID3)
    preds = decision_tree_predictions(tree, test_features)
    assert len(preds) == len(test_targets)
    assert all(0 <= p <= 2 for p in preds)


def test_c45_tree_high_accuracy(dataset):
    train_features, train_targets, test_features, test_targets = dataset
    tree = DecisionTree(train_features, train_targets, purity_threshold=1.0, method=C45)
    preds = decision_tree_predictions(tree, test_features)
    assert accuracy(test_targets, preds) >= 0.9


def test_random_forest_predict(dataset):
    train_features, train_targets, test_features, test_targets = dataset
    forest = RandomForest(
        number_of_trees=10,
        features=train_features,
        targets=train_targets,
        purity_threshold=1.0,
        seed=42,
    )
    preds = forest.predict_batch(test_features)
    assert len(preds) == len(test_targets)
    assert accuracy(test_targets, preds) >= 0.9


def test_random_forest_reproducible_with_seed(dataset):
    train_features, train_targets, *_ = dataset
    first = RandomForest(5, train_features, train_targets, 1.0, seed=7)
    second = RandomForest(5, train_features, train_targets, 1.0, seed=7)
    sample = train_features[0]
    assert first.predict(sample) == second.predict(sample)
