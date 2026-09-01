"""End-to-end Bagging Random Forest experiment on the Iris dataset.

Reproduces the notebook workflow:
1. Load and shuffle the dataset.
2. Exploratory pair plot.
3. Single decision tree (ID3 / C4.5) evaluation on train and test splits.
4. Random forest evaluation + accuracy vs. number-of-trees sweep.
5. Confusion matrices for each evaluation.

Usage:
    python scripts/train_evaluate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data_loader import load_iris, load_iris_frame
from src.evaluation import (
    accuracy,
    decision_tree_predictions,
    random_forest_predictions,
)
from src.models import C45, DecisionTree, ID3, RandomForest
from src.visualization import (
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_pairgrid,
)

SEED = 42


def main() -> None:
    dataset = load_iris(seed=SEED)
    train, test = dataset[:120], dataset[120:]
    train_features, train_targets = train[:, :4], train[:, 4]
    test_features, test_targets = test[:, :4], test[:, 4]

    plot_pairgrid(load_iris_frame())
    plt.show()

    # -- Single decision tree --------------------------------------------
    for method, name in ((ID3, "ID3"), (C45, "C4.5")):
        tree = DecisionTree(train_features, train_targets, purity_threshold=1.0, method=method)
        tree_acc = accuracy(test_targets, decision_tree_predictions(tree, test_features))
        print(f"[{name}] decision tree test accuracy: {tree_acc * 100:.2f} %")
        cm = confusion_matrix(test_targets, decision_tree_predictions(tree, test_features))
        plot_confusion_matrix(cm, cmap="Greens")
        plt.show()

    # -- Random forest ----------------------------------------------------
    forest = RandomForest(
        number_of_trees=20,
        features=train_features,
        targets=train_targets,
        purity_threshold=1.0,
        tree_method=ID3,
        seed=SEED,
    )
    forest_acc = accuracy(
        test_targets, random_forest_predictions(forest, test_features)
    )
    print(f"[Random Forest] test accuracy (20 trees): {forest_acc * 100:.2f} %")
    cm = confusion_matrix(test_targets, random_forest_predictions(forest, test_features))
    plot_confusion_matrix(cm, cmap="Greens")
    plt.show()

    # -- Sweep over number of trees --------------------------------------
    tree_counts = list(range(1, 20, 2))
    accuracies = []
    for n in tree_counts:
        forest = RandomForest(
            number_of_trees=n,
            features=train_features,
            targets=train_targets,
            purity_threshold=1.0,
            tree_method=ID3,
            seed=SEED,
        )
        accuracies.append(
            accuracy(test_targets, random_forest_predictions(forest, test_features))
        )

    plot_accuracy_curve(tree_counts, accuracies, "number of trees", color="green")
    print("Forest size sweep complete — plotting accuracy vs. number of trees.")
    plt.show()


if __name__ == "__main__":
    main()
