"""Random Forest (bagging of decision trees) implemented from scratch.

Each tree is trained on a bootstrap sample of the dataset and every tree
votes for the final prediction. This is a faithful port of the reference
implementation in ``bagging_randomForest.ipynb``.
"""

from __future__ import annotations

import random

import numpy as np

from .decision_tree import C45, DecisionTree, ID3


class RandomForest:
    """An ensemble of decision trees trained with bootstrap aggregating.

    Parameters
    ----------
    number_of_trees : int
        Number of trees in the forest.
    features : np.ndarray
        Training feature matrix of shape ``(n_samples, n_features)``.
    targets : np.ndarray
        Training label vector of shape ``(n_samples,)``.
    purity_threshold : float
        Purity required for a tree node to become a leaf.
    tree_method : int, default=ID3
        Underlying tree algorithm: ``ID3`` or ``C45``.
    seed : int | None, default=None
        Optional random seed for reproducible bootstrap sampling.
    """

    def __init__(
        self,
        number_of_trees: int,
        features: np.ndarray,
        targets: np.ndarray,
        purity_threshold: float,
        tree_method: int = ID3,
        seed: int | None = None,
    ):
        self.number_of_trees = number_of_trees
        self.features = features
        self.targets = targets
        self.purity_threshold = purity_threshold
        self.tree_method = tree_method
        self.trees: list[DecisionTree] = []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._build_forest()

    def _bootstrap(self) -> np.ndarray:
        """Sample a bootstrap dataset with replacement."""
        dataset = np.hstack((self.features, self.targets.reshape(-1, 1)))
        indices = np.random.randint(0, dataset.shape[0], size=dataset.shape[0])
        return dataset[indices]

    def _build_forest(self) -> None:
        for _ in range(self.number_of_trees):
            sample = self._bootstrap()
            tree = DecisionTree(
                sample[:, :-1], sample[:, -1], self.purity_threshold, self.tree_method
            )
            self.trees.append(tree)

    def predict(self, query: np.ndarray) -> int:
        """Predict a label by majority vote across all trees."""
        votes = [tree.predict(query) for tree in self.trees]
        counts = np.bincount(np.asarray(votes, dtype=int))
        return int(counts.argmax())

    def predict_batch(self, samples: np.ndarray) -> list[int]:
        """Predict labels for a batch of samples."""
        return [self.predict(sample) for sample in samples]
