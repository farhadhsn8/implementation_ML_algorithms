"""Decision tree algorithms (ID3 and C4.5) implemented from scratch.

The trees use information gain (ID3) or the gain-ratio criterion (C4.5)
to pick the best split feature, and a fixed midpoint threshold to binarise
continuous features. Growing stops once a node reaches a configured purity
threshold, which turns it into a leaf.

This module is a faithful, production-friendly port of the reference
implementation in ``bagging_randomForest.ipynb``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

ID3 = 0
C45 = 1


class TreeNode:
    """A single node in a decision tree.

    Parameters
    ----------
    name : int
        Unique identifier for the node.
    features : np.ndarray
        Feature matrix of the samples reaching this node.
    targets : np.ndarray
        Label vector of the samples reaching this node.
    purity_threshold : float
        Minimum fraction of the dominant class required for the node
        to become a terminal (leaf) node.
    """

    def __init__(
        self,
        name: int,
        features: np.ndarray,
        targets: np.ndarray,
        purity_threshold: float,
    ):
        self.node_name = name
        self.purity_threshold = purity_threshold
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None
        self.features = features
        self.targets = targets
        self.accuracy = self.compute_accuracy()
        self.label: Optional[int] = None
        self.is_terminal_node = self.is_terminal_node()

    def compute_accuracy(self) -> float:
        """Fraction of samples belonging to the dominant class."""
        if self.targets.shape[0] == 0:
            return 1.0
        counts = np.bincount(self.targets.astype(int))
        return float(counts.max() / self.targets.shape[0])

    def is_terminal_node(self) -> bool:
        """A node is terminal when its samples are pure enough."""
        if self.targets.shape[0] == 0:
            return True
        if self.accuracy >= self.purity_threshold:
            self.label = int(np.bincount(self.targets.astype(int)).argmax())
            return True
        return False

    def find_best_threshold(self, feature_index: int) -> float:
        """Midpoint between the min and max feature value (binary split)."""
        if self.targets.shape[0] == 0:
            return 0.0
        column = self.features[:, feature_index]
        return float((column.min() + column.max()) / 2.0)


class ID3Node(TreeNode):
    """ID3 tree node: splits on the feature with the highest information gain."""

    def __init__(
        self,
        name: int,
        features: np.ndarray,
        targets: np.ndarray,
        purity_threshold: float,
    ):
        super().__init__(name, features, targets, purity_threshold)
        self.best_feature_index = self.find_best_feature()
        self.threshold = self.find_best_threshold(self.best_feature_index)
        left, right = self.compute_children_data()
        self.left_features, self.left_targets = left[:, :-1], left[:, -1]
        self.right_features, self.right_targets = right[:, :-1], right[:, -1]

    def find_best_feature(self) -> int:
        """Feature with the highest information gain."""
        best_gain, best_index = -1.0, 0
        for i in range(self.features.shape[1]):
            gain = self.compute_info_gain(i)
            if gain >= best_gain:
                best_gain, best_index = gain, i
        return best_index

    def compute_info_i(self) -> float:
        """Entropy of the labels at this node."""
        entropy = 0.0
        for count in np.bincount(self.targets.astype(int)):
            if count == 0:
                continue
            p = count / self.targets.shape[0]
            entropy -= p * math.log2(p)
        return entropy

    def compute_info_a(self, feature_index: int) -> float:
        """Expected entropy after splitting on ``feature_index``."""
        if self.targets.shape[0] == 0:
            return 1.0
        threshold = self.find_best_threshold(feature_index)
        column = self.features[:, feature_index]
        smaller = self.targets[column <= threshold]
        bigger = self.targets[column > threshold]
        entropy = 0.0
        for subset in (smaller, bigger):
            if subset.shape[0] == 0:
                continue
            for count in np.bincount(subset.astype(int)):
                if count == 0:
                    continue
                p = count / subset.shape[0]
                entropy -= p * math.log2(p) * (subset.shape[0] / self.targets.shape[0])
        return entropy

    def compute_info_gain(self, feature_index: int) -> float:
        """Information gain of splitting on ``feature_index``."""
        return self.compute_info_i() - self.compute_info_a(feature_index)

    def compute_children_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Split the node data into left (<=) and right (>) subsets."""
        data = np.hstack((self.features, self.targets.reshape(-1, 1)))
        column = data[:, self.best_feature_index]
        return data[column <= self.threshold], data[column > self.threshold]


class C45Node(ID3Node):
    """C4.5 tree node: splits on the feature with the highest gain ratio."""

    def find_best_feature(self) -> int:
        best_ratio, best_index = -1.0, 0
        for i in range(self.features.shape[1]):
            ratio = self.compute_gain_ratio(i)
            if ratio >= best_ratio:
                best_ratio, best_index = ratio, i
        return best_index

    def compute_split_info(self, feature_index: int) -> float:
        """Split information (denominator of the gain ratio)."""
        if self.targets.shape[0] == 0:
            return 1.0
        threshold = self.find_best_threshold(feature_index)
        column = self.features[:, feature_index]
        n_smaller = int(np.sum(column <= threshold))
        n_bigger = int(np.sum(column > threshold))
        split = 0.0
        for n in (n_smaller, n_bigger):
            if n == 0:
                continue
            p = n / self.targets.shape[0]
            split -= p * math.log2(p)
        return split

    def compute_gain_ratio(self, feature_index: int) -> float:
        """Gain ratio of splitting on ``feature_index``."""
        split_info = self.compute_split_info(feature_index)
        if split_info == 0.0:
            return 0.0
        return self.compute_info_gain(feature_index) / split_info


class DecisionTree:
    """A binary decision tree built with ID3 or C4.5.

    Parameters
    ----------
    features : np.ndarray
        Training feature matrix of shape ``(n_samples, n_features)``.
    targets : np.ndarray
        Training label vector of shape ``(n_samples,)``.
    purity_threshold : float
        Purity required for a node to become a leaf (e.g. ``1.0``).
    method : int, default=ID3
        ``ID3`` or ``C45``.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        purity_threshold: float,
        method: int = ID3,
    ):
        self.number_of_nodes = 0
        self.method = method
        self.node_cls = C45Node if method == C45 else ID3Node
        self.head = self.node_cls(
            self._next_node_name(), features, targets, purity_threshold
        )
        self._develop_tree(self.head)

    def _next_node_name(self) -> int:
        self.number_of_nodes += 1
        return self.number_of_nodes

    def _develop_tree(self, node: TreeNode) -> None:
        if node.node_name >= 500:
            node.label = int(np.bincount(node.targets.astype(int)).argmax())
            node.is_terminal_node = True
        if not node.is_terminal_node:
            node.left = self.node_cls(
                self._next_node_name(),
                node.left_features,
                node.left_targets,
                node.purity_threshold,
            )
            node.right = self.node_cls(
                self._next_node_name(),
                node.right_features,
                node.right_targets,
                node.purity_threshold,
            )
            self._develop_tree(node.left)
            self._develop_tree(node.right)

    def predict(self, query: np.ndarray, node: Optional[TreeNode] = None) -> int:
        """Predict the label of a single sample by walking the tree."""
        node = node if node is not None else self.head
        if node.is_terminal_node:
            return 0 if node.label is None else int(node.label)
        if query[node.best_feature_index] < node.threshold:
            return self.predict(query, node.left)
        return self.predict(query, node.right)

    def predict_batch(self, samples: np.ndarray) -> list[int]:
        """Predict labels for a batch of samples."""
        return [self.predict(sample) for sample in samples]

    def show_tree(self, node: Optional[TreeNode] = None, layer: int = 0) -> None:
        """Pretty-print the tree structure."""
        node = node if node is not None else self.head
        print(f"layer {layer} | node {node.node_name} | feature {node.best_feature_index}")
        if not node.is_terminal_node:
            self.show_tree(node.left, layer + 1)
            self.show_tree(node.right, layer + 1)
