"""A flexible multi-layer perceptron trained with backpropagation.

The architecture is fully configurable through a ``parameters`` dict:

- ``LEARNING_RATE`` : float — step size for gradient descent.
- ``CODE_OF_ACTIVATION_FUNCTIONS`` : list[int] — one activation code per
  layer (``1``=sigmoid, ``2``=tanh, ``3``=relu, ``4``=linear).
- ``NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS`` : list[int] — neuron count
  for each hidden layer (empty list for a single-layer network).

The input layer size equals the number of features and the output layer
size equals the number of labels. Hidden layers are sized by the config.
"""

from __future__ import annotations

import random

import numpy as np

from .layer import Layer

_REQUIRED_KEYS = (
    "LEARNING_RATE",
    "CODE_OF_ACTIVATION_FUNCTIONS",
    "NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS",
)


class MLP:
    """A multi-layer perceptron built and trained from scratch.

    Parameters
    ----------
    training_features : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    training_labels : np.ndarray
        Label matrix of shape ``(n_samples, n_outputs)``.
    parameters : dict
        Network configuration (see module docstring).
    seed : int | None, default=None
        Optional random seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        training_features: np.ndarray,
        training_labels: np.ndarray,
        parameters: dict,
        seed: int | None = None,
    ):
        missing = [key for key in _REQUIRED_KEYS if key not in parameters]
        if missing:
            raise KeyError(f"Missing MLP parameters: {missing}")

        self.parameters = parameters
        self.training_features = np.asarray(training_features, dtype=float)
        self.training_labels = np.asarray(training_labels, dtype=float)
        self.sliding_head = 0
        self.learning_rate = self.parameters["LEARNING_RATE"]
        self.num_layers = len(self.parameters["CODE_OF_ACTIVATION_FUNCTIONS"])
        if seed is not None:
            random.seed(seed)
        # Pre-allocate so layers can reference each other during construction.
        self.layers: list[Layer] = [None] * self.num_layers  # type: ignore[list-item]
        for i in range(self.num_layers):
            self.layers[i] = Layer(i, self)

    # -- Public API ------------------------------------------------------

    def train(self, epochs: int = 1) -> None:
        """Train the network for a number of epochs over the full dataset."""
        for _ in range(epochs):
            self.sliding_head = 0
            for _ in range(self.training_features.shape[0]):
                self.backpropagate()
                self.sliding_head += 1

    def predict_row(self, x: np.ndarray) -> np.ndarray:
        """Forward-pass a single sample and return the output vector."""
        self.reset_caches()
        return self.layers[self.num_layers - 1].calculate_layer_output(x)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict outputs for a batch of samples."""
        return np.asarray([self.predict_row(row) for row in features])

    def predict_class(self, x: np.ndarray) -> int:
        """Argmax class index for a single sample."""
        return int(np.argmax(self.predict_row(x)))

    def current_feature_row(self) -> np.ndarray:
        return self.training_features[self.sliding_head]

    def current_label_row(self) -> np.ndarray:
        return self.training_labels[self.sliding_head]

    def reset_caches(self) -> None:
        """Clear cached activations and deltas for a fresh forward pass."""
        for layer in self.layers:
            layer.reset_output()
            for neuron in layer.neurons:
                neuron.reset_delta()

    def backpropagate(self) -> None:
        """One gradient-descent step on the current training sample."""
        for layer in self.layers[:0:-1]:
            layer.update_weights(hard_update=False)
        for layer in self.layers[:0:-1]:
            layer.update_weights(hard_update=True)
        self.reset_caches()

    def clear_all(self) -> None:
        """Reset every weight of the network."""
        for layer in self.layers:
            layer.reset_output()
            for neuron in layer.neurons:
                neuron.reset_delta()
                for branch in neuron.input_branches:
                    branch.reset_weight()
