"""A layer of the multi-layer perceptron."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from .activation import ActivationFunction
from .perceptron import Perceptron

if TYPE_CHECKING:
    from .mlp import MLP

INFINITY = math.inf


class Layer:
    """A collection of perceptrons plus a shared activation function.

    Parameters
    ----------
    layer_index : int
        Position of this layer in the network (0 = input layer).
    network : MLP
        The parent network that owns this layer.
    """

    def __init__(self, layer_index: int, network: "MLP"):
        self.network = network
        self.layer_index = layer_index
        self.num_neurons = self._set_number_of_neurons()
        self.activation = ActivationFunction(
            network.parameters["CODE_OF_ACTIVATION_FUNCTIONS"][layer_index]
        )
        self.neurons = [Perceptron(i, self) for i in range(self.num_neurons)]
        self.output = np.full(self.num_neurons, INFINITY)

    def _set_number_of_neurons(self) -> int:
        if self.layer_index == 0:
            return self.network.training_features.shape[1]
        if self.layer_index == self.network.num_layers - 1:
            return self.network.training_labels.shape[1]
        return self.network.parameters["NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS"][
            self.layer_index - 1
        ]

    def reset_output(self) -> None:
        self.output = np.full(self.num_neurons, INFINITY)

    def get_previous_layer(self) -> Optional["Layer"]:
        if self.layer_index == 0:
            return None
        return self.network.layers[self.layer_index - 1]

    def get_next_layer(self) -> Optional["Layer"]:
        if self.layer_index == self.network.num_layers - 1:
            return None
        return self.network.layers[self.layer_index + 1]

    def calculate_layer_output(self, x: np.ndarray) -> np.ndarray:
        """Forward-propagate ``x`` through this layer.

        The result is cached so repeated calls within one forward pass
        are cheap. The input layer simply returns the features unchanged.
        """
        if not np.any(self.output == INFINITY):
            return self.output
        if self.layer_index == 0:
            self.output = np.asarray(x, dtype=float)
            return self.output

        previous = self.get_previous_layer().calculate_layer_output(x)
        outputs = np.empty(self.num_neurons)
        for i, neuron in enumerate(self.neurons):
            outputs[i] = neuron.forward(previous)
        self.output = outputs
        return self.output

    def derivative(self, net: float) -> float:
        return self.activation.derivative(net)

    def update_weights(self, hard_update: bool = False) -> None:
        for neuron in self.neurons:
            neuron.update_weights(hard_update)
