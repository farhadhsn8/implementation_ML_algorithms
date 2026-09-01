"""A single perceptron (neuron) inside an MLP layer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .input_branch import BIAS_INPUT, InputBranch

if TYPE_CHECKING:
    from .layer import Layer

INFINITY = math.inf


class Perceptron:
    """A neuron with one weighted input branch per incoming connection."""

    def __init__(self, perceptron_index: int, layer: "Layer"):
        self.perceptron_index = perceptron_index
        self.layer = layer
        self.num_inputs = self._get_number_of_inputs()
        self.input_branches: list[InputBranch] = []
        self.delta = INFINITY
        self._build_inputs()

    def _build_inputs(self) -> None:
        self.input_branches = [
            InputBranch(self, i) for i in range(self.num_inputs)
        ]

    def _get_number_of_inputs(self) -> int:
        if self.layer.layer_index == 0:
            return 1
        return self.layer.get_previous_layer().num_neurons + 1  # + bias branch

    def reset_delta(self) -> None:
        self.delta = INFINITY

    def forward(self, x: np.ndarray) -> float:
        """Activation of the neuron given the incoming vector ``x``."""
        return self.layer.activation.apply(self.net_output(x))

    def net_output(self, x: np.ndarray) -> float:
        """Weighted sum of inputs (a constant ``1`` acts as the bias)."""
        inputs = np.concatenate((np.asarray(x, dtype=float), [BIAS_INPUT]))
        return sum(
            branch.branch_output(inputs[i]) for i, branch in enumerate(self.input_branches)
        )

    def get_delta(self) -> float:
        """The error gradient of this neuron (cached after first call)."""
        if self.delta != INFINITY:
            return self.delta
        desired_output = 0.0
        if self.layer.layer_index == self.layer.network.num_layers - 1:
            desired_output = self.layer.network.current_label_row()[self.perceptron_index]
        x = self._input_vector_for_delta()
        self.delta = self._calculate_delta(x, desired_output)
        return self.delta

    def _input_vector_for_delta(self) -> np.ndarray:
        if self.layer.layer_index == 0:
            return self.layer.network.current_feature_row()
        return self.layer.get_previous_layer().calculate_layer_output(
            self.layer.network.current_feature_row()
        )

    def _calculate_delta(self, x: np.ndarray, desired_output: float) -> float:
        net = self.net_output(x)
        if self.layer.layer_index == self.layer.network.num_layers - 1:
            # Output layer: delta = f'(net) * (target - output)
            return self.layer.derivative(net) * (desired_output - self.forward(x))
        # Hidden layer: delta = f'(net) * sum(w * delta of next layer)
        sigma = 0.0
        for perceptron in self.layer.get_next_layer().neurons:
            sigma += (
                perceptron.input_branches[self.perceptron_index].w
                * perceptron.get_delta()
            )
        return self.layer.derivative(net) * sigma

    def update_weights(self, hard_update: bool = False) -> None:
        """Stage (False) or commit (True) weight updates for this neuron."""
        for branch in self.input_branches:
            if hard_update:
                branch.apply_weight_update()
            else:
                branch.update_weight()
