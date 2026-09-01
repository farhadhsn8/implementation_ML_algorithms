"""Weighted connection (branch) between a neuron and one of its inputs.

Every non-input neuron has ``previous_layer_neurons + 1`` branches. The
final branch is the bias branch: it receives a constant input of ``1.0``
so its weight behaves like a proper trainable bias term.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .perceptron import Perceptron

BIAS_INPUT = 1.0


class InputBranch:
    """A single weighted connection feeding into a perceptron."""

    def __init__(self, perceptron: "Perceptron", input_number: int):
        self.input_number = input_number
        self.perceptron = perceptron
        self.reset_weight()
        self.w_new = self.w

    def reset_weight(self) -> None:
        """Initialise the weight (identity for the input layer)."""
        if self.perceptron.layer.layer_index == 0:
            self.w = 1.0
        else:
            self.w = random.uniform(0.0, 1.0)

    def branch_output(self, x: float) -> float:
        """Weighted contribution of input ``x``."""
        return self.w * x

    def update_weight(self) -> None:
        """Stage the delta-rule weight update into ``w_new``."""
        learning_rate = self.perceptron.layer.network.learning_rate
        yi = self._current_input_value()
        self.w_new = self.w + learning_rate * self.perceptron.get_delta() * yi

    def apply_weight_update(self) -> None:
        """Commit the staged ``w_new`` value."""
        self.w = self.w_new

    def _current_input_value(self) -> float:
        """The value flowing into this branch for the current sample.

        The bias branch reads the constant ``BIAS_INPUT``; all other
        branches read the corresponding neuron of the previous layer.
        """
        if self.perceptron.layer.layer_index == 0:
            return self.perceptron.layer.network.current_feature_row()[
                self.input_number
            ]
        previous_output = (
            self.perceptron.layer.get_previous_layer().calculate_layer_output(
                self.perceptron.layer.network.current_feature_row()
            )
        )
        values = list(previous_output) + [BIAS_INPUT]
        return values[self.input_number]
