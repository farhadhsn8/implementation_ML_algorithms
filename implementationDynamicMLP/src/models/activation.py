"""Activation functions and their derivatives.

Each layer of the MLP is configured with an activation-function code:

========  ==========
Code      Function
========  ==========
1         Sigmoid
2         Tanh
3         ReLU
4         Linear
========  ==========
"""

from __future__ import annotations

import math

SIGMOID = 1
TANH = 2
RELU = 3
LINEAR = 4

_FUNCTIONS = {SIGMOID: "sigmoid", TANH: "tanh", RELU: "relu", LINEAR: "linear"}


class ActivationFunction:
    """Applies an activation function and its derivative."""

    def __init__(self, function_code: int):
        if function_code not in _FUNCTIONS:
            raise ValueError(
                f"Unknown activation code {function_code!r}. "
                f"Expected one of {sorted(_FUNCTIONS)}."
            )
        self.function_code = function_code

    def apply(self, x: float) -> float:
        """Forward pass of the activation function."""
        if self.function_code == SIGMOID:
            return self.sigmoid(x)
        if self.function_code == TANH:
            return self.tanh(x)
        if self.function_code == RELU:
            return self.relu(x)
        return self.linear(x)

    def derivative(self, net: float) -> float:
        """Gradient of the activation function evaluated at ``net``."""
        if self.function_code == SIGMOID:
            sig = self.sigmoid(net)
            return (1.0 - sig) * sig
        if self.function_code == TANH:
            return 1.0 - self.tanh(net) ** 2
        if self.function_code == RELU:
            return 0.0 if net < 0 else 1.0
        return 1.0

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def tanh(x: float) -> float:
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)

    @staticmethod
    def linear(x: float) -> float:
        return x
