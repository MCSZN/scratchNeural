from typing import Any, Callable
import numpy as np

arr = np.ndarray[np.generic, Any]
f16 = np.float16


def linear(inputs: arr, weights: arr, bias: f16) -> f16:
    """linear computation"""
    out: f16 = np.dot(inputs, weights) + bias
    return out


def sigmoid(inputs: f16) -> f16:
    """basic activation function maps number between 0 & 1"""
    out: f16 = 1 / (1 + np.exp(-inputs))
    return out


def relu(inputs: f16) -> f16:
    """most popular activation function maps between 0 and +inf"""
    out: f16 = np.maximum(0, inputs)
    return out


def forward(
    activation: Callable[[f16], f16], inputs: arr, weights: arr, bias: f16
) -> f16:
    """automates process of activation output of linear computation"""
    return activation(linear(inputs, weights, bias))
