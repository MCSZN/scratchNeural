import numpy as np
from numba import njit

arr = np.ndarray

@njit
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

@njit
def relu(Z):
    A = np.maximum(0, Z)
    assert A.shape == Z.shape
    cache = Z
    return A, cache

@njit
def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    assert A.shape == Z.shape
    return A, cache

@njit
def relu_back(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z.T <= 0] = 0
    assert dZ.shape == Z.T.shape
    return dZ

@njit
def sigmoid_back(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

@njit
def tanh_back(dA, cache):
    Z = cache
    t = np.maximum(Z, 0)
    dZ = dA * t
    assert dZ.shape == Z.shape
    return dZ

@njit
def linear(X, W, b):
    Z = np.dot(X, W) + b
    cache = (X, W, b)
    return Z, cache

@njit
def linear_back(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W, dZ.T)

    return dA_prev, dW, db


@njit
def fit_grad(
    X: arr,
    y: arr,
    weights: arr,
    bias: arr,
    iterations: int = 1000000,
    alpha: np.float = 0.01,
) -> Tuple[arr, np.float]:
    # optimization loop
    for iteration in range(iterations):
        # forward pass
        y_hat = np.dot(X, weights) + bias

        # compute the MSE loss
        cost = ((y - y_hat) ** 2).mean()

        # optimization step

        # compute gradients
        b_hat_d = -2 * (y - y_hat).mean()
        W_hat_d = -2 * np.dot((y - y_hat), X) / y.shape[0]

        # update weights with gradients
        weights -= W_hat_d * alpha
        bias -= b_hat_d * alpha

    return weights, bias


@njit
def fit_ols(X: arr, y: arr) -> Tuple[arr, np.float]:
    weights = np.linalg.lstsq(X, y, rcond=-1)[0]
    bias = y.mean()
    return weights, bias