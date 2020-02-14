import numpy as np
from typing import Tuple, List
from numba import jitclass, boolean, float64, njit
import warnings
import functional as F


class LinearReg(object):
    def __init__(self, grad: bool = False):
        self.grad = grad

    def fit(self, X: arr, y: arr, iterations: int = 1000000) -> None:
        if self.grad:
            self.weights = np.random.randn(X.shape[1])
            self.bias = np.random.rand()
            self.weights, self.bias = fit_grad(
                X, y, weights=self.weights, bias=self.bias, iterations=iterations
            )
        else:
            if iterations != 1000000:
                warnings.warn("Iterations argument not used when fitting ols")
            self.weights, self.bias = fit_ols(X, y)

    def predict(self, X: arr) -> arr:
        if self.grad:
            return np.dot(X, self.weights) + self.bias
        else:
            return np.dot(X, self.weights)

    def fit_predict(self, X, y, grad: boolean = False):
        self.fit(X, y)
        return self.predict(X)

    def score(self, X: arr, y: arr) -> np.float:
        y_pred = self.predict(X)
        u = (((y - y_pred) ** 2)).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / v)


l = LinearReg(grad=False)
X = np.random.randn(100, 4)

y = np.dot(X, np.array([3, 1, 2, 3]))

if __name__ == "__main__":
    l.fit(X, y)
    print(l.score(X, y))
