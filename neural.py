import functional as F
import numpy as np
import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Iterable, List, Dict, Tuple, Callable


class Model(object):
    def __init__(
        self,
        layers_dims: List[int] = [256, 64, 1],
        activations: List[str] = ["relu", "relu", "sigmoid"],
        lr: float = 0.01,
        epochs: int = 5000,
        residual: bool = False,
        input_shape: Tuple[int] = (1000, 256),
    ):
        super(Model, self).__init__()
        self.layers_dims: Iterable[int] = layers_dims
        assert set(activations).issubset(
            {"relu", "tanh", "sigmoid"}
        ), "activation function not supported"
        assert len(layers_dims) == len(activations)
        self.activations: Iterable[str] = activations
        self.layers_dims: Iterable[int] = layers_dims
        self.input_shape: Tuple[int] = input_shape
        self.lr: float = lr
        self.epochs: int = epochs
        self.residual: bool = residual
        self.parameters: Dict[str, Iterable[float]] = self._init_params()

    def _init_params(self) -> Dict[str, np.ndarray]:
        """create parameter dict
        >>>isinstance(self.init_params([2,2]), dict) == True
        Returns:
            dict -- params['W1'] == arrays of rand with shape (m,n)
        """
        params = {}
        # create threshold value of num layer in network for loop
        L = len(self.layers_dims)
        # loop over each layer and init rand vals for W and zeros for b
        params["W0"] = np.random.randn(self.input_shape[1], self.layers_dims[0])
        params["b0"] = np.random.randn(self.input_shape[0], 1)
        for l in range(1, L):
            params["W" + str(l)] = (
                np.random.randn(self.layers_dims[l - 1], self.layers_dims[l]) * 0.01
            )
            params["b" + str(l)] = np.zeros((self.input_shape[0], 1))

        return params

    def update_params(self, grads: Dict) -> None:
        """Update the parameters of the model with gradient already computed
        
        Arguments:
            grads {dict} -- corresponds to the gradient for each parameter in the model
        
        Returns:
            None
        """
        L = len(self.layers_dims)
        # update parameters of weight and bias with learning rate
        for l in range(L):
            self.parameters["W" + str(l)] -= self.lr * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= self.lr * grads["db" + str(l)]

    def _layer_forward(
        self, X, activation: str, layer: int
    ) -> Tuple[float, Tuple[np.ndarray]]:
        # use the forward function then the activation func
        Z, linear_cache = F.linear(
            X, self.parameters["W" + str(layer)], self.parameters["b" + str(layer)]
        )
        if activation == "sigmoid":
            X, activation_cache = F.sigmoid(Z)
        elif activation == "relu":
            X, activation_cache = F.relu(Z)
        elif activation == "tanh":
            X, activation_cache = F.tanh(Z)
        cache = (linear_cache, activation_cache)
        # return new value of cell and original values
        return X, cache

    def forward(self, X) -> Tuple[float, Tuple[np.ndarray]]:
        caches = []
        for layer, activ in zip(range(len(self.layers_dims)), self.activations):
            X, cache = self._layer_forward(X, activ, layer)
            caches.append(cache)
        return X, caches

    def _layer_backward(
        self, X, cache: Tuple[np.ndarray], activation: str
    ) -> Tuple[np.ndarray]:
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = F.relu_back(X, activation_cache)
        elif activation == "sigmoid":
            dZ = F.sigmoid_back(X, activation_cache)
        elif activation == "tanh":
            dZ = F.tanh_back(X, activation_cache)
        dX_prev, dW, db = F.linear_back(dZ, linear_cache)
        return dX_prev, dW, db

    def backward(
        self, out: np.ndarray, y: np.ndarray, caches: Tuple[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        grads = {}
        L = len(caches)
        # compute derivative of last layer
        dOut = -(np.divide(y, out) - np.divide(1 - y, 1 - out))
        # set current cache
        current_cache = caches[L - 1]
        # get gradients of last activation layer
        (
            grads["dA" + str(L)],
            grads["dW" + str(L)],
            grads["db" + str(L)],
        ) = self._layer_backward(dOut, current_cache, self.activations[-1])

        # get grads for all other layers
        for l, activ in zip(range(L - 1)[::-1], self.activations):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self._layer_backward(
                grads["dA" + str(l + 2)], current_cache, activ
            )
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def loss(self, out: np.ndarray, y: np.ndarray):
        return np.square(out - y).mean()

    def __call__(self, X):
        return self.forward(X)


def fit(model: Model, X: np.ndarray, y: np.ndarray) -> List[float]:
    cost = []
    for _ in range(model.epochs):
        out, cache = model.forward(X)
        cost.append(model.loss(out, y))
        grads = model.backward(out, y, cache)
        model.update_params(grads)
    return cost


def plot_loss(loss: np.ndarray) -> None:
    plt.plot(np.squeeze(loss))
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


if __name__ == "__main__":
    from sklearn.datasets import load_boston

    X, y = load_boston(return_X_y=True)
    y = (y - y.mean()) / y.std()

    classifier = Model(
        layers_dims=[16, 8, 1],
        activations=["relu", "relu", "tanh"],
        lr=0.01,
        epochs=500,
        input_shape=X.shape,
    )

    losses = fit(classifier, X, y)
    print(losses)
