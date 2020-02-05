from xaby import Tensor

import jax.numpy as np
from jax import jit

__all__ = ["Loss", "mse", "nlloss"]


class Loss:
    def __init__(self):
        self._func = self.forward()

    def forward(self):
        raise NotImplementedError

    def __lshift__(self, targets):
        return Tensor(self._func(self.predictions.data, targets.data))

    def __call__(self, predictions):
        self.predictions = predictions
        return self


class MSE(Loss):
    """ Mean Squared Error loss """

    def forward(self):
        @jit
        def func(predictions, targets):
            return np.sum((targets - predictions) ** 2) / predictions.shape[0]

        return func

    def __repr__(self):
        return "MeanSquaredError"


mse = MSE()


class NLLoss(Loss):
    """ Negative Log-Likelihood loss """

    def forward(self):
        @jit
        def func(log_p, targets):
            rows = np.arange(len(log_p))
            return -np.mean(log_p[rows, targets])

        return func


nlloss = NLLoss()
