from typing import Optional, Union, Tuple, Callable

import jax
import jax.numpy as jnp

from xaby import fn, Fn, ArrayList, pack
from xaby import random as xr

__all__ = [
    "relu",
    "sigmoid",
    "softmax",
    "log_softmax",
    "dropout",
]


@fn
def relu(x: jnp.DeviceArray) -> jnp.DeviceArray:
    """ Returns the Rectified Linear Unit (ReLU) activation """
    return jnp.clip(x, a_min=0)


@fn
def sigmoid(x: jnp.DeviceArray) -> jnp.DeviceArray:
    return 1 / (1 + jnp.exp(-x))


@fn
def softmax(x: jnp.DeviceArray, *, axis: int = 0) -> jnp.DeviceArray:
    """ Calculates softmax across a desired axis.
    
        Arguments
        ---------
        x: DeviceArray, Input array
        axis: int, optional,
            Array axis to calculate the sum across
    """
    return jnp.exp(x) / jnp.expand_dims(jnp.sum(jnp.exp(x), axis=axis), axis)


@fn
def log_softmax(x: jnp.DeviceArray, *, axis: int = 0) -> jnp.DeviceArray:
    """ Calculates log-softmax across a desired axis.
    
        Arguments
        ---------
        x: DeviceArray, Input array
        axis: int, optional
            Array axis to calculate the sum across
    """
    return x - jnp.expand_dims(jnp.log(jnp.sum(jnp.exp(x), axis=axis)), axis)


class dropout(Fn):
    """ Dropout regularization with drop probability set by """

    def __init__(self, drop_p=0.5, in_shape=None):

        self.drop_p = drop_p
        self.in_shape = in_shape

        @jax.jit
        def train_dropout(x: ArrayList, params=None) -> ArrayList:
            self._set_shape(x[0].shape)
            # It's possible the input array can be smaller than previous ones, such as the last batch
            # in an epoch. So I'll add a bit in there so the mask will be the same shape as the array
            output = pack(
                x[0] * params["mask"][: x[0].shape[0]] / max(1 - self.drop_p, 0.00001)
            )  # The max prevents dividing by zero
            return output

        @jax.jit
        def eval_dropout(x: ArrayList, params=None) -> ArrayList:
            self._set_shape(x[0].shape)
            return x

        self._train_func = train_dropout
        self._eval_func = eval_dropout

        super().__init__(train_dropout, n_inputs=1, n_outputs=1)

        self.name = "dropout"
        self.params = {
            "mask": xb.random.bernoulli(self.in_shape, 1 - self.drop_p).astype(
                jnp.float32
            )
        }

    def _set_shape(self, shape: Tuple[int, int]):
        if shape[0] > self.shape[0]:
            self.shape = shape

    def _update(self, new_params: dict) -> dict:
        return {
            "mask": xb.random.bernoulli(self.in_shape, 1 - self.drop_p).astype(
                jnp.float32
            )
        }

    def _train(self):
        self.forward = self._train_func

    def _eval(self):
        self.forward = self._eval_func
