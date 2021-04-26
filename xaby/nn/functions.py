from functools import reduce
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
def relu6(x: jnp.DeviceArray) -> jnp.DeviceArray:
    return jnp.clip(x, a_min=0, a_max=6)


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
            array = x[0]
            mask = xr.bernoulli(array.shape)
            # It's possible the input array can be smaller than previous ones, such as the last batch
            # in an epoch. So I'll add a bit in there so the mask will be the same shape as the array
            output = pack(
                array * mask / max(1 - self.drop_p, 0.00001)
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

    def _train(self):
        self.forward = self._train_func

    def _eval(self):
        self.forward = self._eval_func


class max_pool2d(Fn):
    def __init__(self, window_shape: tuple, strides=None, padding="SAME"):

        # Assumes input arrays are in format [N, C, H, W]
        non_spatial_axes = 0, 1
        strides = strides or (1,) * len(window_shape)

        for i in sorted(non_spatial_axes):
            window_shape = window_shape[:i] + (1,) + window_shape[i:]
            strides = strides[:i] + (1,) + strides[i:]

        @jax.jit
        def max_pool2d(inputs: xb.ArrayList, params):
            out = jax.lax.reduce_window(
                inputs[0], -jnp.inf, jax.lax.max, window_shape, strides, padding
            )
            return xb.pack(out)

        super().__init__(max_pool2d, 1, 1)


class sum_pool2d(Fn):
    def __init__(self, window_shape: tuple, strides=None, padding="SAME"):

        # Assumes input arrays are in format [N, C, H, W]
        non_spatial_axes = 0, 1
        strides = strides or (1,) * len(window_shape)

        for i in sorted(non_spatial_axes):
            window_shape = window_shape[:i] + (1,) + window_shape[i:]
            strides = strides[:i] + (1,) + strides[i:]

        @jax.jit
        def sum_pool2d(inputs: ArrayList, params):
            sums = jax.lax.reduce_window(
                inputs[0], 0.0, jax.lax.add, window_shape, strides, padding
            )
            return pack(sums)

        super().__init__(sum_pool2d, 1, 1)


class avg_pool2d(Fn):
    def __init__(self, window_shape: tuple, strides=None, padding="SAME"):

        # Assumes input arrays are in format [N, C, H, W]
        non_spatial_axes = 0, 1
        strides = strides or (1,) * len(window_shape)
        window_size = reduce(lambda x, y: x * y, window_shape, 1)

        for i in sorted(non_spatial_axes):
            window_shape = window_shape[:i] + (1,) + window_shape[i:]
            strides = strides[:i] + (1,) + strides[i:]

        @jax.jit
        def avg_pool2d(inputs: ArrayList, params):
            sums = jax.lax.reduce_window(
                inputs[0], 0.0, jax.lax.add, window_shape, strides, padding
            )
            return pack(sums / window_size)

        super().__init__(avg_pool2d, 1, 1)
