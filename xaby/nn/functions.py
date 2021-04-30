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
    "nll_loss",
    "cross_entropy_loss",
    "binary_cross_entropy_loss",
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

        @jax.jit
        def train_dropout(x: ArrayList, params=None) -> ArrayList:
            array = x[0]
            mask = xr.bernoulli(array.shape, p=(1 - self.drop_p))
            output = pack(
                array * mask / max(1 - self.drop_p, 0.00001)
            )  # The max prevents dividing by zero
            return output

        @jax.jit
        def eval_dropout(x: ArrayList, params=None) -> ArrayList:
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


@jax.jit
def _log_loss(
    log_p: jnp.DeviceArray, targets: jnp.DeviceArray, smoothing: float
) -> jnp.DeviceArray:

    row_idx = jnp.arange(len(log_p))
    true_targets = log_p[row_idx, targets]

    if smoothing is None:
        return -jnp.mean(true_targets)
    else:

        return -jnp.mean(
            smoothing / log_p.shape[1] * log_p.sum(axis=1)
            + (1 - smoothing) * true_targets
        )


class nll_loss(Fn):
    """ Common loss function for use with log-probabilities returned from the log-softmax function """

    def __init__(self, smoothing: Optional[float] = None):
        if smoothing is None:
            smoothing = 0.0

        if smoothing < 0.0 or smoothing >= 1.0:
            raise ValueError("smoothing must be between 0 and 1")

        @jax.jit
        def nll_loss(x: ArrayList, params):
            """ Assumes x are the log-softmax scores for a batch and 
                y is a vector indicating the correct labels as integers """
            log_p, targets = x
            return pack(_log_loss(log_p, targets, smoothing))

        super().__init__(nll_loss, n_inputs=2, n_outputs=1, name="nll_loss")


class cross_entropy_loss(Fn):
    """ Common loss function for use with probabilities returned from the softmax or sigmoid functions """

    def __init__(self, smoothing: Optional[float] = None):
        if smoothing is None:
            smoothing = 0.0

        if smoothing < 0.0 or smoothing >= 1.0:
            raise ValueError("smoothing must be between 0 and 1")

        @jax.jit
        def cross_entropy_loss(x: ArrayList, params):
            p, targets = x
            log_p = jnp.log(p)
            return pack(_log_loss(log_p, targets, smoothing))

        super().__init__(cross_entropy_loss, n_inputs=2, n_outputs=1)


class binary_cross_entropy_loss(Fn):
    """ Common loss function for use with probabilities returned from the sigmoid function """

    def __init__(self):
        @jax.jit
        def binary_cross_entropy_loss(x: ArrayList, params):
            p, y = x
            return pack(-jnp.mean((y * jnp.log(p) + (1 - y) * jnp.log(1 - p))))

        super().__init__(binary_cross_entropy_loss, n_inputs=2, n_outputs=1)
