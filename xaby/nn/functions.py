from typing import Optional, Union, Tuple, Callable

import jax
import jax.numpy as jnp

from xaby import fn

__all__ = [
    "relu",
    "sigmoid",
    "softmax",
    "log_softmax",
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
