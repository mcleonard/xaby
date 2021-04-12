from typing import Optional, Union, Tuple, Callable

import jax
import jax.numpy as jnp

from xaby import fn

__all__ = [
    "relu",
    "sigmoid",
    "softmax",
    "log_softmax",
    "add",
    "sub",
    "mul",
    "div",
    "mean",
    "concatenate",
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


@fn
def add(x: jnp.DeviceArray, y: jnp.DeviceArray) -> jnp.DeviceArray:
    """ Element-wise addition, input arrays must have the same shape """
    return x + y


@fn
def sub(x: jnp.DeviceArray, y: jnp.DeviceArray) -> jnp.DeviceArray:
    """ Element-wise subtraction, input arrays must have the same shape """
    return x - y


@fn
def mul(x: jnp.DeviceArray, y: jnp.DeviceArray) -> jnp.DeviceArray:
    """ Element-wise multiplication, input arrays must have the same shape """
    return x * y


@fn
def div(x: jnp.DeviceArray, y: jnp.DeviceArray) -> jnp.DeviceArray:
    """ Element-wise divison (x/y), input arrays must have the same shape """
    return x / y


@fn
def mean(
    x: jnp.DeviceArray, *, axis: Union[None, int, Tuple[int]] = None
) -> jnp.DeviceArray:
    """ Computes the mean along the specified axis 
        
        Arguments
        ---------
        x : array_like
            Array containing numbers whose mean is desired. If `x` is not an
            array, a conversion is attempted.
        axis : None or int or tuple of ints, optional
            Axis or axes along which the means are computed. The default is to
            compute the mean of the flattened array.
    
    """
    return jnp.mean(x, axis=axis)


@fn
def concatenate(*x: jnp.DeviceArray, axis: int = 0) -> jnp.DeviceArray:
    """ Join multiple arrays along an existing axis """
    return jnp.concatenate(list(x), axis=axis)
