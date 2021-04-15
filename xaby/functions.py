from typing import Union, Tuple

import jax
import jax.numpy as jnp
from xaby import fn, Fn, ArrayList, pack, collect

__all__ = [
    "flatten",
    "shape",
    "add",
    "sub",
    "mul",
    "div",
    "mean",
    "select",
    "parallel",
    "skip"
]


class flatten(Fn):
    def __init__(self, axis=-1):

        self.axis = axis

        # A bit more complicated here because we need the ability to flatten along
        # specific axes. By default, this will flatten an entire array, but we can have
        # it map along a specific axis, so you only flatten along that one. This requires
        # some work with jax.vmap.

        @jax.jit
        def func(x: jnp.DeviceArray) -> jnp.DeviceArray:
            return jnp.ravel(x)

        if axis != -1:
            func = jax.vmap(func, axis, axis)

        @jax.jit
        def flatten(x: ArrayList, params=None) -> jnp.DeviceArray:
            return pack(func(x[0]))

        super().__init__(flatten, n_inputs=1, n_outputs=1)

    def __repr__(self):
        return f"flatten(axis={self.axis})"


class shape(Fn):
    """ Returns the shape of each array in an ArrayList """
    def __init__(self):
        @jax.jit
        def shape(x: ArrayList) -> tuple:
            return tuple(x.shape for array in x)

        super().__init__(shape)


class select(Fn):
    """ Selects specific arrays from the input ArrayList, returns those packed into another ArrayList
    
        Example
        -------
        a = xb.array(...)
        b = xb.array(...)
        
        arrays = pack(a, b)
        
        arrays >> select(0)           # returns [a]
        arrays >> select(1)           # returns [b]
        arrays >> select(0, 0, 1, 1)  # returns [a, a, b, b]
    
    """

    def __init__(self, *indices: int):
        @jax.jit
        def select(arrays: ArrayList) -> ArrayList:
            return ArrayList(arrays[i] for i in self.indices)

        super().__init__(select, n_inputs=None, n_outputs=len(indices))

        self.indices = indices


def skip(x: ArrayList) -> ArrayList:
    """ Returns the input ArrayList """
    return x
skip = Fn(skip)


class parallel(Fn):
    """ Takes multiple functions and passes a single input ArrayList to each function, then
        collects the output of each function into a single ArrayList 
        
        Example
        -------
        a = array(...)
        pack(a) >> parallel(sin, tan)  # returns [sin(a), tan(a)]
        
        Another example is a residual connection:
        
        dense = linear(10, 10) >> relu
        pack(a) >> parallel(skip, dense) >> add 
        
        pack(a) >> parallel(skip, dense) returns [a, dense(a)], then add performs an element-wise 
        addition on those arrays.
        
    """
    def __init__(self, *funcs: Fn):
        
        @jax.jit
        def parallel(x: ArrayList, params: dict) -> ArrayList:
            return collect([f.forward(x, p) for f, p in zip(self.funcs, params)])
        
        super().__init__(parallel)

        self.funcs = funcs
        self.params = [f.params for f in funcs]


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
