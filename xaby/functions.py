import jax
import jax.numpy as jnp
from xaby import fn, Fn

__all__ = ["flatten", "shape"]

class flatten(Fn):
    def __init__(self, axis=-1):
        

        self.axis = axis
        
        # A bit more complicated here because we need the ability to flatten along
        # specific axes. By default, this will flatten an entire array, but we can have
        # it map along a specific axis, so you only flatten along that one. This requires
        # some work with jax.vmap and functional programming.

        @jax.jit
        def func(x: jnp.DeviceArray) -> jnp.DeviceArray:
            return jnp.ravel(x)

        if axis != -1:
            func = jax.vmap(func, axis, axis)
        
        @jax.jit
        def flatten(x: jnp.DeviceArray, params=None) -> jnp.DeviceArray:
            return func(x)

        super().__init__(flatten)

    def __repr__(self):
        return f"flatten(axis={self.axis})"

@fn
def shape(x: jnp.DeviceArray) -> tuple:
    return x.shape