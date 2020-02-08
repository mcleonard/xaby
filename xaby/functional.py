import jax
import jax.numpy as np

from .core import op

__all__ = ["vmap", "shape", "exp", "flatten", "name"]


class vmap:
    def __init__(self, in_axes=0, out_axes=0):
        self.in_axes = in_axes
        self.out_axes = out_axes

    def __call__(self, func):
        out_func = jax.vmap(func, in_axes=self.in_axes, out_axes=self.out_axes)
        return jax.jit(out_func)


@op
def shape(x):
    return x.shape


@op
def exp(x):
    return np.exp(x)


@op
@vmap()
def flatten(x):
    """ Flatten an array """
    return x.flatten()


## Other things! ###
def name(name):
    """ Add a name to a model/module """

    def name_func(model):
        model.name = name
        return model

    return name_func
