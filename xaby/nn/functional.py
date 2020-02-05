from functools import wraps

import jax.numpy as np
import jax

from .core import op
from .utils import arg_count
from . import layers

class vmap:
    def __init__(self, in_axes=0, out_axes=0):
        self.in_axes = in_axes
        self.out_axes = out_axes
    
    def __call__(self, func):
        out_func = jax.vmap(func, in_axes=self.in_axes, out_axes=self.out_axes)
        return jax.jit(out_func)

@op
def relu(x):
    return np.clip(x, a_min=0)
    
@op
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@op
@vmap()
def log_softmax(x):
    return x - np.log(np.sum(np.exp(x)))

@op
@vmap()
def flatten(x):
    """ Flatten an array """
    return x.flatten()

@op
def shape(x):
    return x.shape

def linear(in_size, out_size):
    return layers.Linear(in_size, out_size)
