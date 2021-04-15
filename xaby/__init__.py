import jax.numpy as jnp


def array(*args, **kwargs):
    return jnp.array(*args, **kwargs)


array.__doc__ == jnp.array.__doc__

from .core import ArrayList, pack, Fn, fn, collect
from .functions import *

from . import nn
from . import optim
from . import random
