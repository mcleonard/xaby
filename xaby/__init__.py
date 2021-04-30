import jax.numpy as jnp


def array(*args, **kwargs):
    return jnp.array(*args, **kwargs)


array.__doc__ == jnp.array.__doc__

from .core import (
    Fn,
    fn,
    jit_combinators,
    sequential,
    parallel,
    split,
    describe,
    set_meta,
    eval,
    train,
    update,
    grad,
    value_and_grad,
    batchify,
)

from .arraylist import ArrayList, pack, collect

from .functions import *

from . import nn
from . import optim
from . import random
from . import utils
