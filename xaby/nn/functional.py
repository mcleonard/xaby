import jax.numpy as np
import jax

from xaby import op, vmap
from xaby.utils import arg_count
from . import layers
from . import losses

__all__ = ["relu", "sigmoid", "softmax", "log_softmax"]


@op
def relu(x):
    return np.clip(x, a_min=0)


@op
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@op
@vmap()
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


@op
@vmap()
def log_softmax(x):
    return x - np.log(np.sum(np.exp(x)))


### Layers ###

__all__.extend(["linear", "conv2d"])
linear = layers.Linear
conv2d = layers.Conv2d


### Losses ###
__all__.extend(["mse", "nlloss"])
mse = losses.MSE()
nlloss = losses.NLLoss()
