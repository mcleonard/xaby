from xaby import ArrayList, pack, Fn

import jax
import jax.numpy as jnp


__all__ = ["nll_loss", "cross_entropy_loss"]


class nll_loss(Fn):
    """ Common loss function for use with log-probabilities returned from the log-softmax function """

    def __init__(self):
        @jax.jit
        def nll_loss(x: ArrayList, params):
            """ Assumes x are the log-softmax scores for a batch and 
                y is a vector indicating the correct labels as integers """
            log_p, targets = x
            row_idx = jnp.arange(len(log_p))
            return -jnp.mean(log_p[row_idx, targets])

        super().__init__(nll_loss, n_inputs=2, n_outputs=1)


class cross_entropy_loss(Fn):
    """ Common loss function for use with probabilities returned from the softmax function """

    def __init__(self):
        @jax.jit
        def cross_entropy_loss(x: ArrayList, params):
            p, targets = x
            log_p = jnp.log(p)
            row_idx = jnp.arange(len(log_p))
            return -jnp.mean(log_p[row_idx, targets])

        super().__init__(cross_entropy_loss, n_inputs=2, n_outputs=1)


class binary_cross_entropy_loss(Fn):
    """ Common loss function for use with probabilities returned from the sigmoid function """

    def __init__(self):
        @jax.jit
        def binary_cross_entropy_loss(x: ArrayList, params):
            p, y = x
            return -jnp.mean((y * jnp.log(p) + (1 - y) * jnp.log(1 - p)))

        super().__init__(binary_cross_entropy_loss, n_inputs=2, n_outputs=1)
