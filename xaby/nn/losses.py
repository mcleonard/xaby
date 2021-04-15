from xaby import ArrayList, pack

import jax
import jax.numpy as jnp


__all__ = ["Loss", "nll_loss"]


class Loss:
    pass


class nll_loss(Loss):
    def __init__(self, func):
        self.func = func

        @jax.jit
        def nll_loss(log_p: jnp.DeviceArray, targets: jnp.DeviceArray):
            """ Assumes x are the log-softmax scores for a batch and 
                y is a vector indicating the correct labels as integers """

            row_idx = jnp.arange(len(log_p))
            return -jnp.mean(log_p[row_idx, targets])

        def loss_func(inputs, targets, params):
            (log_p,) = self.func.forward(pack(inputs), params=params)

            if jnp.any(targets > log_p.shape[1] - 1):
                raise ValueError(
                    f"Target labels are out of bounds given log-probabilities with shape {log_p.shape}"
                )

            return nll_loss(log_p, targets)

        self.loss_func = jax.value_and_grad(loss_func, argnums=2)

    def __call__(self, x: ArrayList) -> tuple:
        if len(x) != 2:
            raise ValueError("This function requires two inputs")

        return self.loss_func(x[0], x[1], self.func.params)
