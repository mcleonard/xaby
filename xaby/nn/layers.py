from .core import Op
from xaby.random import key_manager

import jax.numpy as np
from jax import jit, random


class Linear(Op):
    def __init__(self, in_size, out_size, rand_key=None):
        if rand_key is None:
            key = key_manager.key
        else:
            key = rand_key
        self.params = {}
        self.params["weights"] = random.normal(key, shape=(in_size, out_size)) * 0.1
        self.params["bias"] = random.normal(key, shape=(out_size,)) * 0.1

        self._build_op()

    def forward(self):
        @jit
        def func(x, params):
            w, b = params["weights"], params["bias"]
            return np.matmul(x, w) + b

        return func

    def __repr__(self):
        return f"Linear: {self.params['weights'].shape}"
