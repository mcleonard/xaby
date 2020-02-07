from .core import Op
import xaby
from .utils import to_tuple

import jax.numpy as np
from jax import jit, random, vmap
from jax.lax import conv_with_general_padding

def init_weights(k, shape):
    key = xaby.random.key()
    sqrt_k = np.sqrt(k)
    return random.uniform(key, shape=shape, minval=-sqrt_k, maxval=sqrt_k)

class Linear(Op):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.params = {}
        self.params["weights"] = init_weights(1/in_size, shape=(out_size, in_size))
        self.params["bias"] = init_weights(1/in_size, shape=(out_size,))

        self._build_op()

    def forward(self):
        @jit
        def func(x, params):
            w, b = params["weights"], params["bias"]
            return np.matmul(x, w.T) + b

        return func

    def __repr__(self):
        return f"Linear{self.params['weights'].shape}"


class Conv2d(Op):
    def __init__(
        self, in_features, out_features, kernel_size=3, strides=1, padding=0
    ):
        super().__init__() 

        kernel = to_tuple(kernel_size, (2,))
        kernel_shape = (out_features, in_features, kernel[0], kernel[1])
        k = 1 / (in_features * np.prod(np.array(kernel)))
        self.params["weights"] = init_weights(k, kernel_shape)
        self.params["bias"] = init_weights(k, shape=(1, out_features, 1, 1))
        
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        self.strides = to_tuple(strides, (2,))
        self.padding = to_tuple(padding, 2)

        self._build_op()

    def forward(self):
        def func(x, params):
            conv = conv_with_general_padding(
                x, params["weights"], self.strides, self.padding, None, None
            )
            return conv + params["bias"]

        return jit(func)

    def __repr__(self):
        return f"Conv2d({self.in_features}, {self.out_features}, strides={self.strides})"
