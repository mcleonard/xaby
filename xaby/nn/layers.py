import xaby

import jax.numpy as jnp
from jax import jit, random
from jax.lax import conv_with_general_padding

__all__ = ["linear", "conv2d"]


def _to_tuple(in_arg, shape):
    """ Return a number as a tuple repeated according to shape. """
    if isinstance(in_arg, tuple):
        return in_arg
    elif isinstance(in_arg, int):
        shape = _to_tuple(shape, (shape,))
        out = in_arg
        for size in shape:
            out = (out,) * size
        return out


def _init_weights(k, shape):
    key = xaby.random.key()
    sqrt_k = jnp.sqrt(k)
    return random.uniform(key, shape=shape, minval=-sqrt_k, maxval=sqrt_k)


class linear(xaby.Fn):
    def __init__(self, in_features: int, out_features: int):

        self.in_features, self.out_features = in_features, out_features
        
        self.params={}
        self.params["weights"] = _init_weights(
            1 / in_features, shape=(in_features, out_features)
        )
        self.params["bias"] = _init_weights(1 / in_features, shape=(out_features,))

        @jit
        def forward(x, params):
            w, b = params["weights"], params["bias"]
            return jnp.matmul(x, w) + b

        self.forward = forward

    def __repr__(self):
        return f"linear{self.in_features, self.out_features}"


class conv2d(xaby.Fn):
    def __init__(self, in_features, out_features, kernel_size=3, strides=1, padding=0):

        kernel = _to_tuple(kernel_size, (2,))
        kernel_shape = (out_features, in_features, kernel[0], kernel[1])
        k = 1 / (in_features * np.prod(np.array(kernel)))
        
        self.params = {}
        self.params["weights"] = _init_weights(k, kernel_shape)
        self.params["bias"] = _init_weights(k, shape=(1, out_features, 1, 1))

        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        self.strides = _to_tuple(strides, (2,))
        self.padding = _to_tuple(padding, 2)

        @jit
        def forward(x, params):
            w, b = params["weights"], params["bias"]
            conv = conv_with_general_padding(
                x, params["weights"], self.strides, self.padding, None, None
            )
            return conv + b

        self.forward = forward

    def __repr__(self):
        return (
            f"Conv2d({self.in_features}, {self.out_features}, strides={self.strides})"
        )
