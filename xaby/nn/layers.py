import xaby as xb

import jax
import jax.numpy as jnp
from jax import random

__all__ = ["linear", "conv2d"]


def _get_same_padding():
    pass


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
    key = xb.random.get_keys()
    sqrt_k = jnp.sqrt(k)
    return random.uniform(key, shape=shape, minval=-sqrt_k, maxval=sqrt_k)


class linear(xb.Fn):
    def __init__(self, in_features: int, out_features: int, bias=True):
        def linear(x: xb.ArrayList, params=None) -> xb.ArrayList:
            (inputs,) = x
            w, b = params["weights"], params["bias"]
            return xb.pack(jnp.matmul(inputs, w) + b)

        if not bias:

            def linear(x: xb.ArrayList, params=None) -> xb.ArrayList:
                (inputs,) = x
                w = params["weights"]
                return xb.pack(jnp.matmul(inputs, w))

        super().__init__(jax.jit(linear), 1, 1, name="linear")

        self.in_features, self.out_features = in_features, out_features
        self.params["weights"] = _init_weights(
            1 / in_features, shape=(in_features, out_features)
        )
        if bias:
            self.params["bias"] = _init_weights(1 / in_features, shape=(out_features,))

    def __repr__(self):
        return f"linear{self.in_features, self.out_features}"


class conv2d(xb.Fn):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        def conv2d(x: xb.ArrayList, params: dict = None) -> xb.ArrayList:
            (inputs,) = x
            w, b = params["weights"], params["bias"]
            conv = jax.lax.conv_general_dilated(
                inputs, w, self.stride, self.padding, feature_group_count=groups
            )
            return xb.pack(conv + b)

        if not bias:

            def conv2d(x: xb.ArrayList, params: dict = None) -> xb.ArrayList:
                (inputs,) = x
                w = params["weights"]
                conv = jax.lax.conv_general_dilated(
                    inputs, w, self.stride, self.padding, feature_group_count=groups
                )
                return xb.pack(conv)

        super().__init__(jax.jit(conv2d), 1, 1, name="conv2d")

        kernel = _to_tuple(kernel_size, (2,))

        if in_features % groups != 0:
            raise ValueError(f"in_features and groups must be evenly divisible")
        if out_features % groups != 0:
            raise ValueError(f"out_features and groups must be evenly divisible")

        kernel_shape = (out_features, in_features // groups, kernel[0], kernel[1])
        k = 1 / (in_features * jnp.prod(jnp.array(kernel)))

        self.params["weights"] = _init_weights(k, kernel_shape)
        if bias:
            self.params["bias"] = _init_weights(k, shape=(1, out_features, 1, 1))

        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        self.stride = _to_tuple(stride, (2,))
        self.padding = _to_tuple(padding, 2)
        self.bias = bias

    def __repr__(self):
        return (
            f"conv2d({self.in_features}, {self.out_features}, kernel={self.kernel}, "
            f"stride={self.stride}, padding={self.padding}, bias={self.bias})"
        )
