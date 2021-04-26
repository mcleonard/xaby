from xaby import Fn, ArrayList, pack, jnp
import jax


class batchnorm2d(Fn):
    def __init__(self, num_features: int, epsilon: int = 1e-5):
        """ BatchNorm for 2D input, expects 4D input array in (B, C, H, W) format """

        @jax.jit
        def batchnorm2d(inputs: ArrayList, params: dict) -> ArrayList:
            (x,) = inputs
            weights, bias = params["weights"], params["bias"]
            num_features = x.shape[1]

            # Reshaping for broadcasting ease
            x_mean = jnp.mean(x, axis=(0, 2, 3)).reshape(1, num_features, 1, 1)
            x_var = jnp.mean((x - x_mean) ** 2, axis=(0, 2, 3)).reshape(
                1, num_features, 1, 1
            )

            x_norm = (x - x_mean) / jnp.sqrt(x_var + epsilon)
            y = weights * x_norm + bias
            return pack(y)

        super().__init__(batchnorm2d, 1, 1, name="batchnorm2d")

        self.params["weights"] = jnp.ones((1, num_features, 1, 1))
        self.params["bias"] = jnp.zeros((1, num_features, 1, 1))

        self.num_features = num_features
        self.epsilon = epsilon

    def __repr__(self):
        return f"batchnorm({self.num_features}, epsilon={self.epsilon})"
