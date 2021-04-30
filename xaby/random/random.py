import jax.numpy as jnp
from jax import random

__all__ = ["get_keys", "set_seed", "bernoulli", "uniform", "normal", "randint"]

# Singleton class for managing random keys
class KeyManager:
    class __KeyManager:
        def __init__(self, key):
            self.key = key

        def seed(self, seed):
            self.key = random.PRNGKey(seed)

        def split(self, n=2):
            key, *subkeys = random.split(self.key, n)
            self.key = key
            return subkeys

        def __str__(self):
            return repr(self) + self.key

        def __repr__(self):
            return "KeyManager: " + str(self.key)

    instance = None

    def __init__(self, key):
        if not KeyManager.instance:
            KeyManager.instance = KeyManager.__KeyManager(key)
        else:
            KeyManager.instance.key = key

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __repr__(self):
        return self.instance.__repr__()


key_manager = KeyManager(random.PRNGKey(7))


def get_keys(n=1):
    keys = key_manager.split(n + 1)
    if n == 1:
        return keys[0]
    return keys


def set_seed(seed):
    key_manager.seed(seed)


########################
# Implement sampling from distributions here. Mostly it's using JAX's random module but
# handling splitting the keys automatically.


def bernoulli(shape=(), p=0.5, key=None) -> jnp.DeviceArray:
    if key is None:
        key = get_keys()
    return random.bernoulli(key, p=p, shape=shape)


def uniform(shape=(), key=None, dtype=jnp.float32) -> jnp.DeviceArray:
    if key is None:
        key = get_keys()
    return random.uniform(key, shape=shape, dtype=dtype)


def normal(shape=(), key=None, dtype=jnp.float32) -> jnp.DeviceArray:
    if key is None:
        key = get_keys()
    return random.normal(key, shape=shape, dtype=dtype)


def randint(shape=(), key=None, minval=0, maxval=2, dtype=jnp.int32) -> jnp.DeviceArray:
    if key is None:
        key = get_keys()
    return random.randint(key, shape, minval, maxval, dtype)
