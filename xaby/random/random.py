from jax import random

# Singleton class for maintaining random keys
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


def key(n=1):
    keys = key_manager.split(n + 1)
    if n == 1:
        return keys[0]
    return keys


def set_seed(seed):
    key_manager.seed(seed)
