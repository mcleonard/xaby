import jax


class Optimizer:
    def update_step(self, params, grads):
        raise NotImplemented

    def __call__(self, func, grads):
        new_params = jax.tree_multimap(self.update_step, func.params, grads)
        func._update(new_params)
        return func


class sgd(Optimizer):
    def __init__(self, lr=0.003):
        self.lr = lr

    def update_step(self, params, grads):
        return params - self.lr * grads
