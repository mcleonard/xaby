class Optimizer:
    pass


class SGD(Optimizer):
    """ Stochastic gradient descent. """

    def __init__(self, model, lr=0.003):
        self.model = model
        self.lr = lr

    @staticmethod
    def update(model, grads, lr=0.003):
        for op, param_grad in zip(model, grads):
            params = op.params
            for key in params:
                params[key] -= lr * param_grad[key]

    def __call__(self, grads):
        self.update(self.model, grads, self.lr)

    def __repr__(self):
        return f"SGD(lr={self.lr})"
