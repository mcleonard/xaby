from collections import OrderedDict


class Optimizer:
    pass


def traverse(parameters: OrderedDict, grads: OrderedDict, skip=None):
    skip = set() if skip is None else skip
    for key in parameters:
        if key in skip:
            continue
        if isinstance(parameters[key], OrderedDict):
            yield from traverse(parameters[key], grads[key])
        else:
            yield parameters[key], grads[key]


class sgd(Optimizer):
    """ Stochastic gradient descent. """

    def __init__(self, model, lr=0.003, freeze=None):
        self.parameters = model.params
        self.lr = lr

        if freeze is None:
            freeze = set()
        elif not isinstance(freeze, list):
            freeze = set([freeze])
        else:
            freeze = set(freeze)

        self.freeze = freeze

    def update(self, parameters, grads):
        for params, grads in traverse(parameters, grads, skip=self.freeze):
            for key in params:
                params[key] -= self.lr * grads[key]

    def __call__(self, grads):
        self.update(self.parameters, grads)

    def __repr__(self):
        return f"SGD(lr={self.lr})"
