from jax import jit, grad

from xaby import Tensor

class BackProp:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss

        self._loss_func, self._grad_func = self._build_funcs()
    
    def _build_funcs(self):
        forward_func = self.model.forward()
        loss_func = self.loss.forward()

        def func(inputs, params, targets):
            prediction = forward_func(inputs, params)
            loss = loss_func(prediction, targets)
            return loss

        grad_func = jit(grad(func, argnums=1))
        return jit(func), grad_func

    def __lshift__(self, targets):
        loss = self._loss_func(self.inputs.data, self.model.parameters(), targets.data)
        grads = self._grad_func(self.inputs.data, self.model.parameters(), targets.data)
        return Tensor(loss), grads

    def __call__(self, inputs):
        self.inputs = inputs
        return self

    def __repr__(self):
        return "\n".join(
            [
                "BackProp",
                f"Loss: {self.loss.__repr__()}",
                f"Model: {self.net.__repr__()}",
            ]
        )
