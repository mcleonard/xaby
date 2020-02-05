from functools import wraps

from xaby import Tensor
from xaby.optim import Optimizer, BackProp

from .utils import arg_count, func_name

import jax


class BaseOp:
    def __init__(self):
        self.params = {}
        self._func = None

    def forward(self):
        raise NotImplementedError

    def __call__(self, x):
        return Tensor(self._func(x.data, self.params))


class Op(BaseOp):
    def _build_op(self):
        self._func = self.forward()

    def __rshift__(self, other):
        if issubclass(other.__class__, BaseOp):
            return Sequential(self, other)
        elif issubclass(other.__class__, Optimizer):
            return other(self)
        else:
            raise ValueError(f"This op doesn't support objects of type {type(other)}.")


class Sequential(Op):
    def __init__(self, *ops):
        self.ops = list(ops)
        self._build_op()

    def parameters(self):
        return [op.params for op in self.ops]

    def forward(self):
        @jax.jit
        def func(x, params):
            for op, param in zip(self.ops, params):
                x = op.forward()(x, param)
            return x

        return func

    def update(self, op):
        self.ops.append(op)
        self.params = self.parameters()

    def __rshift__(self, other):
        if issubclass(other.__class__, BaseOp):
            self.update(other)
            return self
        elif issubclass(other.__class__, Optimizer):
            return other(self)
        else:
            raise ValueError(f"This op doesn't support objects of type {type(other)}.")

    def __lshift__(self, loss):
        return BackProp(self, loss)

    def __iter__(self):
        return self.ops.__iter__()

    def __repr__(self):
        return str(self.ops)

# Decorator for creating simple ops from a function
def op(func):
    class FuncOp(Op):
        def __init__(self):
            super().__init__()
            self._build_op()
        
        def forward(self):
            n_args = arg_count(func)
            if n_args == 2:
                out_func = jax.jit(func)
            elif n_args == 1:
                @jax.jit
                def out_func(x, params):
                    return func(x)
            else:
                raise ValueError(f"Can't create an op with {n_args} arguments.")
            
            return out_func

        def __repr__(self):
            return func_name(func)


    func_op = FuncOp()
    func_op.__name__ = func_name(func)
    func_op.__doc__ = func.__doc__
    
    return func_op
        


