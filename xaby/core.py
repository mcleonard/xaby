from functools import wraps, partial
import inspect
from typing import Callable

import jax

########
# ArrayList is a


class ArrayList(list):
    def __new__(cls, x):
        return super(ArrayList, cls).__new__(cls, list(x))

    def append(self, element):
        """ Appends to this list, returns a new copy. """
        new = self.copy()
        new.append(element)
        return new

    def extend(self, element):
        new = self.copy()
        new.extend(element)
        return new

    def __rshift__(self, other: Callable):
        return other(self)

    def __repr__(self):
        return "ArrayList:\n" + "\n".join(repr(e) for e in self)


def pack(*x):
    return ArrayList(x)


def p(*x):
    return pack(*x)


########
# This next part are functions so we can use ArrayList objects in JAX jitted functions


def flatten_list(array_list):
    elements = [e for e in array_list]
    aux = None  # we don't need auxiliary information for this simple class
    return (elements, aux)


# careful, switched argument order! (unfortunate baggage from the past...)
def unflatten_list(_aux, elements):
    return ArrayList(elements)


jax.tree_util.register_pytree_node(
    ArrayList, flatten_list, unflatten_list,
)

#########


class Fn:
    def __init__(self, func):
        self.name = func.__name__
        self.__doc__ = "Expected input..." + str(func.__doc__)
        self.forward = func
        self.params = {}

    def _update(self, new_params: dict):
        self.params = new_params

    def __rshift__(self, other: Callable):
        return Sequential(self, other)

    def __call__(self, x: ArrayList) -> ArrayList:
        return pack(self.forward(*x, params=self.params))

    def __repr__(self):
        return f"{self.name}"


def build_func(func, **kwargs):
    if kwargs:
        p_func = partial(func, **kwargs)
        jitted_func = jax.jit(p_func)
    else:
        jitted_func = jax.jit(func)

    @wraps(func)
    def f(*x, params=None):
        return jitted_func(*x)

    return f


def build_kwarg_func(func, kwargspec):
    """ Builds a Fn based on keyword arguments. """

    def func_with_kwargs(**kwargs):
        valid_kws = kwargspec
        if len(kwargs) == 0:
            raise ValueError(
                f"This function requires a keyword argument from: {valid_kws.keys()}"
            )

        for key in kwargs:
            if key not in valid_kws:
                raise ValueError(
                    f"Argument {key} not a valid keyword argument. Use only {valid_kws.keys()}"
                )

        return Fn(build_func(func, **kwargs))

    func_with_kwargs.__doc__ = " \n".join(
        [
            f"Returns the {func.__code__.co_name} function with the provided keyword argument",
            "\nRequired keyword arguments:",
            "\n ".join(
                [f"- {key}: default = {default}" for key, default in kwargspec.items()]
            ),
            f"\nSee the {func.__code__.co_name} documentation for more information: \n",
            f"{func.__doc__}",
        ]
    )
    return func_with_kwargs


def fn(func):
    """ Decorator for converting functions into Fn objects for use in XABY models """

    argspec = inspect.getfullargspec(func)
    # argspec[5] contains a dictionary of kw only arguments and their defaults
    if argspec[5]:
        return build_kwarg_func(func, argspec[5])
    else:
        return Fn(build_func(func))

######################

class Sequential:
    def __init__(self, *funcs):
        self.funcs = list(funcs)
        self.forward = self._build_forward()
        self.params = [f.params for f in self.funcs]

    def _build_forward(self) -> Callable:
        @jax.jit
        def forward(x: ArrayList, params: list) -> ArrayList:
            for f, p in zip(self.funcs, params):
                x = pack(f.forward(*x, params=p))
            return x

        return forward

    def _update(self, new_params: list, propagate=False):

        # When we update parameters, we don't actually need to update the parameters
        # on each layer/operation since the parameters are passed from this object. Instead,
        # we just update the parameters on this object. However, it might be useful to propagate
        # new parameters to the operations, so I'll make it optional.
        if propagate:
            # Propagate new parameters to each of the individual functions
            for f, p in zip(self.funcs, new_params):
                f._update(p)

        self.params = new_params

    def __rshift__(self, other: Callable):
        self.append(other)
        return self

    def append(self, other: Callable):
        self.funcs.append(other)
        self.forward = self._build_forward()

        try:
            self.params.append(other.params)
        except:
            raise ValueError(
                f"Attempting to add {other} but it doesn't have the required params attribute."
            )

        return self

    def __call__(self, x: ArrayList) -> ArrayList:
        return self.forward(x, self.params)
