from functools import wraps, partial
from itertools import chain
import inspect
from typing import Callable, List

from collections import OrderedDict

import jax

from .utils import sum_in_out

################
# ArrayList is a container for JAX DeviceArrays, it's the input data structure for all XABY functions


class ArrayList(list):
    def __new__(cls, x=None):
        return super(ArrayList, cls).__new__(cls, x)

    def __rshift__(self, other: Callable):
        return other(self)

    def __repr__(self):
        return "ArrayList:\n" + "\n".join(repr(e) for e in self)


def collect(lists: List[ArrayList]) -> ArrayList:
    """ Flattens a list of ArrayLists """
    return ArrayList(chain(*lists))


def pack(*x):
    return ArrayList(x)


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
    def __init__(self, func, n_inputs=None, n_outputs=None, name=None, docstring=None):
        self.forward = func
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.name = name if name is not None else func.__name__
        self.__doc__ = docstring if docstring is not None else None

        self.params = OrderedDict()
        self.grad_func = None

        if self.__doc__ is None:
            self.__doc__ = func.__doc__

    def describe(self):
        print(
            f"Fn: {self.name}\n"
            f"Expected inputs: {'variable' if self.n_inputs is None else self.n_inputs}\n"
            f"Expected outputs: {'variable' if self.n_outputs is None else self.n_outputs}\n\n"
            f"{self.__doc__}"
        )

    def _eval(self):
        pass

    def _train(self):
        pass

    def _called(self, caller):
        pass

    def _update(self, new_params: dict) -> dict:
        self.params = new_params

    def __rshift__(self, other: Callable):
        return sequential(self, other)

    def __lshift__(self, x: ArrayList) -> tuple:
        if self.grad_func is None:
            self.grad_func = jax.value_and_grad(self.forward, argnums=1)
        return self.grad_func(x, self.params)

    def __call__(self, x: ArrayList) -> ArrayList:
        return self.forward(x, self.params)

    def __repr__(self):
        return f"{self.name}"


def _build_func(func, **kwargs):
    if kwargs:
        p_func = partial(func, **kwargs)
        jitted_func = jax.jit(p_func)
    else:
        jitted_func = jax.jit(func)

    @wraps(func)
    def f(x: ArrayList, params=None):
        return pack(jitted_func(*x))

    return f


def _build_kwarg_func(func, kwargspec):
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

        return Fn(_build_func(func, **kwargs))

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
        return _build_kwarg_func(func, argspec[5])
    else:
        return Fn(_build_func(func))


def describe(func: Fn):
    return func.describe()


def train(func: Fn):
    func._train()
    return func


def eval(func: Fn):
    func._eval()
    return func


def set_meta(func: Fn, **kwargs):
    for k, v in kwargs.items():
        setattr(func, k, v)
    return func


def update(func: Fn, params: dict = None):
    if params is None:
        params = func.params
    func._update(params)
    return func


def grad(func: Fn):
    func.forward = jax.grad(func.forward, argnums=1)
    return func


def value_and_grad(func: Fn):
    func.forward = jax.value_and_grad(func.forward, argnums=1)
    return func


######################
# Here are objects that contain functions (Fn objects) in different configurations and route arrays appropriately
# For example, there's a container for arranging functions sequentially, another for in parallel


def _addindent(s_: str, num_spaces: int) -> str:
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def safe_insert(func: Fn, funcs: dict) -> str:
    """ Adds a func to a dictionary using the func's name, appends a numerical
        suffix if the name already exists, returns the key.
    """
    key = func.name
    i = 0
    while key in funcs:
        i += 1
        key = func.name + f"_{i}"
    funcs[key] = func

    return key


class Container(Fn):
    def _update(self, new_params: dict):
        self.params = new_params
        for i, (f, p) in enumerate(zip(self.funcs.values(), new_params.values())):
            f._update(p)

    def _eval(self):
        for f in self.funcs.values():
            eval(f)

    def _train(self):
        for f in self.funcs.values():
            train(f)

    def __getattr__(self, name: str):
        if name not in self.funcs:
            raise AttributeError(
                f"{name} is not an attribute of this function {self.name}"
            )
        return self.funcs[name]

    def __repr__(self):
        lines = []
        for key, f in self.funcs.items():
            f_str = f"({key}): " + repr(f)
            f_str = _addindent(f_str, 2)
            lines.append(f_str)
        return (
            f"{type(self).__name__}('{self.name}') {{\n  " + "\n  ".join(lines) + "\n}"
        )


class sequential(Container):
    def __init__(self, *funcs: Fn):
        @jax.jit
        def sequential(x: ArrayList, params: list) -> ArrayList:
            for i, (f, p) in enumerate(zip(self.funcs.values(), params.values())):
                x = f.forward(x, params=p)
            return x

        super().__init__(sequential)

        self.funcs = OrderedDict()
        self.params = OrderedDict()
        for f in funcs:
            self.append(f)

        self.name = "sequential"

    def __rshift__(self, other: Callable):
        # If the user gives a sequence a name, it indicates they want to use it as a standalone function.
        # So, if the name is changed from default, return a new sequential function.
        if self.name != "sequential":
            return self.__class__(self, other)

        self.append(other)
        return self

    def append(self, func: Callable):
        # TODO: Put a check in here to make sure expected inputs and outputs of the functions match

        key = safe_insert(func, self.funcs)

        if len(self.funcs) == 1:
            self.n_inputs = func.n_inputs
        self.n_outputs = func.n_outputs

        try:
            self.params[key] = func.params
        except:
            raise ValueError(
                f"Attempting to add {other} but it doesn't have the required params attribute."
            )

        return self


class split(Container):
    def __init__(self, *funcs: Fn):
        """ Takes N functions and an ArrayList with N arrays, then maps the input arrays to the functions.
            It then collects the output of each function into a single ArrayList.
            
            Example
            -------
            a = array(...)
            b = array(...)
            pack(a, b) >> split(sin, tan)  # returns [sin(a), tan(b)]
        """

        @jax.jit
        def split(arrays: ArrayList, params) -> ArrayList:
            if len(arrays) != len(self.funcs):
                raise ValueError(
                    "The number of input arrays must match the number of functions."
                )

            return collect(
                [
                    f.forward(pack(a), p)
                    for a, f, p in zip(arrays, self.funcs.values(), params.values())
                ]
            )

        super().__init__(
            split,
            n_inputs=len(funcs),
            n_outputs=sum_in_out(f.n_outputs for f in funcs),
        )

        self.name = "split"
        self.funcs = OrderedDict()

        for f in funcs:
            key = safe_insert(f, self.funcs)
            try:
                self.params[key] = f.params
            except:
                raise ValueError(
                    f"Attempting to add {other} but it doesn't have the required params attribute."
                )


class parallel(Container):
    """ Takes multiple functions and passes the input ArrayList to each function, then
        collects the output of each function into a single ArrayList 
        
        Example
        -------
        a = array(...)
        pack(a) >> parallel(sin, tan)  # returns [sin(a), tan(a)]
        
        Another example is a residual connection:
        
        dense = linear(10, 10) >> relu
        pack(a) >> parallel(skip, dense) >> add 
        
        pack(a) >> parallel(skip, dense) returns [a, dense(a)], then add performs an element-wise 
        addition on those arrays.
        
    """

    def __init__(self, *funcs: Fn):
        @jax.jit
        def parallel(x: ArrayList, params: list) -> ArrayList:
            return collect(
                [f.forward(x, p) for f, p in zip(self.funcs.values(), params.values())]
            )

        # TODO: Check expected inputs and outputs, calculate from the input functions
        super().__init__(parallel, name="parallel")

        self.funcs = OrderedDict()

        for f in funcs:
            key = safe_insert(f, self.funcs)
            try:
                self.params[key] = f.params
            except:
                raise ValueError(
                    f"Attempting to add {other} but it doesn't have the required params attribute."
                )
