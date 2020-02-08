def traverse_wrappers(func):
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def arg_count(func):
    func = traverse_wrappers(func)
    return func.__code__.co_argcount


def func_name(func):
    func = traverse_wrappers(func)
    return func.__name__


def to_tuple(in_arg, shape):
    """ Return a number as a tuple repeated according to shape. """
    if isinstance(in_arg, tuple):
        return in_arg
    elif isinstance(in_arg, int):
        shape = to_tuple(shape, (shape,))
        out = in_arg
        for size in shape:
            out = (out,) * size
        return out
