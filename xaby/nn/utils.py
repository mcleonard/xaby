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
