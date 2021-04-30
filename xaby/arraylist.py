from typing import Callable, Any, List

from itertools import chain

import jax

################
# ArrayList is a container for JAX DeviceArrays, it's the input data structure for all XABY functions


class ArrayList(list):
    def __new__(cls, x=None):
        return super(ArrayList, cls).__new__(cls, x)

    def __rshift__(self, other: Callable) -> Any:
        return other(self)

    def __repr__(self):
        return "ArrayList:\n" + "\n".join(repr(e) for e in self)


def collect(lists: List[ArrayList]) -> ArrayList:
    """ Flattens a list of ArrayLists """
    return ArrayList(chain(*lists))


def pack(*x: jax.numpy.DeviceArray) -> ArrayList:
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
