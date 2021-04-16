from typing import Iterable, Union


def sum_in_out(values: Iterable) -> Union[int, None]:
    total = 0
    for val in values:
        if val is None:
            return None
        else:
            total += val

    return total
