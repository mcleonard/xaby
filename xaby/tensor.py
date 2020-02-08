import jax.numpy as np


class tensor:
    def __init__(self, object, dtype=None, copy=True, order="K", ndmin=0):

        # Duck typing for PyTorch tensors and similar objects
        try:
            object = object.numpy()
        except AttributeError:
            pass

        self.data = np.array(object, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

    def item(self):
        return self.data.item()

    @property
    def device(self):
        return self.data.device_buffer.device()

    def numpy(self):
        return self.data

    def __add__(self, other):
        return tensor(self.data + other)

    def __radd__(self, other):
        return tensor(self.data + other)

    def __sub__(self, other):
        return tensor(self.data - other)

    def __rsub__(self, other):
        return tensor(self.data - other)

    def __mul__(self, other):
        return tensor(self.data * other)

    def __rmul__(self, other):
        return tensor(self.data * other)

    def __truediv__(self, other):
        return tensor(self.data / other)

    def __rtruediv__(self, other):
        return tensor(other / self.data)

    def __rshift__(self, other):
        return other(self)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return self.data.__str__()

    def __format__(self, value):
        return self.data.__format__(value)

    def __repr__(self):
        return f"tensor({str(self.data)}, dtype={self.data.dtype.name})"

    def __getattr__(self, name):
        return getattr(self.data, name)
