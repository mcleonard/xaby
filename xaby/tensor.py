import jax.numpy as np


class Tensor:
    def __init__(self, object, dtype=None, copy=True, order="K", ndmin=0):
        
        # Duck typing for PyTorch tensors and similar objects
        try:
            object = object.numpy()
        except AttributeError:
            pass

        self.data = np.array(object, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

    def item(self):
        return self.data.item()

    def device(self):
        return self.data.device_buffer.device()

    def __truediv__(self, other):
        return Tensor(self.data / other)

    def __rshift__(self, other):
        return other(self)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor({str(self.data)}, dtype={self.data.dtype.name})"

    def __getattr__(self, name):
        return getattr(self.data, name)
