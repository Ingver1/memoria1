import math
import random as _random
from typing import Iterable, List


class ndarray(list):
    """Very small ndarray substitute supporting basic operations used in tests."""

    def __init__(self, data):
        if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            super().__init__(data)
        else:
            super().__init__([data])
        self._update_shape()

    def _update_shape(self):
        if self and isinstance(self[0], list):
            self.shape = (len(self), len(self[0]))
        else:
            self.shape = (len(self),)

    def astype(self, dtype, copy=True):
        # dtype is ignored; return copy if requested
        if copy:
            return ndarray([x[:] if isinstance(x, list) else x for x in self])
        return self

float32 = float
uint8 = int

def frombuffer(buffer: bytes, dtype=uint8):
    return ndarray(list(buffer))

def tile(arr: "ndarray", reps: int):
    data = []
    for _ in range(reps):
        data.extend([row[:] if isinstance(row, list) else row for row in arr])
    return ndarray(data)

def vstack(arrays: List["ndarray"]):
    data = []
    for arr in arrays:
        data.extend([row[:] if isinstance(row, list) else row for row in arr])
    return ndarray(data)

class _Linalg:
    @staticmethod
    def norm(arr: "ndarray") -> float:
        if arr and isinstance(arr[0], list):
            flat = [item for sub in arr for item in sub]
        else:
            flat = list(arr)
        return math.sqrt(sum(float(x) * float(x) for x in flat))

linalg = _Linalg()

class _Random:
    @staticmethod
    def rand(*shape: int) -> ndarray:
        if len(shape) == 1:
            return ndarray([_random.random() for _ in range(shape[0])])
        elif len(shape) == 2:
            return ndarray([[_random.random() for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            raise NotImplementedError

random = _Random()

class _Testing:
    @staticmethod
    def assert_array_equal(a: ndarray, b: ndarray) -> None:
        assert list(a) == list(b), f"Arrays not equal: {a} vs {b}"

testing = _Testing()

__all__ = [
    "ndarray",
    "float32",
    "uint8",
    "frombuffer",
    "tile",
    "vstack",
    "linalg",
    "random",
    "testing",
          ]
