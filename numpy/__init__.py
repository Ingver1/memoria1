import math
import random as _random
from typing import Any, Iterable, List, Sequence


class ndarray(list[Any]):
    """Very small ndarray substitute supporting basic operations used in tests."""

    def __init__(self, data: Iterable[Any] | Any) -> None:
        if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            super().__init__(data)
        else:
            super().__init__([data])
        self.shape: tuple[int, ...] = ()
        self._update_shape()

    def _update_shape(self) -> None:
        if self and isinstance(self[0], list):
            self.shape = (len(self), len(self[0]))
        else:
            self.shape = (len(self),)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        if self.ndim == 2:
            return len(self) * len(self[0])
        return len(self)

    def astype(self, dtype: Any, copy: bool = True) -> "ndarray":
        # dtype is ignored; return copy if requested
        if copy:
            return ndarray([x[:] if isinstance(x, list) else x for x in self])
        return self

    def reshape(self, *shape: int) -> "ndarray":
        if len(shape) == 2 and shape[0] == 1 and shape[1] == -1:
            flat: list[Any] = []
            for x in self:
                if isinstance(x, list):
                    flat.extend(x)
                else:
                    flat.append(x)
            return ndarray([flat])
        raise NotImplementedError

    def __truediv__(self, other: float) -> "ndarray":
        return ndarray([x / other for x in self])

float32 = float
uint8 = int
floating = float

def asarray(obj: Sequence[Any], dtype: Any | None = None) -> "ndarray":
    return ndarray(list(obj))

def array(obj: Sequence[Any], dtype: Any | None = None) -> "ndarray":
    return ndarray(list(obj))
    
def frombuffer(buffer: bytes, dtype: type | int = uint8) -> "ndarray":
    return ndarray(list(buffer))

def tile(arr: "ndarray", reps: int) -> "ndarray":
    data = []
    for _ in range(reps):
        data.extend([row[:] if isinstance(row, list) else row for row in arr])
    return ndarray(data)

def vstack(arrays: List["ndarray"]) -> "ndarray":
    data = []
    for arr in arrays:
        data.extend([row[:] if isinstance(row, list) else row for row in arr])
    return ndarray(data)

def empty(shape: int | tuple[int, ...], dtype: Any = float32) -> "ndarray":
    if isinstance(shape, tuple):
        size = 1
        for s in shape:
            size *= s
    else:
        size = shape
    return ndarray([0.0] * size)

def concatenate(arrays: List["ndarray"], axis: int = 0) -> "ndarray":
    data = []
    for arr in arrays:
        data.extend(list(arr))
    return ndarray(data)

def argsort(arr: "ndarray", axis: int = -1) -> "ndarray":
    indexed = list(enumerate(arr))
    indexed.sort(key=lambda x: x[1])
    return ndarray([i for i, _ in indexed])

def take_along_axis(arr: "ndarray", indices: "ndarray", axis: int) -> "ndarray":
    result = []
    for idx in indices:
        result.append(arr[int(idx)])
    return ndarray(result)

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
    "floating",
    "asarray",
    "array",
    "frombuffer",
    "tile",
    "vstack",
    "linalg",
    "random",
    "testing",
          ]
