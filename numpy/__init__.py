from __future__ import annotations

import builtins as _builtins
import struct
import math
import pickle
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
    def T(self) -> "ndarray":
        return self
        
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        if self.ndim == 2:
            return len(self) * len(self[0])
        return len(self)

    def astype(self, dtype: Any, copy: bool = True) -> "ndarray":
        def _cast(val: Any) -> Any:
            if isinstance(val, list):
                return [_cast(v) for v in val]
            return struct.unpack("f", struct.pack("f", float(val)))[0]

        if copy:
            return ndarray([_cast(x) for x in self])
        for i, x in enumerate(self):
            self[i] = _cast(x)
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
        result: list[Any] = []
        for x in self:
            if isinstance(x, list):
                result.append([float(i) / other for i in x])
            else:
                result.append(float(x) / other)
        return ndarray(result)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, tuple):
            row_idx, col_idx = index
            rows = self[row_idx] if isinstance(row_idx, slice) else [super().__getitem__(row_idx)]
            if not isinstance(rows, list):
                rows = [rows]
            result = []
            for row in rows:
                if isinstance(row, list):
                    result.append(row[col_idx])
                else:
                    result.append(row)
            return result[0] if len(result) == 1 and not isinstance(col_idx, slice) else ndarray(result)
        if isinstance(index, (list, ndarray)):
            return ndarray([self[i] for i in index])
        return super().__getitem__(index)
        
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

def logical_not(arr: "ndarray") -> "ndarray":
    return ndarray([not bool(x) for x in arr])
    
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
if axis in (-1, 1) or arr.ndim == 1:
        result = []
        for row in arr:
            if isinstance(row, list):
                indexed = list(enumerate(row))
                indexed.sort(key=lambda x: x[1])
                result.append([i for i, _ in indexed])
            else:
                result.append(0)
        return ndarray(result if arr.ndim > 1 else result[0])
    elif axis == 0:
        rows = len(arr)
        cols = len(arr[0]) if rows else 0
        cols_sorted = []
        for c in range(cols):
            col = [row[c] for row in arr]
            indexed = list(enumerate(col))
            indexed.sort(key=lambda x: x[1])
            cols_sorted.append([i for i, _ in indexed])
        # transpose back to rows x cols
        transposed = []
        for r in range(rows):
            transposed.append([cols_sorted[c][r] for c in range(cols)])
        return ndarray(transposed)
    else:
        raise NotImplementedError
    
def take_along_axis(arr: "ndarray", indices: "ndarray", axis: int) -> "ndarray":
    if axis == 0:
        k = len(indices)
        cols = len(indices[0]) if k else 0
        out = []
        for i in range(k):
            row_vals = []
            for c in range(cols):
                row_vals.append(arr[indices[i][c]][c])
            out.append(row_vals)
        return ndarray(out)
    elif axis == 1:
        out = []
        for row, idx_row in zip(arr, indices, strict=False):
            row_out = []
            for i in idx_row:
                row_out.append(row[i])
            out.append(row_out)
        return ndarray(out)
    else:
        raise NotImplementedError

def savez(path: str, **arrays: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(arrays, f)


def load(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError("Expected dict in numpy.load stub")
    return data

def isin(arr: "ndarray", test_elements: "ndarray", invert: bool = False) -> "ndarray":
    data = []
    for item in arr:
        found = item in test_elements
        data.append(not found if invert else found)
    return ndarray(data)

def sum(arr: "ndarray") -> Any:
    total = 0.0
    for item in arr:
        total += float(item)
     if axis in (-1, 1) or arr.ndim == 1:
        result = []
        for row in arr:
            if isinstance(row, list):
                indexed = list(enumerate(row))
                indexed.sort(key=lambda x: x[1])
                result.append([i for i, _ in indexed])
            else:
                result.append(0)
        return ndarray(result if arr.ndim > 1 else result[0])
    elif axis == 0:
        rows = len(arr)
        cols = len(arr[0]) if rows else 0
        cols_sorted = []
        for c in range(cols):
            col = [row[c] for row in arr]
            indexed = list(enumerate(col))
            indexed.sort(key=lambda x: x[1])
            cols_sorted.append([i for i, _ in indexed])
        # transpose back to rows x cols
        transposed = []
        for r in range(rows):
            transposed.append([cols_sorted[c][r] for c in range(cols)])
        return ndarray(transposed)
    else:
        raise NotImplementedError   return total
    
class _Linalg:
    @staticmethod
    def norm(
        arr: "ndarray", axis: int | None = None, keepdims: bool = False
    ) -> "ndarray" | float:
        if axis is None:
            flat = [item for sub in arr for item in (sub if isinstance(sub, list) else [sub])]
            return math.sqrt(_builtins.sum(float(x) * float(x) for x in flat))

        if axis == 1:
            result = []
            for row in arr:
                if isinstance(row, list):
                    val = math.sqrt(_builtins.sum(float(x) * float(x) for x in row))
                else:
                    val = float(row)
                result.append(val)
            if keepdims:
                return ndarray([[v] for v in result])
            return ndarray(result)
        raise NotImplementedError

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
    "isin",
    "sum",
    "savez",
    "load",
    "linalg",
    "random",
    "testing",
          ]
