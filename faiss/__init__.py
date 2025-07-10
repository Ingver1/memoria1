import numpy as np
from types import SimpleNamespace
from typing import Iterable

METRIC_INNER_PRODUCT = 0
METRIC_L2 = 1


def normalize_L2(vecs: np.ndarray) -> None:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms


def swig_ptr(arr: np.ndarray) -> np.ndarray:
    return arr


class IDSelectorBatch:
    def __init__(self, n: int, ptr: np.ndarray) -> None:
        self.ids = ptr


class IndexHNSWFlat:
    def __init__(self, dim: int, M: int, metric: int) -> None:
        self.dim = dim
        self.metric = metric
        self.vectors = np.empty((0, dim), dtype="float32")
        self.ids = np.array([], dtype="int64")
        self.hnsw = SimpleNamespace(efConstruction=0, efSearch=32)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors])
        self.ids = np.concatenate([self.ids, ids])

    def search(self, vecs: np.ndarray, k: int):
        if self.metric == METRIC_INNER_PRODUCT:
            scores = self.vectors @ vecs.T
            distances = -scores
        else:
            diff = self.vectors[:, None, :] - vecs[None, :, :]
            distances = np.linalg.norm(diff, axis=2)
        idx = np.argsort(distances, axis=0)[:k, :]
        dists = np.take_along_axis(distances, idx, axis=0)
        ids = self.ids[idx]
        return dists.T.astype("float32"), ids.T.astype("int64")

    def remove_ids(self, selector: IDSelectorBatch) -> int:
        mask = np.isin(self.ids, selector.ids, invert=True)
        removed = np.sum(~mask)
        self.ids = self.ids[mask]
        self.vectors = self.vectors[mask]
        return removed

    @property
    def ntotal(self) -> int:
        return len(self.ids)


class IndexIDMap2:
    def __init__(self, base: IndexHNSWFlat) -> None:
        self.base = base
        self.id_map = {}

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        for i in ids:
            self.id_map[int(i)] = True
        self.base.add_with_ids(vectors, ids)

    def search(self, vecs: np.ndarray, k: int):
        return self.base.search(vecs, k)

    def remove_ids(self, selector: IDSelectorBatch) -> int:
        for i in selector.ids:
            self.id_map.pop(int(i), None)
        return self.base.remove_ids(selector)

    @property
    def hnsw(self):
        return self.base.hnsw

    @property
    def ntotal(self) -> int:
        return self.base.ntotal


def write_index(index: IndexIDMap2 | IndexHNSWFlat, path: str) -> None:
    np.savez(path, vectors=index.base.vectors if isinstance(index, IndexIDMap2) else index.vectors,
             ids=index.base.ids if isinstance(index, IndexIDMap2) else index.ids,
             metric=index.base.metric if isinstance(index, IndexIDMap2) else index.metric,
             dim=index.base.dim if isinstance(index, IndexIDMap2) else index.dim)


def read_index(path: str) -> IndexIDMap2:
    data = np.load(path)
    base = IndexHNSWFlat(int(data['dim']), 32, int(data['metric']))
    base.vectors = data['vectors']
    base.ids = data['ids']
    return IndexIDMap2(base)
