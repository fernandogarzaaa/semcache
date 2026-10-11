"""Vector indexes, partitioned by namespace.

Each namespace gets its own exact inner-product index, so a lookup only ever
scores candidates that are allowed to answer the request. Vectors are unit
length, so inner product equals cosine similarity.

* ``NumpyIndex``: brute-force matrix product, no extra dependency.
* ``FaissIndex``: ``faiss.IndexFlatIP`` wrapped in ``IndexIDMap2``. Same exact
  results, faster at tens of thousands of entries per namespace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

import numpy as np


class VectorIndex(Protocol):
    name: str

    def add(self, namespace: str, item_id: int, vector: np.ndarray) -> None: ...

    def remove(self, namespace: str, item_id: int) -> bool: ...

    def search(self, namespace: str, vector: np.ndarray, k: int = 1) -> List[Tuple[int, float]]: ...

    def __len__(self) -> int: ...


class _NumpyPartition:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: List[int] = []
        self.pos: Dict[int, int] = {}
        self.mat = np.zeros((8, dim), dtype=np.float32)

    def add(self, item_id: int, vector: np.ndarray) -> None:
        if item_id in self.pos:
            self.mat[self.pos[item_id]] = vector
            return
        n = len(self.ids)
        if n == self.mat.shape[0]:
            grown = np.zeros((n * 2, self.dim), dtype=np.float32)
            grown[:n] = self.mat
            self.mat = grown
        self.mat[n] = vector
        self.ids.append(item_id)
        self.pos[item_id] = n

    def remove(self, item_id: int) -> bool:
        p = self.pos.pop(item_id, None)
        if p is None:
            return False
        last = len(self.ids) - 1
        if p != last:
            moved = self.ids[last]
            self.ids[p] = moved
            self.mat[p] = self.mat[last]
            self.pos[moved] = p
        self.ids.pop()
        return True

    def search(self, vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        n = len(self.ids)
        if n == 0:
            return []
        scores = self.mat[:n] @ vector
        k = min(k, n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in top]


class NumpyIndex:
    name = "numpy"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._parts: Dict[str, _NumpyPartition] = {}

    def add(self, namespace: str, item_id: int, vector: np.ndarray) -> None:
        self._parts.setdefault(namespace, _NumpyPartition(self.dim)).add(item_id, _as_vec(vector, self.dim))

    def remove(self, namespace: str, item_id: int) -> bool:
        part = self._parts.get(namespace)
        if part is None:
            return False
        removed = part.remove(item_id)
        if not part.ids:
            del self._parts[namespace]
        return removed

    def search(self, namespace: str, vector: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        part = self._parts.get(namespace)
        if part is None:
            return []
        return part.search(_as_vec(vector, self.dim), k)

    def __len__(self) -> int:
        return sum(len(p.ids) for p in self._parts.values())


class FaissIndex:  # pragma: no cover - exercised in tests only when faiss is installed
    name = "faiss"

    def __init__(self, dim: int) -> None:
        import faiss

        self._faiss: Any = faiss
        self.dim = dim
        self._parts: Dict[str, Any] = {}

    def add(self, namespace: str, item_id: int, vector: np.ndarray) -> None:
        idx = self._parts.get(namespace)
        if idx is None:
            idx = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self.dim))
            self._parts[namespace] = idx
        ids = np.array([item_id], dtype=np.int64)
        idx.remove_ids(ids)
        idx.add_with_ids(_as_vec(vector, self.dim).reshape(1, -1), ids)

    def remove(self, namespace: str, item_id: int) -> bool:
        idx = self._parts.get(namespace)
        if idx is None:
            return False
        removed = int(idx.remove_ids(np.array([item_id], dtype=np.int64))) > 0
        if idx.ntotal == 0:
            del self._parts[namespace]
        return removed

    def search(self, namespace: str, vector: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        idx = self._parts.get(namespace)
        if idx is None or idx.ntotal == 0:
            return []
        scores, ids = idx.search(_as_vec(vector, self.dim).reshape(1, -1), min(k, int(idx.ntotal)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=False) if i >= 0]

    def __len__(self) -> int:
        return sum(int(p.ntotal) for p in self._parts.values())


def _as_vec(vector: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    if v.shape[0] != dim:
        raise ValueError(f"expected a {dim}-dimensional vector, got {v.shape[0]}")
    return v


def make_index(kind: str, dim: int) -> VectorIndex:
    """``numpy``, ``faiss``, or ``auto`` (faiss when importable, else numpy)."""
    kind = kind.lower()
    if kind == "numpy":
        return NumpyIndex(dim)
    if kind == "faiss":
        return FaissIndex(dim)
    if kind == "auto":
        try:
            import faiss  # noqa: F401
        except ImportError:
            return NumpyIndex(dim)
        return FaissIndex(dim)  # pragma: no cover - depends on the environment
    raise ValueError(f"unknown index {kind!r} (expected 'auto', 'numpy' or 'faiss')")
