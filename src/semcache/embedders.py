"""Embedders: turn query text into unit-length vectors.

* ``HashingEmbedder``: deterministic, dependency-free feature hashing over word
  tokens and character trigrams. It understands spelling overlap, not meaning,
  so it is the test/CI double, not a production embedder.
* ``SentenceTransformerEmbedder``: a local sentence-transformers model on CPU
  (default ``all-MiniLM-L6-v2``, 384 dimensions, ~10 ms per query on 2 vCPUs).
  Free, no API key, the model downloads once from Hugging Face.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, List, Protocol, Sequence

import numpy as np

from .normalize import _stem_token

_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with L2-normalised rows."""
        ...


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalise rows to unit length; all-zero rows stay zero."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def _stem(word: str) -> str:
    return _stem_token(word)


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
    return value % dim, sign


class HashingEmbedder:
    """Feature-hashing embedder. Deterministic across runs, processes and platforms."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def features(self, text: str) -> List[str]:
        words = [_stem(w) for w in _WORD_RE.findall(text.lower())]
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"#{w}#"
            feats.extend(f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2))
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self.features(text):
                idx, sign = _bucket(feat, self.dim)
                weight = 2.0 if feat.startswith("w:") else 1.0
                out[row, idx] += sign * weight
        return l2_normalize(out)


class SentenceTransformerEmbedder:  # pragma: no cover - needs the optional model download
    """Local sentence-transformers model. Install with ``pip install 'semcache[local]'``."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "SEMCACHE_EMBEDDER=minilm needs sentence-transformers: pip install 'semcache[local]'"
            ) from exc
        self._model: Any = SentenceTransformer(model_name, device=device)
        getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        dim = getter()
        self.dim = int(dim) if dim is not None else len(self.embed(["probe"])[0])
        self.name = model_name.rsplit("/", 1)[-1]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(kind: str, model: str = "", dim: int = 256) -> Embedder:
    """Build an embedder from config: ``hashing`` or ``minilm``/``sentence-transformers``."""
    kind = kind.lower()
    if kind == "hashing":
        return HashingEmbedder(dim=dim)
    if kind in ("minilm", "sentence-transformers", "st"):  # pragma: no cover - optional dependency
        return SentenceTransformerEmbedder(model or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"unknown embedder {kind!r} (expected 'hashing' or 'minilm')")
