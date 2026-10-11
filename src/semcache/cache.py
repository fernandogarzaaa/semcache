"""The semantic cache: entries, lookup, storage, TTL, LRU eviction, invalidation.

Lookups and stores are thread-safe. Embedding is done by the caller (outside the
lock), so a slow embedder never serialises the cache.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .embedders import Embedder
from .index import VectorIndex
from .keys import CacheKey
from .normalize import normalize_query, order_swapped

# Two stored prompts this similar are the same prompt; the newer answer replaces the older.
DUPLICATE_SIMILARITY = 0.9999


@dataclass
class CacheEntry:
    id: int
    namespace: str
    model: str
    system_hash: str
    prompt: str
    normalized: str
    response: Dict[str, Any]
    created_at: float
    expires_at: float
    tier: str = "default"
    task: str = "general"
    tags: Tuple[str, ...] = ()
    hit_count: int = 0
    last_hit_at: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def expired(self, now: float) -> bool:
        return now >= self.expires_at

    def public(self) -> Dict[str, Any]:
        """Entry metadata without the response body (for admin endpoints)."""
        return {
            "id": self.id,
            "model": self.model,
            "system_hash": self.system_hash,
            "prompt": self.prompt,
            "tier": self.tier,
            "task": self.task,
            "tags": list(self.tags),
            "hit_count": self.hit_count,
            "created_at": self.created_at,
            "expires_at": None if math.isinf(self.expires_at) else self.expires_at,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class Lookup:
    hit: bool
    similarity: float
    threshold: float
    entry: Optional[CacheEntry] = None
    nearest_prompt: Optional[str] = None
    near_miss: bool = False


@dataclass
class NearMiss:
    at: float
    model: str
    query: str
    nearest_prompt: str
    similarity: float
    threshold: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "at": self.at,
            "model": self.model,
            "query": self.query,
            "nearest_prompt": self.nearest_prompt,
            "similarity": round(self.similarity, 4),
            "threshold": round(self.threshold, 4),
            "gap": round(self.threshold - self.similarity, 4),
        }


@dataclass
class CacheCounters:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    near_misses: int = 0
    stores: int = 0
    guard_rejections: int = 0
    evictions: Dict[str, int] = field(default_factory=lambda: {"capacity": 0, "expired": 0, "invalidated": 0})


EvictHook = Callable[[str, int], None]


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        *,
        max_entries: int = 10_000,
        normalize: bool = True,
        near_miss_window: float = 0.05,
        near_miss_capacity: int = 500,
        history_capacity: int = 20_000,
        search_k: int = 4,
        order_guard: bool = True,
        clock: Callable[[], float] = time.time,
        on_evict: Optional[EvictHook] = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.embedder = embedder
        self.index = index
        self.max_entries = max_entries
        self.normalize = normalize
        self.near_miss_window = near_miss_window
        self.search_k = max(1, search_k)
        self.order_guard = order_guard
        self.clock = clock
        self.on_evict = on_evict
        self.counters = CacheCounters()
        self._entries: OrderedDict[int, CacheEntry] = OrderedDict()  # LRU order: oldest first
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self._near: Deque[NearMiss] = deque(maxlen=near_miss_capacity)
        # (best similarity, hit?) for every lookup that had a candidate; feeds the threshold tuner.
        self._history: Deque[Tuple[float, bool]] = deque(maxlen=history_capacity)

    # ---- embedding -------------------------------------------------------------------------
    def embedding_text(self, query: str) -> str:
        return normalize_query(query) if self.normalize else query

    def embed(self, query: str) -> np.ndarray:
        return self.embedder.embed([self.embedding_text(query)])[0]

    # ---- lookup -----------------------------------------------------------------------------
    def lookup(self, key: CacheKey, threshold: float, vector: Optional[np.ndarray] = None) -> Lookup:
        vec = self.embed(key.query) if vector is None else vector
        with self._lock:
            now = self.clock()
            self.counters.lookups += 1
            best: Optional[CacheEntry] = None
            best_sim = -1.0
            text = self.embedding_text(key.query)
            for item_id, sim in self.index.search(key.namespace, vec, self.search_k):
                entry = self._entries.get(item_id)
                if entry is None:
                    continue
                if entry.expired(now):
                    self._evict(entry, "expired")
                    continue
                if self.order_guard and sim >= threshold and order_swapped(entry.normalized, text):
                    self.counters.guard_rejections += 1
                    continue
                if sim > best_sim:
                    best, best_sim = entry, sim
            if best is None:
                self.counters.misses += 1
                return Lookup(hit=False, similarity=-1.0, threshold=threshold)
            if best_sim >= threshold:
                best.hit_count += 1
                best.last_hit_at = now
                self._entries.move_to_end(best.id)
                self.counters.hits += 1
                self._history.append((best_sim, True))
                return Lookup(
                    hit=True, similarity=best_sim, threshold=threshold, entry=best, nearest_prompt=best.prompt
                )
            self.counters.misses += 1
            self._history.append((best_sim, False))
            near = best_sim >= threshold - self.near_miss_window
            if near:
                self.counters.near_misses += 1
                self._near.append(NearMiss(now, key.model, key.query, best.prompt, best_sim, threshold))
            return Lookup(
                hit=False, similarity=best_sim, threshold=threshold, nearest_prompt=best.prompt, near_miss=near
            )

    # ---- store ------------------------------------------------------------------------------
    def store(
        self,
        key: CacheKey,
        response: Mapping[str, Any],
        *,
        ttl: float,
        vector: Optional[np.ndarray] = None,
        tier: str = "default",
        task: str = "general",
        tags: Iterable[str] = (),
    ) -> Optional[CacheEntry]:
        """Store a complete response. ``ttl <= 0`` stores nothing; ``inf`` never expires."""
        if ttl <= 0:
            return None
        vec = self.embed(key.query) if vector is None else vector
        usage = response.get("usage") or {}
        with self._lock:
            now = self.clock()
            for item_id, sim in self.index.search(key.namespace, vec, 1):
                old = self._entries.get(item_id)
                if old is not None and sim >= DUPLICATE_SIMILARITY:
                    self._remove(old)
            while len(self._entries) >= self.max_entries:
                _, oldest = next(iter(self._entries.items()))
                self._evict(oldest, "capacity")
            entry = CacheEntry(
                id=next(self._ids),
                namespace=key.namespace,
                model=key.model,
                system_hash=key.system_hash,
                prompt=key.query,
                normalized=self.embedding_text(key.query),
                response=dict(response),
                created_at=now,
                expires_at=now + ttl,
                tier=tier,
                task=task,
                tags=tuple(sorted(set(tags))),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            )
            self._entries[entry.id] = entry
            self.index.add(key.namespace, entry.id, vec)
            self.counters.stores += 1
            return entry

    # ---- removal ----------------------------------------------------------------------------
    def _remove(self, entry: CacheEntry) -> None:
        self._entries.pop(entry.id, None)
        self.index.remove(entry.namespace, entry.id)

    def _evict(self, entry: CacheEntry, reason: str) -> None:
        self._remove(entry)
        self.counters.evictions[reason] = self.counters.evictions.get(reason, 0) + 1
        if self.on_evict is not None:
            self.on_evict(reason, 1)

    def invalidate(
        self,
        *,
        model: Optional[str] = None,
        system_hash: Optional[str] = None,
        tag: Optional[str] = None,
        prefix: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """Remove every entry matching *all* given filters. At least one filter is required.

        ``model`` also matches dated snapshots (``gpt-4o`` matches
        ``gpt-4o-2024-08-06`` but not ``gpt-4o-mini``); ``prefix`` matches the
        start of the original prompt.
        """
        if model is None and system_hash is None and tag is None and prefix is None and namespace is None:
            raise ValueError("invalidate needs at least one filter")
        with self._lock:
            doomed = [
                e
                for e in self._entries.values()
                if (model is None or _same_model_family(e.model, model))
                and (system_hash is None or e.system_hash == system_hash)
                and (tag is None or tag in e.tags)
                and (prefix is None or e.prompt.startswith(prefix))
                and (namespace is None or e.namespace == namespace)
            ]
            for e in doomed:
                self._evict(e, "invalidated")
            return len(doomed)

    def purge_expired(self) -> int:
        with self._lock:
            now = self.clock()
            doomed = [e for e in self._entries.values() if e.expired(now)]
            for e in doomed:
                self._evict(e, "expired")
            return len(doomed)

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            for e in list(self._entries.values()):
                self._evict(e, "invalidated")
            return n

    # ---- introspection ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def get(self, entry_id: int) -> Optional[CacheEntry]:
        with self._lock:
            return self._entries.get(entry_id)

    def entries(self) -> List[CacheEntry]:
        with self._lock:
            return list(self._entries.values())

    def near_misses(self, limit: int = 50) -> List[NearMiss]:
        with self._lock:
            items = list(self._near)
        return items[::-1][:limit]

    def history(self) -> List[Tuple[float, bool]]:
        with self._lock:
            return list(self._history)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            c = self.counters
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "lookups": c.lookups,
                "hits": c.hits,
                "misses": c.misses,
                "hit_rate": round(c.hits / c.lookups, 4) if c.lookups else 0.0,
                "near_misses": c.near_misses,
                "guard_rejections": c.guard_rejections,
                "stores": c.stores,
                "evictions": dict(c.evictions),
                "embedder": self.embedder.name,
                "index": self.index.name,
            }


def _same_model_family(entry_model: str, model: str) -> bool:
    """``model`` itself, or a dated snapshot of it (``<model>-<digit>...``)."""
    if entry_model == model:
        return True
    rest = entry_model[len(model) :]
    return entry_model.startswith(model) and len(rest) >= 2 and rest[0] == "-" and rest[1].isdigit()
