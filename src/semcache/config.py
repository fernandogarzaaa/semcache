"""Settings, read from ``SEMCACHE_*`` environment variables (12-factor; no config file needed)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "")
    return float(raw) if raw.strip() else default


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "")
    return int(raw) if raw.strip() else default


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class Settings:
    upstream_url: str = "http://localhost:9000/v1"
    upstream_api_key: str = ""
    routes_json: str = ""
    upstream_timeout: float = 120.0
    embedder: str = "hashing"
    embedder_model: str = ""
    hashing_dim: int = 256
    index: str = "auto"
    max_entries: int = 10_000
    normalize: bool = True
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {"classification": 0.88, "general": 0.92, "creative": 0.97}
    )
    adaptive: bool = True
    ttl_short: float = 3600.0
    ttl_default: float = 6 * 3600.0
    ttl_long: float = 7 * 24 * 3600.0
    near_miss_window: float = 0.05
    prices_json: str = ""
    admin_token: str = ""
    enabled: bool = True
    embed_batch: int = 32
    embed_wait_ms: float = 2.0

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Settings:
        e: Mapping[str, str] = os.environ if env is None else env
        s = cls()
        s.upstream_url = e.get("SEMCACHE_UPSTREAM_URL", s.upstream_url)
        s.upstream_api_key = e.get("SEMCACHE_UPSTREAM_API_KEY", "")
        s.routes_json = e.get("SEMCACHE_ROUTES", "")
        s.upstream_timeout = _float(e, "SEMCACHE_UPSTREAM_TIMEOUT", s.upstream_timeout)
        s.embedder = e.get("SEMCACHE_EMBEDDER", s.embedder)
        s.embedder_model = e.get("SEMCACHE_EMBEDDER_MODEL", "")
        s.hashing_dim = _int(e, "SEMCACHE_HASHING_DIM", s.hashing_dim)
        s.index = e.get("SEMCACHE_INDEX", s.index)
        s.max_entries = _int(e, "SEMCACHE_MAX_ENTRIES", s.max_entries)
        s.normalize = _bool(e, "SEMCACHE_NORMALIZE", s.normalize)
        raw = e.get("SEMCACHE_THRESHOLDS", "")
        if raw.strip():
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError('SEMCACHE_THRESHOLDS must be a JSON object, e.g. {"general": 0.92}')
            s.thresholds = {**s.thresholds, **{str(k): float(v) for k, v in data.items()}}
        s.adaptive = _bool(e, "SEMCACHE_ADAPTIVE", s.adaptive)
        s.ttl_short = _float(e, "SEMCACHE_TTL_SHORT", s.ttl_short)
        s.ttl_default = _float(e, "SEMCACHE_TTL_DEFAULT", s.ttl_default)
        s.ttl_long = _float(e, "SEMCACHE_TTL_LONG", s.ttl_long)
        s.near_miss_window = _float(e, "SEMCACHE_NEAR_MISS_WINDOW", s.near_miss_window)
        s.prices_json = e.get("SEMCACHE_PRICES", "")
        s.admin_token = e.get("SEMCACHE_ADMIN_TOKEN", "")
        s.enabled = _bool(e, "SEMCACHE_ENABLED", s.enabled)
        s.embed_batch = _int(e, "SEMCACHE_EMBED_BATCH", s.embed_batch)
        s.embed_wait_ms = _float(e, "SEMCACHE_EMBED_WAIT_MS", s.embed_wait_ms)
        return s

    def redacted(self) -> Dict[str, object]:
        """Settings safe to print or return from an endpoint."""
        out = dict(self.__dict__)
        for k in ("upstream_api_key", "admin_token"):
            out[k] = "***" if out[k] else ""
        return out
