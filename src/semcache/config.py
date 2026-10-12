"""Settings, read from ``SEMCACHE_*`` environment variables (12-factor; no config file needed)."""

from __future__ import annotations

import hashlib
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


def parse_proxy_tokens(raw: str) -> Dict[str, str]:
    """``SEMCACHE_PROXY_TOKENS``: ``tenant-a:tok1,tenant-b:tok2`` or bare ``tok1,tok2``.

    Returns token -> tenant name. A bare token's tenant is a short hash of it,
    so each token gets its own cache partition.
    """
    out: Dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        tenant, sep, token = item.partition(":")
        if not sep:
            token, tenant = tenant, "t-" + hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:12]
        tenant, token = tenant.strip(), token.strip()
        if not token or not tenant:
            raise ValueError("SEMCACHE_PROXY_TOKENS entries must be 'tenant:token' or 'token'")
        out[token] = tenant
    return out


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
    # Admin endpoints are disabled unless admin_token is set, or this is
    # explicitly turned on (local development only).
    admin_open: bool = False
    # token -> tenant; when non-empty every proxy request must present one.
    proxy_tokens: Dict[str, str] = field(default_factory=dict)
    # Opt-in: share cached answers across callers/tenants (single-tenant use).
    shared_cache: bool = False
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
        s.admin_open = _bool(e, "SEMCACHE_ADMIN_OPEN", s.admin_open)
        s.proxy_tokens = parse_proxy_tokens(e.get("SEMCACHE_PROXY_TOKENS", ""))
        s.shared_cache = _bool(e, "SEMCACHE_SHARED_CACHE", s.shared_cache)
        s.enabled = _bool(e, "SEMCACHE_ENABLED", s.enabled)
        s.embed_batch = _int(e, "SEMCACHE_EMBED_BATCH", s.embed_batch)
        s.embed_wait_ms = _float(e, "SEMCACHE_EMBED_WAIT_MS", s.embed_wait_ms)
        return s

    def redacted(self) -> Dict[str, object]:
        """Settings safe to print or return from an endpoint."""
        out = dict(self.__dict__)
        for k in ("upstream_api_key", "admin_token"):
            out[k] = "***" if out[k] else ""
        out["proxy_tokens"] = sorted(set(self.proxy_tokens.values()))  # tenant names only
        return out

    def has_server_side_keys(self) -> bool:
        """True when the proxy spends its own provider key(s), not the caller's."""
        return bool(self.upstream_api_key) or "api_key_env" in self.routes_json
