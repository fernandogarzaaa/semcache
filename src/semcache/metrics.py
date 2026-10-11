"""Prometheus metrics. Each app instance owns its registry (tests build many apps)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
SIMILARITY_BUCKETS = tuple(round(0.5 + i * 0.025, 3) for i in range(21))


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "semcache_requests_total", "Chat requests by model and cache result", ["model", "result"], registry=r
        )
        self.latency = Histogram(
            "semcache_request_seconds",
            "End-to-end proxy latency by cache result",
            ["result"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.similarity = Histogram(
            "semcache_similarity",
            "Best cosine similarity per lookup, split into hits, near misses and misses",
            ["outcome"],
            buckets=SIMILARITY_BUCKETS,
            registry=r,
        )
        self.saved_usd = Counter(
            "semcache_saved_usd_total", "Estimated upstream spend avoided by cache hits (USD)", ["model"], registry=r
        )
        self.saved_tokens = Counter(
            "semcache_saved_tokens_total", "Upstream tokens avoided by cache hits", ["model", "kind"], registry=r
        )
        self.upstream_usd = Counter(
            "semcache_upstream_usd_total",
            "Estimated spend on upstream calls that went through (USD)",
            ["model"],
            registry=r,
        )
        self.evictions = Counter("semcache_evictions_total", "Entries removed, by reason", ["reason"], registry=r)
        self.entries = Gauge("semcache_entries", "Entries currently cached", registry=r)
        self.capacity = Gauge("semcache_capacity_entries", "Configured maximum entries", registry=r)
        self.threshold = Gauge("semcache_threshold", "Current similarity threshold per task type", ["task"], registry=r)
        self.upstream_errors = Counter(
            "semcache_upstream_errors_total", "Upstream responses that were not cacheable", ["reason"], registry=r
        )
