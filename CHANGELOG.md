# Changelog

## Unreleased

## 0.1.1 (2026-10-13)

### Fixed
- Malformed requests now get a 400 instead of a 500: a non-object item in
  `messages` (e.g. `[null]`) and a non-numeric `temperature` on a creative
  prompt both crashed the handler.
- `X-Semcache-Threshold` must be a finite number in [0, 1] and `X-Semcache-TTL`
  a finite number >= 0. `nan` previously disabled hits silently and a negative
  TTL stored already-expired entries.

## 0.1.0 (2026-10-13)

First tagged release.

### Security (Oct 2026 audit)
- Cache entries are partitioned per caller (proxy-token tenant, or a hash of the
  caller's `Authorization` header) and the caller is identified before any cache
  lookup, so cached answers are no longer served across keys or to callers
  without a valid key. `SEMCACHE_SHARED_CACHE=1` opts back into one shared cache.
- Optional proxy authentication: `SEMCACHE_PROXY_TOKENS` (`X-Semcache-Key` or
  `Authorization: Bearer`).
- Admin endpoints are disabled until `SEMCACHE_ADMIN_TOKEN` is set
  (`SEMCACHE_ADMIN_OPEN=1` for local dev); constant-time token comparison.
- `semcache serve` binds `127.0.0.1` by default and refuses open-relay binds.
- SECURITY.md.

### Features

- OpenAI-compatible caching proxy with streaming, namespace/query key split, word-order guard.
- MiniLM (CPU) and hashing embedders, NumPy and FAISS indexes, embed micro-batching.
- Freshness TTL tiers, task-type adaptive thresholds, invalidation by model/system prompt/tag/prefix.
- Prometheus metrics, Grafana dashboard, near-miss log, threshold tuner (`/v1/cache/tune`, `semcache tune`).
- Anthropic adapter, model-prefix routing.
- Benchmarks: offline threshold sweep, HTTP load test, QQP pair sweep. testteeth mutation score 94.3% (611/648, CI re-run; 94.4% locally before).
- README rewritten as onboarding docs: quick start, architecture decisions with rationale, benchmark results checked against `bench/results/*.json`, and an explicit untested list.
