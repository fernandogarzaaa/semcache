# Changelog

## 0.1.0 (unreleased)

- OpenAI-compatible caching proxy with streaming, namespace/query key split, word-order guard.
- MiniLM (CPU) and hashing embedders, NumPy and FAISS indexes, embed micro-batching.
- Freshness TTL tiers, task-type adaptive thresholds, invalidation by model/system prompt/tag/prefix.
- Prometheus metrics, Grafana dashboard, near-miss log, threshold tuner (`/v1/cache/tune`, `semcache tune`).
- Anthropic adapter, model-prefix routing.
- Benchmarks: offline threshold sweep, HTTP load test, QQP pair sweep. testteeth mutation score 94.3% (611/648, CI re-run; 94.4% locally before).
- README rewritten as onboarding docs: quick start, architecture decisions with rationale, benchmark results checked against `bench/results/*.json`, and an explicit untested list.
