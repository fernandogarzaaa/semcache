# semcache

A drop-in semantic cache for OpenAI-compatible chat APIs. Point your client's `base_url` at semcache instead of the provider; when a new prompt means the same thing as one already answered, semcache returns the stored answer instead of paying for another call.

- **What it is:** a FastAPI proxy for `POST /v1/chat/completions` (streaming and non-streaming), plus admin endpoints, Prometheus metrics, and a Grafana dashboard.
- **What it buys you:** on the bundled benchmark workload it served 65.1% of requests from cache, cutting estimated spend from $6.55 to $2.22 at gpt-4o list prices, with 0.51% of hits being wrong answers. Exact-match caching on the same workload gets 47.1%.
- **What it costs:** one embedding per request (≈0.07 ms with the built-in hashing embedder, ≈4 ms per query batched with MiniLM on CPU), and a small, measured rate of wrong answers that you tune with the threshold.
- **Status:** 0.1.0, unreleased. Tested against a mock upstream only. See [What has not been tested](#what-has-not-been-tested) before putting real traffic through it.

## Contents

- [Quick start](#quick-start)
- [How a request flows](#how-a-request-flows)
- [Architecture decisions](#architecture-decisions)
- [Results](#results)
- [Configuration](#configuration)
- [Operating it](#operating-it)
- [Where it fits with llmgate and costroute](#where-it-fits-with-llmgate-and-costroute)
- [What has not been tested](#what-has-not-been-tested)
- [Development](#development)

## Quick start

Needs Python 3.10+. No API key: the bundled mock upstream answers deterministically.

```bash
pip install -e ".[faiss,local]"        # local = sentence-transformers (MiniLM on CPU)

semcache mock-upstream --port 9000 &   # stand-in for OpenAI
SEMCACHE_UPSTREAM_URL=http://127.0.0.1:9000/v1 semcache serve --port 8080 --embedder minilm
```

If your shell has an `HTTP_PROXY` set, also export `NO_PROXY=127.0.0.1,localhost`, or the proxy's upstream calls to the mock go through it and fail.

Ask a question, then a paraphrase, then something that is only loosely related:

```bash
ask() {
  curl -s -D - -o /dev/null localhost:8080/v1/chat/completions \
    -H 'content-type: application/json' \
    -d "{\"model\":\"gpt-4o-mini\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}" \
    | grep -iE '^x-semcache(:|-similarity)'
}
ask "How do I undo my last git commit?"
ask "hey, how do I undo my last git commit? thanks"
ask "take back my last commit"
```

What you should see (MiniLM, default `general` threshold 0.92):

| Prompt | `X-Semcache` | `X-Semcache-Similarity` | Why |
|---|---|---|---|
| How do I undo my last git commit? | `miss` | `none` | Empty cache; answer is fetched and stored. |
| hey, how do I undo my last git commit? thanks | `hit` | `1.0000` | Normalisation strips "hey" / "thanks", so the embedded text is identical. Without normalisation MiniLM scores the pair 0.978, still a hit. |
| take back my last commit | `miss` | `0.7356` | Same intent, different words. Well below 0.92, so it goes upstream. |

The third row is the honest picture: a sentence embedder at a safe threshold catches rewordings, not every rephrasing. Lowering the threshold catches more and serves more wrong answers; [Results](#results) shows that trade-off measured.

With Docker (hashing embedder, no model download), the full stack with Prometheus and Grafana:

```bash
docker compose up --build    # proxy :8080, Grafana :3000, Prometheus :9090, mock upstream
```

To use a real provider, set `SEMCACHE_UPSTREAM_URL=https://api.openai.com/v1` and send your normal `Authorization: Bearer ...` header; semcache forwards the client's header unless `SEMCACHE_UPSTREAM_API_KEY` is set.

## How a request flows

```
client ──> semcache ──────────────────────────────────────────────> upstream (OpenAI-compatible / Anthropic)
             1. build key: namespace hash (model, system prompt, params, earlier turns)
                           + query (last user message)
             2. bypass?  live-data prompt, creative + high temperature, X-Semcache-Bypass, no user turn
             3. normalise query, embed (micro-batched), search the namespace's index
             4. best match >= threshold for this task type, and not word-order swapped?
                    yes -> return stored answer   (X-Semcache: hit)
                    no  -> forward, store answer with a TTL from its freshness tier (X-Semcache: miss)
```

Every response carries `X-Semcache` (`hit`, `miss`, `bypass`), the best similarity, and the threshold used, so a wrong answer can be traced to the score that produced it.

## Architecture decisions

### 1. The key is an exact hash of the context plus similarity on the last user turn only

`keys.py` splits each request in two:

- **Namespace** — SHA-256 of everything that must match exactly for two requests to share an answer: model, system prompt, sampling parameters (temperature, top_p, max_tokens, stop, seed, tools, response_format and the rest), and the whole conversation before the last user message.
- **Query** — the last user message, which is the only thing embedded and compared, and only against entries in the same namespace.

**Why:** embedding the whole conversation lets a long shared system prompt dominate the vector, so unrelated questions under the same system prompt look 0.95-similar. Hashing the context makes cross-contamination impossible by construction: "Summarise this" under a legal-review prompt is never answered from a marketing prompt, and a temperature-0 answer is never served to a temperature-1 request. The cost is that a cosmetic change to the system prompt empties the effective cache for that prompt, which is the safe direction to fail.

### 2. A word-order guard runs after the similarity check

Short prompts embed close to a bag of words: "convert km to miles" and "convert miles to km" score about 0.98 and need opposite answers. `normalize.order_swapped` rejects a would-be hit when the two prompts have the same content words in a different order, or are both directional ("X to Y", "X vs Y", "X than Y") with shared words in a different relative order.

**Why:** a miss costs one upstream call; a false hit costs a wrong answer. On the benchmark at threshold 0.92 (MiniLM, normalised) the guard drops the false-hit rate from 1.70% to 0.51% for a hit-rate cost of 0.4 points (66.0% → 65.6%). Rejections are counted in `guard_rejections` so you can see when it fires.

### 3. Embeddings are micro-batched

`batching.EmbedBatcher` queues concurrent lookups, waits at most `SEMCACHE_EMBED_WAIT_MS` (default 2 ms) for company, and embeds up to `SEMCACHE_EMBED_BATCH` (default 32) prompts in one forward pass on a worker thread.

**Why:** a CPU transformer embeds 32 short prompts in one pass far faster than 32 separate passes, and separate passes from many request handlers fight over the same cores. In the 32-concurrent MiniLM load test the mean batch was 9.6 prompts (250 batches for 2,400 requests). Under no load a request waits only the 2 ms window.

### 4. TTL tiers instead of one global expiry

`policy.classify_freshness` puts each prompt in a tier with simple, readable regexes:

| Tier | Example prompts | Default TTL | Env var |
|---|---|---|---|
| `no_cache` | "right now", "live score", "breaking", "real-time" | not cached | — |
| `short` | "today", "latest", "current", "price", "weather", a year like 2026 | 1 hour | `SEMCACHE_TTL_SHORT` |
| `default` | everything else | 6 hours | `SEMCACHE_TTL_DEFAULT` |
| `long` | starts with "what is", "explain", "how do", "history of", "define" | 7 days | `SEMCACHE_TTL_LONG` |

**Why:** the right expiry depends on whether the answer can go stale, and that is visible in the wording far more often than not. Rules instead of a model keep it inspectable, and `X-Semcache-TTL` overrides the tier per request when the caller knows better.

### 5. Thresholds differ by task type and adapt to feedback

`policy.classify_task` labels each prompt `classification` (constrained answers such as sentiment, yes/no, extraction: 0.88), `general` (0.92) or `creative` (poems, stories, slogans: 0.97, and not cached at all at high temperature). `AdaptiveThresholds` moves each task type's threshold from feedback posted to `/v1/cache/feedback`: a reported wrong hit raises the threshold above that hit's similarity; a confirmed good hit lowers it slightly, never below the floor or the worst wrong hit seen.

**Why:** one threshold is wrong for everything. A classifier's answer space is small, so a looser match is safe; a creative request wants a fresh answer. Feedback lets production traffic correct the starting values without a redeploy. Turn it off with `SEMCACHE_ADAPTIVE=0`.

### Smaller decisions

- **Query normalisation** (`normalize.py`) strips leading filler ("hey", "quick question", "can you please") and trailing thanks before embedding. The original prompt is still stored for debugging.
- **Two embedders.** `hashing` (default; 256-dim feature hashing, no model download, ≈0.07 ms per query) and `minilm` (`all-MiniLM-L6-v2` via sentence-transformers on CPU). MiniLM is better on real paraphrases; hashing is good enough for near-duplicates and keeps the Docker image small.
- **Two indexes.** FAISS when installed, a NumPy brute-force index otherwise (`SEMCACHE_INDEX=auto`). Both are partitioned by namespace. CI runs the suite without FAISS too.
- **Streaming.** Misses stream through byte for byte and are stored when the stream completes; hits are replayed as SSE.
- **Providers.** Any OpenAI-compatible endpoint (OpenAI, Ollama `/v1`, vLLM, llama.cpp server, Groq, OpenRouter, llmgate). An Anthropic adapter translates to and from the Messages API (text only). `SEMCACHE_ROUTES` picks a provider by model-name prefix; the model is part of the namespace, so one provider's answer is never served for another's model.

## Results

All numbers come from `bench/results/*.json` in this repo, produced on a 2-vCPU sandbox against the mock upstream. Regenerate with `make bench`.

**The workload** (`bench/make_workload.py`, fixed seed): 2,400 requests in arrival order. About 75% are paraphrases of hand-written intents (`bench/intents.py`) with Zipf popularity, so a few questions come up a lot and most rarely. About 25% are one-off questions from the GLUE QQP validation split with no paraphrase in the stream, so any hit on them is a false hit. **False-hit rate** = hits whose stored answer belongs to a different intent ÷ hits.

### Hit rate against the obvious alternatives (offline replay, `sweep.json`)

| Strategy | Hit rate | False-hit rate |
|---|---|---|
| Exact match on the raw prompt | 47.1% | 0% |
| Exact match after normalisation | 59.9% | 0% |
| semcache, hashing embedder @ 0.92 | 63.4% | 0.53% |
| **semcache, MiniLM @ 0.92 (default)** | **65.6%** | **0.51%** |
| semcache, MiniLM @ 0.90 (highest hit rate with false hits ≤ 1%) | 66.8% | 1.0% |
| semcache, MiniLM @ 0.92, word-order guard off | 66.0% | 1.70% |

Semantic matching adds 5.7 points over normalised exact match on this workload, and 18.5 over a plain exact-match cache. Most of the gain over raw exact match comes from normalisation alone, which is worth knowing if you want zero false hits.

### Load test through the HTTP proxy (`loadtest_*.json`)

2,400 requests, 32 concurrent clients, mock upstream latency 300 ms + uniform 0–600 ms, FAISS index, 0 errors in every run.

| Embedder | Hit rate | False-hit rate | Throughput | gpt-4o cost without → with cache | gpt-4o-mini cost without → with |
|---|---|---|---|---|---|
| **MiniLM** | **65.1%** | 0.51% (8 of 1,563) | **54.0 req/s** | **$6.55 → $2.22** (−66.1%) | $0.393 → $0.133 |
| hashing | 62.5% | 0.47% (7 of 1,499) | 126.4 req/s | $6.49 → $2.39 (−63.1%) | $0.389 → $0.144 |

Cost is estimated from the token counts in the mock's responses at list prices built into `pricing.py` (overridable with `SEMCACHE_PRICES`); it is not a bill. The hit rate climbs as the cache fills: for MiniLM it was 35% in the first 200 requests and roughly 65–75% from request 600 on.

**Latency (ms)**

| Run | Hit p50 / p95 / p99 | Miss p50 / p95 / p99 | All p50 / p95 / p99 |
|---|---|---|---|
| hashing, 32 concurrent | 13.0 / 27.4 / 38.5 | 640.4 / 892.1 / 920.4 | 19.3 / 846.2 / 905.5 |
| MiniLM, 32 concurrent | 332.7 / 800.7 / 988.7 | 944.3 / 1430.2 / 1726.2 | 448.2 / 1239.9 / 1571.2 |
| MiniLM, 4 concurrent (400 requests) | 64.5 / 202.3 / 348.9 | 667.9 / 966.6 / 1210.6 | 429.6 / 930.9 / 1001.8 |

Read the MiniLM rows carefully: at 32 concurrent on 2 CPU cores the embedder is the bottleneck, so hits queue behind it and miss latency rises above the upstream's own (~600 ms mean) by about 370 ms. At 4 concurrent a MiniLM hit is ~65 ms at p50. If you need MiniLM at high concurrency, give it more cores or a GPU (untested), or use the hashing embedder, whose hits stay under 40 ms at p99.

### Caution: open-domain questions (QQP pairs, `sweep.json`)

On 4,000 labelled Quora Question Pairs, MiniLM at 0.92 lets through only 31.9% of true duplicates, and **18.5% of the pairs that clear the threshold are not duplicates** (4.2% of all non-duplicate pairs clear it). The hashing embedder is worse (7.0% recall, 51.7% of passing pairs wrong).

That is far worse than the 0.51% on the main workload, because QQP is built from hard near-misses ("culture at Sprint" vs "culture at Apache", "sort a dictionary by key" vs "by value"). Several of the false hits in the load test are exactly this shape. Treat the headline numbers as valid for repetitive traffic (support bots, internal FAQ, classification) and **re-measure on your own traffic before trusting it with open-domain questions**: `semcache tune your.jsonl` and the near-miss log are there for that.

## Configuration

Everything is an environment variable (`config.py`); there is no config file.

| Variable | Default | Meaning |
|---|---|---|
| `SEMCACHE_UPSTREAM_URL` | `http://localhost:9000/v1` | OpenAI-compatible base URL |
| `SEMCACHE_UPSTREAM_API_KEY` | (empty) | If set, replaces the client's `Authorization` header |
| `SEMCACHE_ROUTES` | (empty) | JSON list of model-prefix routes, including `anthropic` providers |
| `SEMCACHE_EMBEDDER` | `hashing` | `hashing` or `minilm` |
| `SEMCACHE_INDEX` | `auto` | `auto`, `faiss`, or `numpy` |
| `SEMCACHE_MAX_ENTRIES` | `10000` | Capacity before eviction |
| `SEMCACHE_NORMALIZE` | `true` | Strip filler before embedding |
| `SEMCACHE_THRESHOLDS` | `{"classification":0.88,"general":0.92,"creative":0.97}` | JSON, merged over the defaults |
| `SEMCACHE_ADAPTIVE` | `true` | Move thresholds from feedback |
| `SEMCACHE_TTL_SHORT` / `_DEFAULT` / `_LONG` | 3600 / 21600 / 604800 s | Freshness tiers |
| `SEMCACHE_NEAR_MISS_WINDOW` | `0.05` | Log lookups within this distance below the threshold |
| `SEMCACHE_EMBED_BATCH` / `SEMCACHE_EMBED_WAIT_MS` | 32 / 2.0 | Micro-batching |
| `SEMCACHE_ADMIN_TOKEN` | (empty) | Bearer token required on `/v1/cache/*` when set |
| `SEMCACHE_ENABLED` | `true` | `false` turns semcache into a plain pass-through |

Per-request headers: `X-Semcache-Bypass: 1`, `X-Semcache-Threshold: 0.95`, `X-Semcache-TTL: 600`, `X-Semcache-Tags: billing,faq` (tags can be invalidated later).

## Operating it

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness, version, entry count |
| `GET /metrics` | Prometheus: requests by result, latency by result, similarity histogram, saved USD and tokens |
| `GET /v1/cache/stats` | Hit rate, near misses, guard rejections, evictions, current thresholds, batching stats |
| `GET /v1/cache/entries` | Recent entries |
| `POST /v1/cache/invalidate` | Drop entries by model, system prompt, tag, or prompt prefix (after a prompt or model change) |
| `POST /v1/cache/feedback` | Report a hit as good or bad; feeds the adaptive thresholds |
| `GET /v1/cache/near-misses` | Lookups that just missed: candidates for lowering a threshold |
| `POST /v1/cache/tune` | Hit rate each threshold would have produced on live history |

`semcache tune workload.jsonl` runs the offline replay on your own labelled prompts (`{"prompt", "intent"}` per line) and recommends a threshold for a target false-hit rate. `semcache stats` prints a running proxy's stats.

The Grafana dashboard is provisioned from `deploy/grafana/` by `docker compose up`.

The cache is in process memory: it is lost on restart and not shared between replicas.

## Where it fits with llmgate and costroute

semcache only sees OpenAI-shaped requests, so it stacks in front of either of Fernando's other gateways without code changes:

```
app ──> semcache ──> llmgate (gateway) or costroute (cost-based model routing) ──> providers
```

Put semcache first so a hit never reaches the gateway at all. Point `SEMCACHE_UPSTREAM_URL` at the gateway's `/v1` base URL. One thing to check when combining with a router: if the router changes the model after semcache has keyed the request, the cached answer is keyed by the model the client asked for, which is usually what you want.

This combination has not been run end to end yet.

## What has not been tested

Stated plainly so nobody finds out in production:

- **Real providers.** Every test and benchmark ran against the bundled mock upstream. No call has gone to OpenAI, Ollama, vLLM, or any hosted API.
- **Anthropic beyond the mock.** The Messages API adapter is unit-tested against a mocked transport only; streaming from Anthropic is one upstream call replayed as SSE.
- **Redis or any shared store.** Not implemented. The cache is per process, in memory, and lost on restart.
- **Multiple replicas.** Follows from the above: each replica has its own cache and its own adaptive thresholds.
- **The Grafana dashboard visually.** The JSON is provisioned and the queries match the metric names, but nobody has looked at it rendered.
- **GPU embedding.** MiniLM has only run on CPU.
- **llmgate / costroute in front.** Expected to work (both speak the OpenAI shape); not run.

## Development

```bash
pip install -e ".[dev,faiss]"
make lint        # ruff check, ruff format --check, mypy (strict-ish, src only)
make test        # unit + integration, no network, no model download
make e2e         # starts real proxy + mock processes
make mutation    # testteeth mutation score of the core modules
make bench       # regenerate the workload, sweep, and load test (downloads QQP and MiniLM)
```

**Test quality.** 226 tests (225 unit and integration, 1 end-to-end). Mutation testing with [testteeth](https://pypi.org/project/testteeth/) over the dependency-light core (`keys`, `normalize`, `embedders`, `policy`, `cache`, `index`, `tuner`, `pricing`, `streaming`) kills **94.3% (611 / 648)** of generated mutants (CI run of 2026-10-11; lowest module is `index.py` at 85.5%, `keys.py` and `streaming.py` are at 100%). CI runs lint, the test matrix on Python 3.10–3.12 (plus one run without FAISS), a wheel install smoke test, a Docker build with a proxy-plus-mock container smoke test, and the mutation score as a non-blocking job.

## License

MIT
