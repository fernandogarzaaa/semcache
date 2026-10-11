#!/usr/bin/env bash
# Start the mock upstream and the proxy, replay the workload, stop both.
#   EMBEDDER=minilm INDEX=faiss LATENCY_MS=300 JITTER_MS=600 bench/run_loadtest.sh
set -euo pipefail
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
EMBEDDER="${EMBEDDER:-minilm}"
INDEX="${INDEX:-faiss}"
LATENCY_MS="${LATENCY_MS:-300}"
JITTER_MS="${JITTER_MS:-600}"
CONCURRENCY="${CONCURRENCY:-32}"
LIMIT="${LIMIT:-0}"
OUT="${OUT:-bench/results/loadtest_${EMBEDDER}.json}"

python -m semcache mock-upstream --port 19101 --latency-ms "$LATENCY_MS" --jitter-ms "$JITTER_MS" &
MOCK=$!
SEMCACHE_UPSTREAM_URL=http://127.0.0.1:19101/v1 SEMCACHE_EMBEDDER="$EMBEDDER" SEMCACHE_INDEX="$INDEX" \
  python -m semcache serve --host 127.0.0.1 --port 19180 &
PROXY=$!
trap 'kill $MOCK $PROXY 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf --noproxy '*' http://127.0.0.1:19180/healthz >/dev/null 2>&1 && break
  sleep 0.5
done
python bench/loadtest.py --url http://127.0.0.1:19180 --concurrency "$CONCURRENCY" --limit "$LIMIT" --out "$OUT" \
  --label "embedder=$EMBEDDER index=$INDEX mock_latency=${LATENCY_MS}+U(0,${JITTER_MS})ms c=$CONCURRENCY limit=$LIMIT"
