#!/usr/bin/env bash
# Mutation-test each core module against its own unit tests with testteeth
# (https://pypi.org/project/testteeth/). Usage: scripts/mutation.sh [module ...]
# Writes mutation/<module>.json; scripts/mutation_summary.py aggregates them.
set -euo pipefail
declare -A TESTS=(
  [keys]=tests/unit/test_keys.py
  [normalize]=tests/unit/test_normalize_embedders.py
  [embedders]=tests/unit/test_normalize_embedders.py
  [policy]=tests/unit/test_policy.py
  [cache]=tests/unit/test_cache.py
  [index]=tests/unit/test_index.py
  [tuner]=tests/unit/test_tuner_pricing_streaming.py
  [pricing]=tests/unit/test_tuner_pricing_streaming.py
  [streaming]=tests/unit/test_tuner_pricing_streaming.py
)
MODULES=("$@")
[ ${#MODULES[@]} -eq 0 ] && MODULES=(keys normalize embedders policy cache index tuner pricing streaming)
mkdir -p mutation
for m in "${MODULES[@]}"; do
  testteeth run -q --workers "${WORKERS:-2}" --pytest-args "${TESTS[$m]} tests/unit/test_hardening.py -q -x -p no:cacheprovider" \
    --json "mutation/$m.json" "src/semcache/$m.py" | tail -4
done
