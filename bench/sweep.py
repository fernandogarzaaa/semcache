"""Offline threshold sweep: the hit-rate vs. false-hit trade-off, measured.

1. Replays ``bench/data/workload.jsonl`` through an empty cache at every
   threshold (``semcache.tuner.replay``) for each embedder, with and without
   query normalisation.
2. Scores the labelled QQP pairs (``bench/data/qqp_pairs.jsonl``) at every
   threshold (``semcache.tuner.sweep_pairs``).

Writes ``bench/results/sweep.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from semcache.embedders import HashingEmbedder, SentenceTransformerEmbedder
from semcache.normalize import normalize_query, order_swapped
from semcache.tuner import default_thresholds, pick_threshold, replay, sweep_pairs

DATA = Path("bench/data")
OUT = Path("bench/results")


def main() -> None:
    import sys

    only = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = [json.loads(line) for line in open(DATA / "workload.jsonl")]
    pairs = [json.loads(line) for line in open(DATA / "qqp_pairs.jsonl")]
    intents = [r["intent"] for r in rows]
    thresholds = default_thresholds(0.60, 0.99, 0.01)
    factories = {"minilm": SentenceTransformerEmbedder, "hashing": HashingEmbedder}
    embedders = {k: f() for k, f in factories.items() if not only or k == only}
    raw_texts = [r["prompt"] for r in rows]
    dummy = np.zeros((len(rows), 1), dtype=np.float32)
    result: dict = {
        "requests": len(rows),
        "qqp_pairs": len(pairs),
        "exact_match_baseline": replay(dummy, intents, 1.0, texts=raw_texts, exact_only=True).as_dict(),
        "exact_match_after_normalization": replay(
            dummy, intents, 1.0, texts=[normalize_query(t) for t in raw_texts], exact_only=True
        ).as_dict(),
        "embedders": {},
    }
    previous = OUT / "sweep.json"
    if only and previous.exists():
        result["embedders"] = json.loads(previous.read_text()).get("embedders", {})
    print("exact baseline", result["exact_match_baseline"]["hit_rate"], flush=True)
    for name, emb in embedders.items():
        entry: dict = {"embedder": emb.name}
        for norm in (True, False):
            texts = [normalize_query(r["prompt"]) if norm else r["prompt"] for r in rows]
            t0 = time.perf_counter()
            vecs = emb.embed(texts)
            per_query_ms = (time.perf_counter() - t0) / len(texts) * 1000
            label = "normalized" if norm else "raw"
            no_guard = [replay(vecs, intents, t) for t in thresholds]
            sweep = [replay(vecs, intents, t, texts=texts, guard=order_swapped) for t in thresholds]
            entry[label + "_no_guard"] = {
                "recommended_fhr_1pct": (pick_threshold(no_guard, 0.01) or no_guard[-1]).as_dict(),
                "sweep": [r.as_dict() for r in no_guard],
            }
            entry[label] = {
                "batch_embed_ms_per_query": round(per_query_ms, 3),
                "recommended_fhr_1pct": (pick_threshold(sweep, 0.01) or sweep[-1]).as_dict(),
                "recommended_fhr_3pct": (pick_threshold(sweep, 0.03) or sweep[-1]).as_dict(),
                "sweep": [r.as_dict() for r in sweep],
            }
        q1 = emb.embed([normalize_query(p["q1"]) for p in pairs])
        q2 = emb.embed([normalize_query(p["q2"]) for p in pairs])
        sims = np.sum(q1 * q2, axis=1)
        entry["qqp"] = [p.as_dict() for p in sweep_pairs(sims.tolist(), [p["duplicate"] for p in pairs], thresholds)]
        result["embedders"][name] = entry
        best = entry["normalized"]["recommended_fhr_1pct"]
        print(name, "fhr<=1%:", best, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
