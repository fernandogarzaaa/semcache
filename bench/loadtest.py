"""HTTP load test: replay the workload through a running semcache proxy.

Usage (see bench/run_loadtest.sh, which starts the mock upstream and proxy):

    python bench/loadtest.py --url http://127.0.0.1:8080 --concurrency 32

Ground truth for false hits: the mock upstream's answer starts with a tag that
is a hash of the prompt that produced it (``semcache.mock_upstream.mock_answer``),
so the script knows which original prompt, and therefore which intent, every
served answer belongs to. A hit is false when that intent differs from the
request's intent (or the request is a one-off with no paraphrase).

Writes bench/results/loadtest.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

import httpx

from semcache.mock_upstream import mock_answer
from semcache.pricing import PriceTable

TAG_RE = re.compile(r"^\[answer:(\d{7})\]")
MODEL = "gpt-4o-mini"
SYSTEM = "You are a concise, accurate assistant for a software company."


def body(prompt: str) -> dict:
    return {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}


def tag_for(prompt: str) -> str:
    b = body(prompt)
    seed = json.dumps([MODEL, b["messages"], None], sort_keys=True)
    text, _ = mock_answer(seed)
    m = TAG_RE.match(text)
    assert m
    return m.group(1)


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q / 100 * (len(s) - 1)))))
    return s[k]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--workload", default="bench/data/workload.jsonl")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default="bench/results/loadtest.json")
    ap.add_argument("--label", default="")
    ap.add_argument("--limit", type=int, default=0, help="only the first N requests")
    args = ap.parse_args()

    with open(args.workload, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    if args.limit:
        rows = rows[: args.limit]
    tag_to_intent: dict[str, object] = {}
    tag_to_prompt: dict[str, str] = {}
    for r in rows:
        t = tag_for(r["prompt"])
        tag_to_intent[t] = r["intent"]
        tag_to_prompt[t] = r["prompt"]

    results: list[dict] = [{} for _ in rows]
    sem = asyncio.Semaphore(args.concurrency)
    # Requests are released in workload order; completion order varies with concurrency.
    async with httpx.AsyncClient(base_url=args.url, timeout=60, trust_env=False) as client:

        async def one(i: int, row: dict) -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.post("/v1/chat/completions", json=body(row["prompt"]))
                dt = time.perf_counter() - t0
            data = r.json()
            text = data["choices"][0]["message"]["content"] if r.status_code == 200 else ""
            m = TAG_RE.match(text)
            served = tag_to_intent.get(m.group(1), "<unknown>") if m else "<none>"
            results[i] = {
                "status": r.status_code,
                "cache": r.headers.get("x-semcache", ""),
                "latency": dt,
                "similarity": r.headers.get("x-semcache-similarity"),
                "intent": row["intent"],
                "served_intent": served,
                "served_prompt": tag_to_prompt.get(m.group(1), "") if m else "",
                "prompt": row["prompt"],
                "usage": data.get("usage") or {},
                "finished_at": time.perf_counter(),
            }

        t_start = time.perf_counter()
        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
        wall = time.perf_counter() - t_start
        stats = (await client.get("/v1/cache/stats")).json()

    hits = [r for r in results if r["cache"] == "hit"]
    misses = [r for r in results if r["cache"] == "miss"]
    false_hits = [r for r in hits if r["intent"] is None or r["served_intent"] != r["intent"]]
    errors = [r for r in results if r["status"] != 200]
    by_time = sorted(results, key=lambda r: r["finished_at"])
    window = 200
    convergence = [
        round(sum(r["cache"] == "hit" for r in by_time[s : s + window]) / len(by_time[s : s + window]), 4)
        for s in range(0, len(by_time), window)
    ]

    prices = PriceTable()

    def spend(rs: list[dict], model: str) -> float:
        return sum(
            prices.cost(model, r["usage"].get("prompt_tokens", 0), r["usage"].get("completion_tokens", 0)) for r in rs
        )

    lat_hit = [r["latency"] * 1000 for r in hits]
    lat_miss = [r["latency"] * 1000 for r in misses]
    lat_all = [r["latency"] * 1000 for r in results]
    out = {
        "label": args.label,
        "requests": len(results),
        "concurrency": args.concurrency,
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(results) / wall, 1),
        "errors": len(errors),
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": round(len(hits) / len(results), 4),
        "false_hits": len(false_hits),
        "false_hit_rate": round(len(false_hits) / len(hits), 4) if hits else 0.0,
        "hit_rate_convergence_per_200": convergence,
        "latency_ms": {
            "hit": {
                "p50": round(pct(lat_hit, 50), 1),
                "p95": round(pct(lat_hit, 95), 1),
                "p99": round(pct(lat_hit, 99), 1),
            },
            "miss": {
                "p50": round(pct(lat_miss, 50), 1),
                "p95": round(pct(lat_miss, 95), 1),
                "p99": round(pct(lat_miss, 99), 1),
            },
            "all": {
                "p50": round(pct(lat_all, 50), 1),
                "p95": round(pct(lat_all, 95), 1),
                "p99": round(pct(lat_all, 99), 1),
            },
            "miss_mean": round(statistics.fmean(lat_miss), 1) if lat_miss else None,
        },
        "estimated_cost_usd": {
            model: {
                "without_cache": round(spend(results, model), 4),
                "with_cache": round(spend(misses, model), 4),
                "saved": round(spend(hits, model), 4),
                "saved_pct": round(100 * spend(hits, model) / spend(results, model), 1)
                if spend(results, model)
                else 0.0,
            }
            for model in ("gpt-4o-mini", "gpt-4o")
        },
        "proxy_stats": stats,
        "false_hit_examples": [
            {"request": r["prompt"], "served_answer_for": r["served_prompt"], "similarity": r["similarity"]}
            for r in false_hits[:15]
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("proxy_stats", "false_hit_examples")}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
