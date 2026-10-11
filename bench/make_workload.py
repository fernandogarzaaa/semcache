"""Build the load-test workload and the labelled pair set.

Outputs (git-ignored; regenerate with this script, fixed seed):

* ``bench/data/workload.jsonl``: N requests in arrival order, ``{"prompt", "intent"}``.
  ~75% are paraphrases of the hand-written intents in ``bench/intents.py`` with a
  Zipf popularity (a few questions are asked a lot, most rarely); ~25% are
  one-off questions (``intent: null``) drawn from the GLUE QQP validation split,
  so they have no paraphrase in the stream and *any* hit on them is a false hit.
* ``bench/data/qqp_pairs.jsonl``: a random sample of labelled QQP validation
  pairs ``{"q1", "q2", "duplicate"}`` for the pair-level threshold sweep.

QQP text is downloaded from Hugging Face at run time and not committed.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intents import INTENTS  # noqa: E402

# Keep the published example output workplace-safe; these rows are simply skipped.
BLOCKLIST = (
    "sex",
    "porn",
    "nude",
    "naked",
    "fuck",
    "shit",
    "dick",
    "boob",
    "penis",
    "vagina",
    "kill",
    "suicide",
    "drug",
)

QQP_URL = "https://huggingface.co/api/datasets/nyu-mll/glue/parquet/qqp/validation"


def load_qqp() -> list[dict]:
    import pyarrow.parquet as pq

    with urllib.request.urlopen(QQP_URL, timeout=60) as r:
        files = json.loads(r.read())
    rows: list[dict] = []
    for url in files:
        with urllib.request.urlopen(url, timeout=120) as r:
            table = pq.read_table(io.BytesIO(r.read()))
        rows.extend(table.to_pylist())
    return rows


PREFIXES = ("hey, ", "hi! ", "quick question: ", "can you tell me ", "ok so ", "please help: ", "um ")
SUFFIXES = (" thanks", " thx", " pls", "??", " :)", "", " asap")


def surface_variant(prompt: str, rng: random.Random) -> str:
    """The same question typed differently: casing, punctuation, filler. Same intent by construction."""
    p = prompt
    roll = rng.random()
    if roll < 0.35:
        p = p.lower().rstrip("?")
    elif roll < 0.5:
        p = p.upper() if len(p) < 40 else p.lower()
    if rng.random() < 0.5:
        p = rng.choice(PREFIXES) + p[0].lower() + p[1:]
    if rng.random() < 0.5:
        p = p + rng.choice(SUFFIXES)
    return p if p != prompt else p.lower() + " pls"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2400)
    ap.add_argument("--unique-share", type=float, default=0.25)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--surface-noise", type=float, default=0.3)
    ap.add_argument("--out", default="bench/data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qqp = load_qqp()

    intent_ids = list(INTENTS)
    rng.shuffle(intent_ids)  # popularity rank is random, not file order
    weights = [1.0 / (rank + 1) ** args.zipf for rank in range(len(intent_ids))]

    all_paraphrases = {p.strip().lower() for v in INTENTS.values() for p in v}
    seen: set[str] = set()
    uniques: list[str] = []
    for row in rng.sample(qqp, len(qqp)):
        q = str(row["question1"]).strip()
        key = q.lower()
        if any(w in key for w in BLOCKLIST):
            continue
        if 15 <= len(q) <= 160 and key not in seen and key not in all_paraphrases:
            seen.add(key)
            uniques.append(q)

    n_unique = int(round(args.requests * args.unique_share))
    stream: list[dict] = [{"prompt": q, "intent": None} for q in uniques[:n_unique]]
    for _ in range(args.requests - n_unique):
        intent = rng.choices(intent_ids, weights=weights, k=1)[0]
        prompt = rng.choice(INTENTS[intent])
        if rng.random() < args.surface_noise:
            prompt = surface_variant(prompt, rng)
        stream.append({"prompt": prompt, "intent": intent})
    rng.shuffle(stream)

    with open(out / "workload.jsonl", "w", encoding="utf-8") as fh:
        for row in stream:
            fh.write(json.dumps(row) + "\n")

    clean = [r for r in qqp if not any(w in (r["question1"] + r["question2"]).lower() for w in BLOCKLIST)]
    pairs = rng.sample(clean, min(args.pairs, len(clean)))
    with open(out / "qqp_pairs.jsonl", "w", encoding="utf-8") as fh:
        for row in pairs:
            fh.write(
                json.dumps({"q1": row["question1"], "q2": row["question2"], "duplicate": bool(row["label"])}) + "\n"
            )

    distinct_intents = len({r["intent"] for r in stream if r["intent"]})
    print(
        json.dumps(
            {
                "requests": len(stream),
                "unique_one_off": n_unique,
                "intent_requests": len(stream) - n_unique,
                "intents_seen": distinct_intents,
                "intents_defined": len(INTENTS),
                "paraphrases_defined": sum(len(v) for v in INTENTS.values()),
                "qqp_pairs": len(pairs),
                "qqp_duplicate_share": round(sum(p["label"] for p in pairs) / len(pairs), 4),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
