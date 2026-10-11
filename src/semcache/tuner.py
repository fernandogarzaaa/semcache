"""Threshold tuning: the hit-rate vs. wrong-answer trade-off, measured.

Three views of the same question ("what threshold should we run at?"):

* ``sweep_pairs``: labelled (similarity, same_intent) pairs, e.g. a paraphrase
  dataset. Reports, per threshold, how many true paraphrases would hit
  (recall) and what share of hits would be wrong answers (false-hit rate).
* ``sweep_history``: the live cache's own lookup log (best similarity per
  lookup). No labels, so it reports only the hit rate each threshold would have
  produced, which is what the near-miss analysis needs.
* ``replay``: a full sequential simulation of a labelled workload through an
  empty cache at each threshold. This is the number that matters in production
  because cache contents depend on the threshold.

Definitions used throughout (and in the README):

* hit rate = hits / requests
* false-hit rate = hits whose cached answer belongs to a different intent / hits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np


def default_thresholds(lo: float = 0.70, hi: float = 0.99, step: float = 0.01) -> List[float]:
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


@dataclass(frozen=True)
class PairPoint:
    threshold: float
    recall: float
    false_hit_rate: float
    false_positive_rate: float
    hits: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "threshold": self.threshold,
            "recall": round(self.recall, 4),
            "false_hit_rate": round(self.false_hit_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "hits": self.hits,
        }


def sweep_pairs(similarities: Sequence[float], labels: Sequence[bool], thresholds: Sequence[float]) -> List[PairPoint]:
    """Per threshold: recall over positive pairs, and the wrong share of everything that would hit."""
    if len(similarities) != len(labels):
        raise ValueError("similarities and labels must have the same length")
    sims = np.asarray(similarities, dtype=np.float64)
    lab = np.asarray(labels, dtype=bool)
    pos = int(lab.sum())
    neg = int((~lab).sum())
    out: List[PairPoint] = []
    for t in thresholds:
        hit = sims >= t
        hits = int(hit.sum())
        tp = int((hit & lab).sum())
        fp = hits - tp
        out.append(
            PairPoint(
                threshold=float(t),
                recall=tp / pos if pos else 0.0,
                false_hit_rate=fp / hits if hits else 0.0,
                false_positive_rate=fp / neg if neg else 0.0,
                hits=hits,
            )
        )
    return out


def sweep_history(history: Sequence[Tuple[float, bool]], thresholds: Sequence[float]) -> List[Dict[str, float]]:
    """Hit rate each threshold would have produced on the recorded lookups (unlabelled)."""
    sims = np.asarray([s for s, _ in history], dtype=np.float64)
    n = len(sims)
    return [
        {"threshold": float(t), "hit_rate": round(float((sims >= t).sum()) / n, 4) if n else 0.0, "lookups": n}
        for t in thresholds
    ]


@dataclass(frozen=True)
class ReplayResult:
    threshold: float
    requests: int
    hits: int
    false_hits: int
    hit_rate: float
    false_hit_rate: float
    correct_hit_rate: float
    hit_rate_by_window: Tuple[float, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "threshold": self.threshold,
            "requests": self.requests,
            "hits": self.hits,
            "false_hits": self.false_hits,
            "hit_rate": round(self.hit_rate, 4),
            "false_hit_rate": round(self.false_hit_rate, 4),
            "correct_hit_rate": round(self.correct_hit_rate, 4),
            "hit_rate_by_window": [round(x, 4) for x in self.hit_rate_by_window],
        }


def replay(
    vectors: np.ndarray,
    intents: Sequence[Optional[Hashable]],
    threshold: float,
    window: int = 200,
    texts: Optional[Sequence[str]] = None,
    guard: Optional[Callable[[str, str], bool]] = None,
    exact_only: bool = False,
) -> ReplayResult:
    """Run a labelled request stream through an empty single-namespace cache.

    ``vectors`` are unit-length query embeddings in arrival order. ``intents``
    label what each query asks; ``None`` marks a one-off query with no
    paraphrases, so any hit on it is a false hit. On a miss the query is stored
    with its own intent; a hit is false when the stored intent differs.

    ``guard(stored_text, text)`` returning True vetoes a hit (needs ``texts``),
    mirroring the cache's word-order guard. ``exact_only`` replaces similarity
    with exact text equality: the baseline an ordinary key-value cache gets.
    """
    if (guard is not None or exact_only) and texts is None:
        raise ValueError("guard and exact_only need texts")
    n = len(intents)
    if vectors.shape[0] != n:
        raise ValueError("vectors and intents must have the same length")
    if window <= 0:
        raise ValueError("window must be positive")
    # One matrix product up front; each step then reads a row slice of it.
    gram = vectors @ vectors.T if n and not exact_only else np.zeros((n, n), dtype=np.float32)
    stored_rows = np.zeros(max(n, 1), dtype=np.int64)
    stored_intents: List[Optional[Hashable]] = []
    stored_texts: List[str] = []
    exact: Dict[str, int] = {}
    hits = false_hits = 0
    hit_flags: List[bool] = []
    for i in range(n):
        k = len(stored_intents)
        hit = False
        j = -1
        if exact_only:
            assert texts is not None
            j = exact.get(texts[i], -1)
        elif k:
            sims = gram[i, stored_rows[:k]]
            j = int(np.argmax(sims))
            if float(sims[j]) < threshold:
                j = -1
            elif guard is not None and texts is not None and guard(stored_texts[j], texts[i]):
                j = -1
        if j >= 0:
            hit = True
            hits += 1
            if intents[i] is None or stored_intents[j] != intents[i]:
                false_hits += 1
        else:
            stored_rows[k] = i
            stored_intents.append(intents[i])
            if texts is not None:
                stored_texts.append(texts[i])
                exact.setdefault(texts[i], k)
        hit_flags.append(hit)
    windows = tuple(sum(hit_flags[s : s + window]) / len(hit_flags[s : s + window]) for s in range(0, n, window))
    return ReplayResult(
        threshold=float(threshold),
        requests=n,
        hits=hits,
        false_hits=false_hits,
        hit_rate=hits / n if n else 0.0,
        false_hit_rate=false_hits / hits if hits else 0.0,
        correct_hit_rate=(hits - false_hits) / n if n else 0.0,
        hit_rate_by_window=windows,
    )


def pick_threshold(results: Sequence[ReplayResult], max_false_hit_rate: float) -> Optional[ReplayResult]:
    """Lowest threshold (most hits) whose false-hit rate stays within budget."""
    ok = [r for r in results if r.false_hit_rate <= max_false_hit_rate]
    if not ok:
        return None
    return max(ok, key=lambda r: (r.correct_hit_rate, -r.threshold))
