"""Cache policy: how long an answer stays fresh and how close a match must be.

Three small, inspectable rule sets instead of a model:

* **Freshness tiers** (``classify_freshness``): prompts about the present
  ("today", "latest", "current price") get a short TTL; prompts that ask for
  live data ("right now", "live score", "breaking") are not cached at all;
  definitional prompts ("what is", "explain", "history of") get a long TTL.
* **Task types** (``classify_task``): classification/extraction prompts have a
  constrained answer space and tolerate a looser match; creative prompts need a
  near-exact match, and at high temperature are not cached.
* **Adaptive thresholds** (``AdaptiveThresholds``): per task type, moved by
  feedback. A reported wrong hit raises the threshold above that similarity; a
  confirmed good hit lowers it a little, never below the floor or the worst
  wrong hit seen.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

TIER_NO_CACHE = "no_cache"
TIER_SHORT = "short"
TIER_DEFAULT = "default"
TIER_LONG = "long"

TASK_CLASSIFICATION = "classification"
TASK_CREATIVE = "creative"
TASK_GENERAL = "general"
TASK_TYPES = (TASK_CLASSIFICATION, TASK_CREATIVE, TASK_GENERAL)

_LIVE_RE = re.compile(
    r"\b(right now|live (?:score|scores|feed|price|prices|update|updates)|breaking|real[- ]time|at this moment|"
    r"as of now|this minute|currently trading)\b"
)
_TIME_RE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|now|current(?:ly)?|latest|recent(?:ly)?|this (?:week|month|year|morning|"
    r"evening)|news|weather|forecast|stock|price|prices|exchange rate|score|scores|trending|20[2-9][0-9])\b"
)
_STABLE_RE = re.compile(
    r"^(what is|what are|what's|define|definition of|explain|how does|how do|how to|why does|why do|"
    r"difference between|history of|meaning of|who (?:was|invented|wrote|discovered))\b"
)
_CLASSIFY_RE = re.compile(
    r"\b(classify|categori[sz]e|label|sentiment|is (?:this|it) (?:spam|positive|negative)|yes or no|true or false|"
    r"which category|extract|answer with one word|choose one|multiple choice)\b"
)
_CREATIVE_RE = re.compile(
    r"\b(write (?:a|an|me)? ?(?:poem|story|song|haiku|limerick|joke|essay|tweet|slogan|novel)|poem|haiku|story|"
    r"brainstorm|imagine|creative|lyrics|fiction|slogan|tagline)\b"
)


def classify_freshness(query: str) -> str:
    """Freshness tier of a (normalised, lower-case) query."""
    q = query.lower()
    if _LIVE_RE.search(q):
        return TIER_NO_CACHE
    if _TIME_RE.search(q):
        return TIER_SHORT
    if _STABLE_RE.search(q):
        return TIER_LONG
    return TIER_DEFAULT


def classify_task(query: str, system: str = "") -> str:
    """Task type from the query and the system prompt."""
    text = f"{system}\n{query}".lower()
    if _CLASSIFY_RE.search(text):
        return TASK_CLASSIFICATION
    if _CREATIVE_RE.search(text):
        return TASK_CREATIVE
    return TASK_GENERAL


@dataclass(frozen=True)
class TTLPolicy:
    """Seconds to live per freshness tier. ``0`` means do not cache."""

    short: float = 3600.0
    default: float = 6 * 3600.0
    long: float = 7 * 24 * 3600.0

    def ttl_for(self, tier: str) -> float:
        if tier == TIER_NO_CACHE:
            return 0.0
        if tier == TIER_SHORT:
            return self.short
        if tier == TIER_LONG:
            return self.long
        return self.default


@dataclass
class _TaskState:
    threshold: float
    worst_false_hit: float = -1.0
    good: int = 0
    bad: int = 0


@dataclass
class AdaptiveThresholds:
    """Per-task-type similarity thresholds that learn from hit feedback."""

    base: Mapping[str, float] = field(
        default_factory=lambda: {TASK_CLASSIFICATION: 0.88, TASK_GENERAL: 0.92, TASK_CREATIVE: 0.97}
    )
    floor: float = 0.70
    ceiling: float = 0.995
    margin: float = 0.01
    step: float = 0.002
    adaptive: bool = True
    _state: Dict[str, _TaskState] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.floor <= self.ceiling <= 1.0:
            raise ValueError("need 0 < floor <= ceiling <= 1")
        for task, value in self.base.items():
            self._state[task] = _TaskState(threshold=self._clamp(value))

    def _clamp(self, value: float) -> float:
        return min(self.ceiling, max(self.floor, value))

    def _get(self, task: str) -> _TaskState:
        st = self._state.get(task)
        if st is None:
            st = _TaskState(threshold=self._clamp(self.base.get(TASK_GENERAL, 0.92)))
            self._state[task] = st
        return st

    def threshold(self, task: str) -> float:
        with self._lock:
            return self._get(task).threshold

    def feedback(self, task: str, similarity: float, correct: bool) -> float:
        """Record whether a hit at ``similarity`` was a correct answer; return the new threshold."""
        with self._lock:
            st = self._get(task)
            if correct:
                st.good += 1
                if self.adaptive:
                    lowered = st.threshold - self.step
                    if st.worst_false_hit >= 0:
                        lowered = max(lowered, st.worst_false_hit + self.margin)
                    st.threshold = self._clamp(min(st.threshold, lowered))
            else:
                st.bad += 1
                st.worst_false_hit = max(st.worst_false_hit, similarity)
                if self.adaptive and similarity + self.margin > st.threshold:
                    st.threshold = self._clamp(similarity + self.margin)
            return st.threshold

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                task: {
                    "threshold": round(st.threshold, 4),
                    "worst_false_hit": round(st.worst_false_hit, 4),
                    "good_feedback": float(st.good),
                    "bad_feedback": float(st.bad),
                }
                for task, st in sorted(self._state.items())
            }


def creative_bypass(task: str, temperature: Optional[float], limit: float = 0.8) -> bool:
    """Creative prompts sampled at high temperature want a fresh answer every time."""
    return task == TASK_CREATIVE and temperature is not None and float(temperature) >= limit
