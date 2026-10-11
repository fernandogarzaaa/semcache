"""Query normalisation applied before embedding.

Stripping politeness and filler ("hey, can you please tell me ...") moves
paraphrases closer together without changing what is being asked. The original
prompt text is still stored on the entry for debugging; only the embedding
input is normalised.
"""

from __future__ import annotations

import re

# Leading phrases that carry no meaning for the answer. Matched repeatedly at the start.
_LEADING_FILLERS = (
    "hey",
    "hi",
    "hello",
    "ok",
    "okay",
    "so",
    "please",
    "quick question",
    "i was wondering",
    "i wonder",
    "can you",
    "could you",
    "would you",
    "can you please",
    "could you please",
    "would you please",
    "help me",
    "tell me",
    "i want to know",
    "i'd like to know",
    "i would like to know",
    "do you know",
    "um",
    "uh",
    "hmm",
    "help:",
    "please help:",
    "question:",
)
_LEADING_RE = re.compile(
    r"^(?:"
    + "|".join(sorted((re.escape(f) for f in _LEADING_FILLERS), key=len, reverse=True))
    + r")(?:\b|(?<=:))[\s,:;!-]*"
)
_TRAILING_RE = re.compile(r"[\s,]*(?:please|thanks|thank you|thx)[\s.!?]*$")
_SPACE_RE = re.compile(r"\s+")
_PUNCT_TAIL_RE = re.compile(r"[\s?.!]+$")


def normalize_query(text: str) -> str:
    """Lower-case, collapse whitespace, strip leading filler and trailing pleasantries.

    Never returns an empty string for non-empty input: if stripping would
    remove everything, the whitespace-collapsed lower-case text is returned.
    """
    base = _SPACE_RE.sub(" ", text.strip().lower())
    out = base
    while True:
        stripped = _LEADING_RE.sub("", out, count=1)
        if stripped == out:
            break
        out = stripped
    out = _TRAILING_RE.sub("", out)
    out = _PUNCT_TAIL_RE.sub("", out).strip()
    return out or base


_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Words that carry no direction; dropped before comparing word order.
_STOPWORDS = frozenset(
    "a an the is are am be of in on at for do does did i me my you your how what what's whats which can "
    "should would could it its this that there way".split()
)


def _stem_token(word: str) -> str:
    """Tiny suffix stripper: ing/ed/s, and es only after s, x, z, ch, sh (boxes -> box, miles -> mile)."""
    for suffix in ("ing", "ed"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    if len(word) > 4 and word.endswith("es") and (word[-3] in "sxz" or word[-4:-2] in ("ch", "sh")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stemmed content words in order (stopwords removed)."""
    return [_stem_token(w) for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]


_CONNECTORS = frozenset({"to", "into", "than", "from"})
# "to" after these words is part of a verb phrase ("how to", "want to"), not a direction.
_NON_DIRECTIONAL_TO = frozenset(
    {"how", "want", "need", "like", "way", "have", "going", "able", "try", "trying", "used"}
)


def _directional(words: list[str]) -> bool:
    for i, w in enumerate(words):
        if w in _CONNECTORS and not (w == "to" and i > 0 and words[i - 1] in _NON_DIRECTIONAL_TO):
            return True
    return False


def order_swapped(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` ask about the same things in the opposite direction.

    Embeddings are close to bag-of-words for short prompts, so "convert km to
    miles" and "convert miles to km" score ~0.98 while needing opposite
    answers. Two rules, both cheap:

    * same content words, different order; or
    * both prompts are directional ("X to Y", "X into Y", "X than Y", "X vs Y")
      and the words they share appear in a different relative order.

    The cost is an occasional extra miss on a re-ordered paraphrase; a miss
    costs one upstream call, a false hit costs a wrong answer.
    """
    ta, tb = content_tokens(a), content_tokens(b)
    if len(ta) > 1 and ta != tb and sorted(ta) == sorted(tb):
        return True
    wa = _TOKEN_RE.findall(a.lower())
    wb = _TOKEN_RE.findall(b.lower())
    if not (_directional(wa) and _directional(wb)):
        return False
    shared = (set(ta) & set(tb)) - _CONNECTORS
    sa = [t for t in dict.fromkeys(ta) if t in shared]
    sb = [t for t in dict.fromkeys(tb) if t in shared]
    return len(shared) >= 2 and sa != sb
