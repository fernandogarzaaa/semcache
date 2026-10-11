"""Price table for the "dollars saved" estimate.

Prices are USD per 1M tokens (input, output). The defaults are public list
prices at the time of writing and are only an *estimate*: override them with
``SEMCACHE_PRICES='{"my-model": [0.5, 1.5]}'``. Unknown models cost 0, so the
savings figure under-reports rather than invents.
"""

from __future__ import annotations

import json
from typing import Dict, Mapping, Optional, Tuple

DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
}


class PriceTable:
    def __init__(self, prices: Optional[Mapping[str, Tuple[float, float]]] = None) -> None:
        self.prices: Dict[str, Tuple[float, float]] = dict(DEFAULT_PRICES if prices is None else prices)

    @classmethod
    def from_json(cls, raw: str, base: Optional[Mapping[str, Tuple[float, float]]] = None) -> PriceTable:
        """Merge a JSON object ``{"model": [in, out]}`` over ``base`` (default list prices)."""
        table = dict(DEFAULT_PRICES if base is None else base)
        if raw.strip():
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("SEMCACHE_PRICES must be a JSON object")
            for model, pair in data.items():
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"price for {model!r} must be [input_per_1m, output_per_1m]")
                table[str(model)] = (float(pair[0]), float(pair[1]))
        return cls(table)

    def lookup(self, model: str) -> Tuple[float, float]:
        """Exact match first, then the longest known prefix (``gpt-4o-mini-2024-07-18`` -> ``gpt-4o-mini``)."""
        if model in self.prices:
            return self.prices[model]
        best = ""
        for name in self.prices:
            if model.startswith(name) and len(name) > len(best):
                best = name
        return self.prices[best] if best else (0.0, 0.0)

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        p_in, p_out = self.lookup(model)
        return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000
