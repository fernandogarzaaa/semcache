"""Tests written from testteeth's surviving-mutant report: defaults, rounding, boundaries."""

import json
import sys
import types

import numpy as np
import pytest

from semcache import embedders as emb_mod
from semcache.cache import CacheEntry, Lookup, NearMiss, SemanticCache
from semcache.embedders import HashingEmbedder, make_embedder
from semcache.index import make_index
from semcache.keys import build_key, has_non_text_content, is_cacheable
from semcache.normalize import _directional, _stem_token, order_swapped
from semcache.policy import AdaptiveThresholds
from semcache.pricing import DEFAULT_PRICES, PriceTable
from semcache.streaming import StreamAccumulator, completion_to_sse, sse
from semcache.tuner import PairPoint, ReplayResult, default_thresholds, replay, sweep_history


def key(q, model="m"):
    return build_key({"model": model, "messages": [{"role": "user", "content": q}]})


def resp():
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]}


# ---- keys ----------------------------------------------------------------------------------------


def test_keys_predicates_return_real_booleans():
    assert has_non_text_content([{"content": "x"}]) is False
    assert has_non_text_content([{"content": [{"type": "image_url"}]}]) is True
    assert is_cacheable({"model": "m", "n": 2}) is False
    assert is_cacheable({"model": "m"}) is True


# ---- normalize -----------------------------------------------------------------------------------


def test_stem_length_guards():
    assert _stem_token("boxed") == "box"
    assert _stem_token("string") == "str"
    assert _stem_token("cats") == "cat"
    assert _stem_token("its") == "its"


def test_directional_first_word_does_not_look_at_last_word():
    assert _directional(["to", "b", "how"]) is True
    assert _directional(["a", "b"]) is False


def test_two_shared_words_in_swapped_order_is_enough():
    assert order_swapped("convert x to y", "y from x") is True
    assert order_swapped("convert x to y", "x from y") is False


# ---- embedders -----------------------------------------------------------------------------------


def test_hashing_defaults_and_minimum_dim():
    e = HashingEmbedder()
    assert e.dim == 256 and e.name == "hashing-256"
    one = HashingEmbedder(dim=1)
    v = one.embed(["abc"])
    assert v.shape == (1, 1) and abs(float(v[0, 0])) == pytest.approx(1.0)


class _FakeST:
    instances: list = []

    def __init__(self, name, device="cpu"):
        self.name, self.device = name, device
        self.calls = []
        _FakeST.instances.append(self)

    def get_embedding_dimension(self):
        return 3

    def get_sentence_embedding_dimension(self):  # pragma: no cover - the newer name wins
        raise AssertionError("old API used")

    def encode(self, texts, normalize_embeddings, convert_to_numpy, batch_size):
        self.calls.append((list(texts), normalize_embeddings, convert_to_numpy, batch_size))
        return np.ones((len(texts), 3), dtype=np.float64)


class _OldFakeST(_FakeST):
    get_embedding_dimension = None

    def get_sentence_embedding_dimension(self):
        return None


@pytest.fixture
def fake_st(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    _FakeST.instances.clear()
    return mod


def test_sentence_transformer_embedder_with_fake_model(fake_st):
    e = emb_mod.SentenceTransformerEmbedder()
    m = _FakeST.instances[-1]
    assert m.name == "sentence-transformers/all-MiniLM-L6-v2" and m.device == "cpu"
    assert e.dim == 3 and e.name == "all-MiniLM-L6-v2"
    out = e.embed(("a", "b"))
    assert out.dtype == np.float32 and out.shape == (2, 3)
    assert m.calls[-1] == (["a", "b"], True, True, 32)


def test_sentence_transformer_dim_fallback(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _OldFakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    e = emb_mod.SentenceTransformerEmbedder("org/custom", device="cpu")
    assert e.dim == 3 and e.name == "custom"


def test_make_embedder_minilm_aliases(fake_st):
    for alias in ("minilm", "sentence-transformers", "st", "MiniLM"):
        assert make_embedder(alias).dim == 3
    make_embedder("minilm", model="org/other")
    assert _FakeST.instances[-1].name == "org/other"


def test_missing_sentence_transformers_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="pip install 'semcache\\[local\\]'"):
        emb_mod.SentenceTransformerEmbedder()


# ---- index ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["numpy", "faiss"])
def test_index_default_k_id_zero_and_cleanup(kind):
    if kind == "faiss":
        pytest.importorskip("faiss")
    idx = make_index(kind, 2)
    idx.add("ns", 0, np.array([1.0, 0.0]))
    idx.add("ns", 5, np.array([0.0, 1.0]))
    res = idx.search("ns", np.array([1.0, 0.0]))
    assert len(res) == 1 and res[0][0] == 0
    assert idx.remove("ns", 0) is True
    assert idx.remove("ns", 5) is True
    assert idx._parts == {}
    assert len(idx) == 0


# ---- pricing -------------------------------------------------------------------------------------


def test_default_price_table_values():
    assert DEFAULT_PRICES == {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1": (2.00, 8.00),
        "claude-3-5-haiku": (0.80, 4.00),
        "claude-3-5-sonnet": (3.00, 15.00),
    }


def test_cost_is_exact_per_million():
    p = PriceTable()
    assert p.cost("gpt-4o", 1_000_000, 0) == 2.5
    assert p.cost("gpt-4o", 0, 1_000_000) == 10.0
    assert p.cost("gpt-4.1-mini-2025", 1_000_000, 1_000_000) == 2.0


# ---- policy --------------------------------------------------------------------------------------


def test_policy_defaults():
    t = AdaptiveThresholds()
    assert t.margin == 0.01 and t.step == 0.002 and t.floor == 0.70 and t.ceiling == 0.995 and t.adaptive


def test_worst_false_hit_of_zero_still_bounds_lowering():
    t = AdaptiveThresholds(base={"general": 0.75}, margin=0.8, floor=0.7, ceiling=0.99)
    assert t.feedback("general", 0.0, correct=False) == pytest.approx(0.8)
    assert t.feedback("general", 0.9, correct=True) == pytest.approx(0.8)


def test_snapshot_rounds_to_four_places():
    t = AdaptiveThresholds(base={"general": 0.9})
    t.feedback("general", 0.933333333, correct=False)
    snap = t.snapshot()["general"]
    assert snap["threshold"] == 0.9433 and snap["worst_false_hit"] == 0.9333


def test_instances_do_not_share_state():
    a, b = AdaptiveThresholds(), AdaptiveThresholds()
    a.feedback("general", 0.99, correct=False)
    assert b.threshold("general") == 0.92


# ---- cache ---------------------------------------------------------------------------------------


def test_dataclass_defaults():
    e = CacheEntry(1, "n", "m", "s", "p", "p", {}, 0.0, 1.0)
    assert (e.tier, e.task, e.tags, e.hit_count, e.last_hit_at, e.prompt_tokens, e.completion_tokens) == (
        "default",
        "general",
        (),
        0,
        0.0,
        0,
        0,
    )
    lk = Lookup(hit=False, similarity=0.1, threshold=0.9)
    assert lk.near_miss is False and lk.entry is None and lk.nearest_prompt is None


def test_near_miss_rounding():
    d = NearMiss(1.0, "m", "q", "p", 0.123456, 0.987654).as_dict()
    assert d["similarity"] == 0.1235 and d["threshold"] == 0.9877
    assert NearMiss(1.0, "m", "q", "p", 0.123456, 0.9).as_dict()["gap"] == 0.7765


def test_cache_constructor_defaults():
    e = HashingEmbedder()
    c = SemanticCache(e, make_index("numpy", e.dim))
    assert c.max_entries == 10_000 and c.search_k == 4 and c.near_miss_window == 0.05
    assert c.normalize and c.order_guard
    assert c._near.maxlen == 500 and c._history.maxlen == 20_000
    one = SemanticCache(e, make_index("numpy", e.dim), max_entries=1)
    one.store(key("aaa bbb"), resp(), ttl=10)
    one.store(key("ccc ddd"), resp(), ttl=10)
    assert len(one) == 1


def test_counters_after_empty_lookup_and_fractional_ttl():
    e = HashingEmbedder()
    c = SemanticCache(e, make_index("numpy", e.dim))
    c.lookup(key("x"), 0.9)
    assert c.counters.misses == 1 and c.counters.hits == 0 and c.counters.lookups == 1
    assert c.store(key("x"), resp(), ttl=0.5) is not None


def test_hit_rate_rounding():
    e = HashingEmbedder()
    c = SemanticCache(e, make_index("numpy", e.dim))
    c.store(key("alpha beta"), resp(), ttl=60)
    c.lookup(key("alpha beta"), 0.9)
    c.lookup(key("zzz"), 0.9)
    c.lookup(key("yyy"), 0.9)
    assert c.stats()["hit_rate"] == 0.3333


def test_near_misses_default_limit_and_full_order():
    e = HashingEmbedder()
    c = SemanticCache(e, make_index("numpy", e.dim), near_miss_window=2.0)
    c.store(key("alpha beta gamma"), resp(), ttl=60)
    for i in range(55):
        c.lookup(key(f"alpha beta q{i}"), 0.9999)
    nm = c.near_misses()
    assert len(nm) == 50
    assert [n.query for n in nm[:3]] == ["alpha beta q54", "alpha beta q53", "alpha beta q52"]


def test_guard_applies_at_exact_threshold():
    e = HashingEmbedder()
    c = SemanticCache(e, make_index("numpy", e.dim))
    c.store(key("convert km to miles"), resp(), ttl=60)
    sim = c.lookup(key("convert miles to km"), 2.0).similarity
    assert not c.lookup(key("convert miles to km"), sim).hit
    assert c.counters.guard_rejections == 1


def test_invalidate_short_dated_suffix():
    e = HashingEmbedder()
    c = SemanticCache(e, make_index("numpy", e.dim))
    c.store(key("q", model="gpt-4o-1"), resp(), ttl=60)
    assert c.invalidate(model="gpt-4o") == 1


# ---- tuner ---------------------------------------------------------------------------------------


def test_threshold_and_result_rounding():
    assert default_thresholds(0.0, 1.0, 1 / 3) == [0.0, 0.3333, 0.6667, 1.0]
    p = PairPoint(0.9, 1 / 3, 2 / 3, 1 / 7, 3).as_dict()
    assert p == {"threshold": 0.9, "recall": 0.3333, "false_hit_rate": 0.6667, "false_positive_rate": 0.1429, "hits": 3}
    r = ReplayResult(0.9, 3, 1, 1, 1 / 3, 2 / 3, 1 / 7, ()).as_dict()
    assert (r["hit_rate"], r["false_hit_rate"], r["correct_hit_rate"]) == (0.3333, 0.6667, 0.1429)
    assert sweep_history([(0.95, True), (0.5, False), (0.5, False)], [0.9])[0]["hit_rate"] == 0.3333


def test_replay_default_window_is_200():
    v = np.ones((201, 1), dtype=np.float32)
    assert len(replay(v, ["a"] * 201, 0.5).hit_rate_by_window) == 2


# ---- streaming -----------------------------------------------------------------------------------


def test_data_line_without_space():
    acc = StreamAccumulator()
    acc.feed(b'data:{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n')
    acc.feed(b"data:[DONE]\n")
    assert acc.complete and acc.to_completion()["choices"][0]["message"]["content"] == "hi"


def test_unfinished_choice_blocks_caching():
    acc = StreamAccumulator()
    acc.feed(sse({"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": "stop"}]}))
    acc.feed(sse({"choices": [{"index": 1, "delta": {"content": "b"}}]}))
    acc.feed(b"data: [DONE]\n\n")
    assert acc.done and not acc.complete


def test_completion_to_sse_defaults_missing_index_to_zero():
    chunks = list(completion_to_sse({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}))
    first = json.loads(chunks[0][6:])
    second = json.loads(chunks[1][6:])
    assert first["choices"][0]["index"] == 0 and second["choices"][0]["index"] == 0
