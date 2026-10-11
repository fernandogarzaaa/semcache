import math

import pytest

from semcache.cache import DUPLICATE_SIMILARITY, CacheEntry, SemanticCache
from semcache.embedders import HashingEmbedder
from semcache.index import NumpyIndex
from semcache.keys import build_key


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def key(q, model="m", system=None, **kw):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return build_key({"model": model, "messages": msgs, **kw})


def resp(text="answer", pt=10, ct=20):
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
    }


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def cache(clock):
    emb = HashingEmbedder()
    return SemanticCache(emb, NumpyIndex(emb.dim), max_entries=100, clock=clock)


def test_miss_on_empty_then_hit_exact(cache):
    k = key("how do I reverse a list in python")
    lk = cache.lookup(k, 0.85)
    assert not lk.hit and lk.similarity == -1.0 and lk.entry is None and lk.threshold == 0.85
    e = cache.store(k, resp(), ttl=60)
    assert e is not None and e.id == 1
    lk = cache.lookup(k, 0.85)
    assert lk.hit and lk.entry is e and lk.similarity == pytest.approx(1.0, abs=1e-5)
    assert lk.nearest_prompt == "how do I reverse a list in python"
    assert e.hit_count == 1


def test_paraphrase_hits_unrelated_misses(cache):
    cache.store(key("How do I reverse a list in Python?"), resp(), ttl=60)
    assert cache.lookup(key("hey, how do i reverse a list in python please"), 0.85).hit
    lk = cache.lookup(key("best pizza in naples"), 0.85)
    assert not lk.hit and lk.similarity < 0.5 and lk.nearest_prompt is not None


def test_threshold_boundary_is_inclusive(cache):
    k = key("alpha beta gamma")
    cache.store(k, resp(), ttl=60)
    sim = cache.lookup(key("alpha beta delta"), 0.0).similarity
    assert cache.lookup(key("alpha beta delta"), sim).hit
    assert not cache.lookup(key("alpha beta delta"), sim + 1e-4).hit


def test_namespace_isolation(cache):
    cache.store(key("what is python", system="be brief"), resp(), ttl=60)
    assert not cache.lookup(key("what is python", system="be verbose"), 0.5).hit
    assert not cache.lookup(key("what is python", model="other"), 0.5).hit
    assert not cache.lookup(key("what is python", system="be brief", temperature=1.0), 0.5).hit
    assert cache.lookup(key("what is python", system="be brief"), 0.5).hit


def test_entry_metadata(cache, clock):
    k = key("Hey, what is Python?", system="S")
    e = cache.store(k, resp(pt=7, ct=9), ttl=30, tier="long", task="general", tags=["b", "a", "a"])
    assert e.prompt == "Hey, what is Python?"
    assert e.normalized == "what is python"
    assert e.created_at == 1000.0 and e.expires_at == 1030.0
    assert e.tier == "long" and e.task == "general"
    assert e.tags == ("a", "b")
    assert e.prompt_tokens == 7 and e.completion_tokens == 9
    assert e.model == "m" and e.system_hash == k.system_hash and e.namespace == k.namespace
    pub = e.public()
    assert "response" not in pub and pub["tags"] == ["a", "b"] and pub["expires_at"] == 1030.0
    assert pub["hit_count"] == 0 and pub["prompt_tokens"] == 7 and pub["completion_tokens"] == 9
    assert pub["id"] == e.id and pub["created_at"] == 1000.0 and pub["tier"] == "long"


def test_public_infinite_ttl():
    e = CacheEntry(1, "n", "m", "s", "p", "p", {}, 0.0, math.inf)
    assert e.public()["expires_at"] is None
    assert not e.expired(1e18)


def test_missing_usage_defaults_to_zero(cache):
    e = cache.store(key("q"), {"choices": []}, ttl=60)
    assert e.prompt_tokens == 0 and e.completion_tokens == 0
    e2 = cache.store(key("q2 other"), {"choices": [], "usage": None}, ttl=60)
    assert e2.prompt_tokens == 0


def test_zero_or_negative_ttl_stores_nothing(cache):
    assert cache.store(key("q"), resp(), ttl=0) is None
    assert cache.store(key("q"), resp(), ttl=-5) is None
    assert len(cache) == 0
    assert cache.counters.stores == 0


def test_ttl_expiry_is_lazy_and_counted(cache, clock):
    evicted = []
    cache.on_evict = lambda reason, n: evicted.append((reason, n))
    k = key("what is a monad")
    cache.store(k, resp(), ttl=10)
    clock.t += 9.999
    assert cache.lookup(k, 0.9).hit
    clock.t += 0.001  # now == expires_at -> expired
    lk = cache.lookup(k, 0.9)
    assert not lk.hit and lk.similarity == -1.0
    assert len(cache) == 0
    assert cache.counters.evictions["expired"] == 1
    assert evicted == [("expired", 1)]


def test_expired_neighbour_skipped_for_live_one(cache, clock):
    cache.store(key("alpha beta gamma"), resp("old"), ttl=5)
    cache.store(key("alpha beta gamma delta"), resp("new"), ttl=100)
    clock.t += 6
    lk = cache.lookup(key("alpha beta gamma"), 0.5)
    assert lk.hit and lk.entry.response["choices"][0]["message"]["content"] == "new"


def test_purge_expired(cache, clock):
    cache.store(key("one thing"), resp(), ttl=5)
    cache.store(key("another topic entirely"), resp(), ttl=50)
    clock.t += 10
    assert cache.purge_expired() == 1
    assert len(cache) == 1
    assert cache.purge_expired() == 0


def test_lru_capacity_eviction(clock):
    emb = HashingEmbedder()
    c = SemanticCache(emb, NumpyIndex(emb.dim), max_entries=2, clock=clock)
    a, b, d = key("apples and oranges"), key("quantum chromodynamics"), key("medieval castles")
    c.store(a, resp("A"), ttl=60)
    c.store(b, resp("B"), ttl=60)
    assert c.lookup(a, 0.9).hit  # a becomes most recently used
    c.store(d, resp("D"), ttl=60)
    assert len(c) == 2
    assert c.lookup(a, 0.9).hit
    assert not c.lookup(b, 0.9).hit
    assert c.lookup(d, 0.9).hit
    assert c.counters.evictions["capacity"] == 1
    assert len(c.index) == 2


def test_duplicate_prompt_replaces_entry(cache):
    k = key("what is rust")
    first = cache.store(k, resp("v1"), ttl=60)
    second = cache.store(k, resp("v2"), ttl=60)
    assert len(cache) == 1
    assert cache.get(first.id) is None and cache.get(second.id) is second
    assert cache.lookup(k, 0.9).entry.response["choices"][0]["message"]["content"] == "v2"
    assert cache.counters.evictions == {"capacity": 0, "expired": 0, "invalidated": 0}
    assert DUPLICATE_SIMILARITY > 0.999


def test_paraphrase_store_keeps_both(cache):
    cache.store(key("how do I reverse a list in python"), resp(), ttl=60)
    cache.store(key("how to reverse a python list"), resp(), ttl=60)
    assert len(cache) == 2


def test_near_miss_recorded(cache):
    cache.store(key("alpha beta gamma"), resp(), ttl=60)
    sim = cache.lookup(key("alpha beta delta"), 0.0).similarity
    lk = cache.lookup(key("alpha beta delta"), sim + 0.01)
    assert not lk.hit and lk.near_miss
    nm = cache.near_misses()
    assert len(nm) == 1
    d = nm[0].as_dict()
    assert d["query"] == "alpha beta delta" and d["nearest_prompt"] == "alpha beta gamma"
    assert d["gap"] == pytest.approx(0.01, abs=1e-4)
    assert d["model"] == "m" and d["at"] == 1000.0
    far = cache.lookup(key("alpha beta delta"), sim + 0.2)
    assert not far.near_miss
    assert cache.counters.near_misses == 1
    # boundary: exactly window below threshold still counts
    edge = cache.lookup(key("alpha beta delta"), sim + cache.near_miss_window)
    assert edge.near_miss


def test_near_misses_newest_first_and_limited(cache):
    cache.near_miss_window = 1.0
    cache.store(key("alpha beta gamma"), resp(), ttl=60)
    for q in ("alpha beta delta", "alpha beta epsilon", "alpha beta zeta"):
        cache.lookup(key(q), 0.999)
    sims = [n.query for n in cache.near_misses()]
    assert sims[0] == "alpha beta zeta"
    assert len(cache.near_misses(limit=2)) == 2


def test_history_records_best_similarity(cache):
    cache.lookup(key("nothing stored"), 0.9)
    assert cache.history() == []
    cache.store(key("alpha beta gamma"), resp(), ttl=60)
    cache.lookup(key("alpha beta gamma"), 0.9)
    cache.lookup(key("zzz yyy"), 0.9)
    h = cache.history()
    assert h[0][1] is True and h[0][0] == pytest.approx(1.0, abs=1e-5)
    assert h[1][1] is False


def test_invalidate_filters(cache):
    cache.store(key("what is python", model="gpt-4o", system="A"), resp(), ttl=60, tags=["docs"])
    cache.store(key("what is python", model="gpt-4o-2024-08-06", system="A"), resp(), ttl=60)
    cache.store(key("what is java", model="gpt-4o-mini", system="B"), resp(), ttl=60, tags=["docs"])
    cache.store(key("explain go", model="claude", system="A"), resp(), ttl=60)
    with pytest.raises(ValueError):
        cache.invalidate()
    assert cache.invalidate(model="gpt-4o") == 2  # versioned variant too, but not gpt-4o-mini
    assert len(cache) == 2
    assert cache.invalidate(tag="docs") == 1
    assert cache.invalidate(prefix="explain") == 1
    assert len(cache) == 0
    assert cache.counters.evictions["invalidated"] == 4


def test_invalidate_system_hash_and_combined_filters(cache):
    k1 = key("q one", model="m1", system="A")
    k2 = key("q two", model="m2", system="A")
    cache.store(k1, resp(), ttl=60)
    cache.store(k2, resp(), ttl=60)
    assert cache.invalidate(system_hash=k1.system_hash, model="m2") == 1
    assert cache.lookup(k1, 0.9).hit
    assert cache.invalidate(namespace=k1.namespace) == 1
    assert cache.invalidate(system_hash="nope") == 0


def test_invalidate_model_does_not_match_plain_prefix(cache):
    cache.store(key("q", model="gpt-4o2"), resp(), ttl=60)
    cache.store(key("q", model="gpt-4o-"), resp(), ttl=60)
    cache.store(key("q", model="gpt-4o-x1"), resp(), ttl=60)
    cache.store(key("q", model="xgpt-4o-1"), resp(), ttl=60)
    assert cache.invalidate(model="gpt-4o") == 0
    cache.store(key("q", model="gpt-4o-0613"), resp(), ttl=60)
    assert cache.invalidate(model="gpt-4o") == 1


def test_clear(cache):
    cache.store(key("one"), resp(), ttl=60)
    cache.store(key("two other"), resp(), ttl=60)
    assert cache.clear() == 2
    assert len(cache) == 0 and len(cache.index) == 0


def test_stats(cache):
    cache.store(key("alpha beta"), resp(), ttl=60)
    cache.lookup(key("alpha beta"), 0.9)
    cache.lookup(key("zzz"), 0.9)
    s = cache.stats()
    assert s["entries"] == 1 and s["max_entries"] == 100
    assert s["lookups"] == 2 and s["hits"] == 1 and s["misses"] == 1 and s["hit_rate"] == 0.5
    assert s["stores"] == 1 and s["embedder"] == "hashing-256" and s["index"] == "numpy"


def test_stats_empty_hit_rate(cache):
    assert cache.stats()["hit_rate"] == 0.0


def test_entries_and_get(cache):
    e = cache.store(key("x y"), resp(), ttl=60)
    assert cache.entries() == [e]
    assert cache.get(999) is None


def test_lookup_with_precomputed_vector(cache):
    k = key("what is python")
    v = cache.embed(k.query)
    cache.store(k, resp(), ttl=60, vector=v)
    assert cache.lookup(k, 0.99, vector=v).hit


def test_normalize_flag(clock):
    emb = HashingEmbedder()
    off = SemanticCache(emb, NumpyIndex(emb.dim), normalize=False, clock=clock)
    assert off.embedding_text("Hey, What?") == "Hey, What?"
    on = SemanticCache(emb, NumpyIndex(emb.dim), clock=clock)
    assert on.embedding_text("Hey, What is X?") == "what is x"


def test_constructor_validation():
    emb = HashingEmbedder()
    with pytest.raises(ValueError):
        SemanticCache(emb, NumpyIndex(emb.dim), max_entries=0)
    c = SemanticCache(emb, NumpyIndex(emb.dim), search_k=0)
    assert c.search_k == 1


def test_order_guard_blocks_direction_swap(cache):
    cache.store(key("convert km to miles"), resp("km->mi"), ttl=60)
    lk = cache.lookup(key("convert miles to km"), 0.5)
    assert not lk.hit
    assert cache.counters.guard_rejections == 1
    assert cache.stats()["guard_rejections"] == 1
    assert cache.lookup(key("please convert km to miles"), 0.5).hit


def test_order_guard_only_checks_candidates_above_threshold(cache):
    cache.store(key("convert km to miles"), resp(), ttl=60)
    lk = cache.lookup(key("convert miles to km"), 1.01)
    assert not lk.hit and cache.counters.guard_rejections == 0 and lk.similarity > 0.9


def test_order_guard_can_be_disabled(clock):
    emb = HashingEmbedder()
    c = SemanticCache(emb, NumpyIndex(emb.dim), order_guard=False, clock=clock)
    c.store(key("convert km to miles"), resp(), ttl=60)
    assert c.lookup(key("convert miles to km"), 0.5).hit


def test_guard_falls_back_to_next_candidate(cache):
    cache.store(key("convert km to miles"), resp("swap"), ttl=60)
    cache.store(key("convert miles to km quickly"), resp("right"), ttl=60)
    lk = cache.lookup(key("convert miles to km"), 0.5)
    assert lk.hit and lk.entry.response["choices"][0]["message"]["content"] == "right"
