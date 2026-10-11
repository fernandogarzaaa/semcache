import asyncio

import numpy as np
import pytest

from semcache.batching import EmbedBatcher


def fake_embed(calls):
    def embed(texts):
        calls.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    return embed


async def test_concurrent_requests_are_batched_and_results_routed():
    calls = []
    b = EmbedBatcher(fake_embed(calls), max_batch=8, max_wait_ms=50)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    out = await asyncio.gather(*(b.embed(t) for t in texts))
    assert [float(v[0]) for v in out] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(calls) == 1 and calls[0] == texts
    assert b.batches == 1 and b.items == 5
    await b.aclose()


async def test_max_batch_splits():
    calls = []
    b = EmbedBatcher(fake_embed(calls), max_batch=2, max_wait_ms=50)
    out = await asyncio.gather(*(b.embed(str(i) * (i + 1)) for i in range(5)))
    assert [len(c) for c in calls] == [2, 2, 1]
    assert [float(v[0]) for v in out] == [1.0, 2.0, 3.0, 4.0, 5.0]
    await b.aclose()


async def test_single_request_does_not_wait_for_a_full_batch():
    calls = []
    b = EmbedBatcher(fake_embed(calls), max_batch=64, max_wait_ms=1)
    v = await asyncio.wait_for(b.embed("xyz"), 2)
    assert float(v[0]) == 3.0 and calls == [["xyz"]]
    await b.aclose()


async def test_errors_propagate_and_worker_survives():
    state = {"fail": True}

    def embed(texts):
        if state["fail"]:
            raise RuntimeError("model down")
        return np.ones((len(texts), 2), dtype=np.float32)

    b = EmbedBatcher(embed, max_wait_ms=1)
    with pytest.raises(RuntimeError, match="model down"):
        await b.embed("x")
    state["fail"] = False
    assert (await b.embed("x")).shape == (2,)
    assert b.batches == 1
    await b.aclose()
    await b.aclose()  # idempotent


def test_validation():
    with pytest.raises(ValueError):
        EmbedBatcher(lambda t: np.zeros((0, 1)), max_batch=0)
    b = EmbedBatcher(lambda t: np.zeros((0, 1)), max_batch=1, max_wait_ms=5)
    assert b.max_wait == 0.005 and b.max_batch == 1
