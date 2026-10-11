import json

import numpy as np
import pytest

from semcache.pricing import DEFAULT_PRICES, PriceTable
from semcache.streaming import StreamAccumulator, completion_to_sse, estimate_tokens, sse
from semcache.tuner import ReplayResult, default_thresholds, pick_threshold, replay, sweep_history, sweep_pairs

# ---- tuner -------------------------------------------------------------------------------------


def test_default_thresholds():
    t = default_thresholds()
    assert t[0] == 0.70 and t[-1] == 0.99 and len(t) == 30
    assert default_thresholds(0.8, 0.9, 0.05) == [0.8, 0.85, 0.9]


def test_sweep_pairs():
    sims = [0.95, 0.9, 0.85, 0.8, 0.6]
    labels = [True, True, False, True, False]
    pts = {p.threshold: p for p in sweep_pairs(sims, labels, [0.5, 0.85, 0.9, 0.99])}
    assert pts[0.5].recall == 1.0 and pts[0.5].hits == 5
    assert pts[0.5].false_hit_rate == pytest.approx(2 / 5) and pts[0.5].false_positive_rate == 1.0
    assert pts[0.85].hits == 3 and pts[0.85].recall == pytest.approx(2 / 3)
    assert pts[0.85].false_hit_rate == pytest.approx(1 / 3) and pts[0.85].false_positive_rate == 0.5
    assert pts[0.9].hits == 2 and pts[0.9].false_hit_rate == 0.0 and pts[0.9].false_positive_rate == 0.0
    assert pts[0.99].hits == 0 and pts[0.99].recall == 0.0 and pts[0.99].false_hit_rate == 0.0
    d = pts[0.85].as_dict()
    assert d == {"threshold": 0.85, "recall": 0.6667, "false_hit_rate": 0.3333, "false_positive_rate": 0.5, "hits": 3}


def test_sweep_pairs_edge_cases():
    with pytest.raises(ValueError):
        sweep_pairs([0.5], [True, False], [0.5])
    (p,) = sweep_pairs([0.9], [True], [0.5])
    assert p.false_positive_rate == 0.0 and p.recall == 1.0
    (q,) = sweep_pairs([0.9], [False], [0.5])
    assert q.recall == 0.0 and q.false_positive_rate == 1.0


def test_sweep_history():
    h = [(0.9, True), (0.8, False), (0.95, True), (0.5, False)]
    out = sweep_history(h, [0.8, 0.9, 0.99])
    assert out[0] == {"threshold": 0.8, "hit_rate": 0.75, "lookups": 4}
    assert out[1]["hit_rate"] == 0.5
    assert out[2]["hit_rate"] == 0.0
    assert sweep_history([], [0.9]) == [{"threshold": 0.9, "hit_rate": 0.0, "lookups": 0}]


def _vecs(rows):
    v = np.array(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_replay_counts_hits_and_false_hits():
    vecs = _vecs([[1, 0, 0], [1, 0.1, 0], [0, 1, 0], [0.1, 1, 0], [1, 0.05, 0], [0, 0, 1]])
    intents = ["a", "a", "b", "c", "a", None]
    r = replay(vecs, intents, threshold=0.95, window=3)
    # req1 hits a (correct); req3 hits b but is c (false); req4 hits a (correct); req5 new
    assert r.requests == 6 and r.hits == 3 and r.false_hits == 1
    assert r.hit_rate == pytest.approx(0.5)
    assert r.false_hit_rate == pytest.approx(1 / 3)
    assert r.correct_hit_rate == pytest.approx(2 / 6)
    assert r.hit_rate_by_window == pytest.approx((1 / 3, 2 / 3))
    assert r.threshold == 0.95


def test_replay_none_intent_hit_is_false_and_misses_are_stored():
    vecs = _vecs([[1, 0], [1, 0], [1, 0]])
    r = replay(vecs, [None, None, "x"], threshold=0.9)
    assert r.hits == 2 and r.false_hits == 2


def test_replay_threshold_inclusive_and_strict_above():
    vecs = _vecs([[1, 0], [1, 1]])
    sim = float(vecs[0] @ vecs[1])
    assert replay(vecs, ["a", "a"], threshold=sim).hits == 1
    assert replay(vecs, ["a", "a"], threshold=sim + 1e-4).hits == 0


def test_replay_picks_best_stored_match():
    vecs = _vecs([[1, 0, 0], [0, 1, 0], [0.1, 1, 0]])
    r = replay(vecs, ["a", "b", "b"], threshold=0.9)
    assert r.hits == 1 and r.false_hits == 0


def test_replay_validation_and_empty():
    with pytest.raises(ValueError):
        replay(_vecs([[1, 0]]), ["a", "b"], 0.9)
    with pytest.raises(ValueError):
        replay(_vecs([[1, 0]]), ["a"], 0.9, window=0)
    r = replay(np.zeros((0, 2), dtype=np.float32), [], 0.9)
    assert r.hit_rate == 0.0 and r.false_hit_rate == 0.0 and r.correct_hit_rate == 0.0
    assert r.hit_rate_by_window == ()


def test_replay_as_dict():
    r = ReplayResult(0.9, 10, 4, 1, 0.4, 0.25, 0.3, (0.123456, 0.5))
    assert r.as_dict() == {
        "threshold": 0.9,
        "requests": 10,
        "hits": 4,
        "false_hits": 1,
        "hit_rate": 0.4,
        "false_hit_rate": 0.25,
        "correct_hit_rate": 0.3,
        "hit_rate_by_window": [0.1235, 0.5],
    }


def test_pick_threshold():
    rs = [
        ReplayResult(0.80, 10, 6, 2, 0.6, 0.33, 0.4, ()),
        ReplayResult(0.85, 10, 5, 0, 0.5, 0.0, 0.5, ()),
        ReplayResult(0.90, 10, 5, 0, 0.5, 0.0, 0.5, ()),
        ReplayResult(0.95, 10, 3, 0, 0.3, 0.0, 0.3, ()),
    ]
    assert pick_threshold(rs, 0.01).threshold == 0.85
    assert pick_threshold(rs, 0.5).threshold == 0.85  # 0.80 has fewer correct hits
    assert pick_threshold(rs[:1], 0.33).threshold == 0.80
    assert pick_threshold(rs[:1], 0.1) is None


# ---- pricing -----------------------------------------------------------------------------------


def test_price_lookup_exact_prefix_unknown():
    p = PriceTable()
    assert p.lookup("gpt-4o") == (2.50, 10.00)
    assert p.lookup("gpt-4o-mini-2024-07-18") == (0.15, 0.60)  # longest prefix wins over gpt-4o
    assert p.lookup("unknown-model") == (0.0, 0.0)
    assert p.cost("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.5)
    assert p.cost("gpt-4o-mini", 1000, 2000) == pytest.approx((1000 * 0.15 + 2000 * 0.60) / 1e6)
    assert p.cost("unknown", 10, 10) == 0.0


def test_price_from_json_merges_and_validates():
    p = PriceTable.from_json('{"mock-gpt": [1, 2]}')
    assert p.lookup("mock-gpt") == (1.0, 2.0)
    assert p.lookup("gpt-4o") == DEFAULT_PRICES["gpt-4o"]
    assert PriceTable.from_json("  ").prices == DEFAULT_PRICES
    assert PriceTable.from_json('{"a": [1, 1]}', base={}).prices == {"a": (1.0, 1.0)}
    with pytest.raises(ValueError):
        PriceTable.from_json("[1, 2]")
    with pytest.raises(ValueError):
        PriceTable.from_json('{"a": [1]}')
    with pytest.raises(ValueError):
        PriceTable.from_json('{"a": 3}')
    assert PriceTable({}).prices == {}


# ---- streaming ---------------------------------------------------------------------------------


def chunk(content=None, finish=None, role=None, idx=0, **extra):
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    obj = {"id": "c1", "model": "m", "created": 5, "choices": [{"index": idx, "delta": delta, "finish_reason": finish}]}
    obj.update(extra)
    return sse(obj)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_sse_format():
    assert sse({"a": 1}) == b'data: {"a":1}\n\n'


def test_accumulator_complete_stream_split_across_chunks():
    raw = chunk("", role="assistant") + chunk("Hel") + chunk("lo") + chunk(finish="stop") + b"data: [DONE]\n\n"
    acc = StreamAccumulator()
    for i in range(0, len(raw), 7):  # arbitrary byte boundaries
        acc.feed(raw[i : i + 7])
    acc.close()
    assert acc.done and acc.complete
    out = acc.to_completion("prompt words here")
    assert out["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
    ]
    assert out["id"] == "c1" and out["model"] == "m" and out["created"] == 5 and out["object"] == "chat.completion"
    assert out["usage"]["estimated"] is True
    assert out["usage"]["completion_tokens"] == estimate_tokens("Hello")
    assert out["usage"]["prompt_tokens"] == estimate_tokens("prompt words here")
    assert out["usage"]["total_tokens"] == out["usage"]["prompt_tokens"] + out["usage"]["completion_tokens"]


def test_accumulator_uses_stream_usage():
    acc = StreamAccumulator()
    acc.feed(chunk("x", role="assistant") + chunk(finish="stop"))
    acc.feed(sse({"id": "c1", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}))
    acc.feed(b"data: [DONE]\n\n")
    assert acc.to_completion()["usage"] == {"prompt_tokens": 3, "completion_tokens": 4}


def test_accumulator_incomplete_cases():
    no_done = StreamAccumulator()
    no_done.feed(chunk("x") + chunk(finish="stop"))
    assert not no_done.complete

    length = StreamAccumulator()
    length.feed(chunk("x") + chunk(finish="length") + b"data: [DONE]\n\n")
    assert length.done and not length.complete

    broken = StreamAccumulator()
    broken.feed(chunk("x") + b"data: {not json\n\n" + chunk(finish="stop") + b"data: [DONE]\n\n")
    assert broken.broken and not broken.complete

    err = StreamAccumulator()
    err.feed(sse({"error": {"message": "boom"}}) + b"data: [DONE]\n\n")
    assert err.broken and not err.complete

    nonobj = StreamAccumulator()
    nonobj.feed(b"data: [1, 2]\n\n")
    assert nonobj.broken

    tools = StreamAccumulator()
    tools.feed(sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"id": "t"}]}, "finish_reason": "stop"}]}))
    tools.feed(b"data: [DONE]\n\n")
    assert tools.has_tool_calls and not tools.complete

    fn = StreamAccumulator()
    fn.feed(sse({"choices": [{"index": 0, "delta": {"function_call": {"name": "f"}}}]}))
    assert fn.has_tool_calls

    empty = StreamAccumulator()
    empty.feed(b"data: [DONE]\n\n")
    assert empty.done and not empty.complete


def test_accumulator_ignores_comments_and_handles_unterminated_last_line():
    acc = StreamAccumulator()
    acc.feed(b": keep-alive\n\nevent: x\n" + chunk("a", role="assistant") + chunk(finish="stop") + b"data: [DONE]")
    assert not acc.done
    acc.close()
    assert acc.done and acc.complete
    acc.close()  # idempotent


def test_accumulator_multiple_choices_and_defaults():
    acc = StreamAccumulator()
    acc.feed(chunk("A", idx=0) + chunk("B", idx=1) + chunk(finish="stop", idx=0) + chunk(finish="stop", idx=1))
    acc.feed(b"data: [DONE]\n\n")
    out = acc.to_completion()
    assert [c["message"]["content"] for c in out["choices"]] == ["A", "B"]
    assert out["choices"][1]["message"]["role"] == "assistant"
    blank = StreamAccumulator()
    blank.feed(sse({"choices": [{"delta": {"content": "z"}, "finish_reason": "stop"}]}) + b"data: [DONE]\n\n")
    out = blank.to_completion()
    assert out["id"] == "chatcmpl-semcache" and out["model"] == "" and isinstance(out["created"], int)
    assert out["choices"][0]["index"] == 0


def test_accumulator_first_id_wins():
    acc = StreamAccumulator()
    acc.feed(sse({"id": "first", "model": "m1", "created": 1, "choices": []}))
    acc.feed(sse({"id": "second", "model": "m2", "created": 2, "choices": []}))
    assert acc.id == "first" and acc.model == "m1" and acc.created == 1


def _parse(stream):
    events = []
    for raw in b"".join(stream).split(b"\n\n"):
        if raw.startswith(b"data: "):
            payload = raw[6:]
            events.append(payload if payload == b"[DONE]" else json.loads(payload))
    return events


def test_completion_to_sse_roundtrip():
    response = {
        "id": "x",
        "created": 9,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    events = _parse(completion_to_sse(response))
    assert events[-1] == b"[DONE]"
    assert events[0]["object"] == "chat.completion.chunk" and events[0]["id"] == "x" and events[0]["created"] == 9
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": "hi there"}
    assert events[1]["choices"][0]["finish_reason"] == "stop"
    assert events[2]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2} and events[2]["choices"] == []
    acc = StreamAccumulator()
    for c in completion_to_sse(response):
        acc.feed(c)
    assert acc.complete and acc.to_completion()["choices"][0]["message"]["content"] == "hi there"


def test_completion_to_sse_tool_calls_and_no_usage():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function"}]},
                "finish_reason": "tool_calls",
            }
        ]
    }
    events = _parse(completion_to_sse(response))
    delta = events[0]["choices"][0]["delta"]
    assert delta["content"] == "" and delta["tool_calls"] == [{"index": 0, "id": "t1", "type": "function"}]
    assert events[0]["id"] == "chatcmpl-semcache" and events[0]["model"] == ""
    assert len(events) == 3  # delta, finish, [DONE]


def test_replay_exact_only_baseline():
    v = np.zeros((4, 1), dtype=np.float32)
    r = replay(v, ["a", "a", "b", "a"], 0.0, texts=["x", "x", "y", "x2"], exact_only=True)
    assert r.hits == 1 and r.false_hits == 0
    r2 = replay(v[:2], ["a", "b"], 0.0, texts=["x", "x"], exact_only=True)
    assert r2.hits == 1 and r2.false_hits == 1
    with pytest.raises(ValueError):
        replay(v, ["a"] * 4, 0.9, exact_only=True)
    with pytest.raises(ValueError):
        replay(v, ["a"] * 4, 0.9, guard=lambda a, b: False)


def test_replay_guard_vetoes_hit_and_stores_query():
    vecs = _vecs([[1, 0], [1, 0], [1, 0]])
    vetoes = []

    def guard(stored, text):
        vetoes.append((stored, text))
        return text == "b"

    r = replay(vecs, ["a", "b", "b"], 0.9, texts=["a", "b", "c"], guard=guard)
    assert r.hits == 1 and r.false_hits == 1  # "c" hits the first stored entry ("a")
    assert vetoes[0] == ("a", "b")
    r2 = replay(vecs, ["a", "a", "a"], 0.9, texts=["a", "b", "c"], guard=lambda s, t: False)
    assert r2.hits == 2 and r2.false_hits == 0


def test_replay_exact_first_occurrence_wins():
    v = np.zeros((3, 1), dtype=np.float32)
    r = replay(v, ["a", "a", "a"], 0.0, texts=["x", "y", "x"], exact_only=True)
    assert r.hits == 1 and r.hit_rate_by_window == (pytest.approx(1 / 3),)
