import pytest

from semcache.policy import (
    TASK_CLASSIFICATION,
    TASK_CREATIVE,
    TASK_GENERAL,
    TIER_DEFAULT,
    TIER_LONG,
    TIER_NO_CACHE,
    TIER_SHORT,
    AdaptiveThresholds,
    TTLPolicy,
    classify_freshness,
    classify_task,
    creative_bypass,
)


@pytest.mark.parametrize(
    "q,tier",
    [
        ("what is the score right now", TIER_NO_CACHE),
        ("live score of the lakers game", TIER_NO_CACHE),
        ("breaking news on the election", TIER_NO_CACHE),
        ("real-time btc price", TIER_NO_CACHE),
        ("What's the weather today", TIER_SHORT),
        ("latest python release", TIER_SHORT),
        ("current usd to eur exchange rate", TIER_SHORT),
        ("who won the 2026 world cup", TIER_SHORT),
        ("what is a closure", TIER_LONG),
        ("Explain the CAP theorem", TIER_LONG),
        ("difference between tcp and udp", TIER_LONG),
        ("history of rome", TIER_LONG),
        ("who invented the telephone", TIER_LONG),
        ("sort this list for me", TIER_DEFAULT),
        ("give me a regex for emails", TIER_DEFAULT),
        ("known issues with nowadays tooling", TIER_DEFAULT),
    ],
)
def test_classify_freshness(q, tier):
    assert classify_freshness(q) == tier


@pytest.mark.parametrize(
    "q,system,task",
    [
        ("Is this spam? 'win money'", "", TASK_CLASSIFICATION),
        ("great product, loved it", "Classify the sentiment as positive or negative", TASK_CLASSIFICATION),
        ("extract the dates from this text", "", TASK_CLASSIFICATION),
        ("write me a poem about the sea", "", TASK_CREATIVE),
        ("brainstorm names for a bakery", "", TASK_CREATIVE),
        ("a haiku on autumn", "", TASK_CREATIVE),
        ("what is a closure", "", TASK_GENERAL),
        ("what is a closure", "You are a helpful assistant", TASK_GENERAL),
    ],
)
def test_classify_task(q, system, task):
    assert classify_task(q, system) == task


def test_classification_beats_creative():
    assert classify_task("classify this story", "") == TASK_CLASSIFICATION


def test_ttl_policy():
    p = TTLPolicy(short=10, default=20, long=30)
    assert p.ttl_for(TIER_NO_CACHE) == 0.0
    assert p.ttl_for(TIER_SHORT) == 10
    assert p.ttl_for(TIER_DEFAULT) == 20
    assert p.ttl_for(TIER_LONG) == 30
    assert p.ttl_for("anything-else") == 20
    d = TTLPolicy()
    assert d.short == 3600 and d.default == 21600 and d.long == 604800


def test_creative_bypass():
    assert creative_bypass(TASK_CREATIVE, 0.8)
    assert creative_bypass(TASK_CREATIVE, 1.2)
    assert not creative_bypass(TASK_CREATIVE, 0.79)
    assert not creative_bypass(TASK_CREATIVE, None)
    assert not creative_bypass(TASK_GENERAL, 1.5)
    assert creative_bypass(TASK_CREATIVE, 0.5, limit=0.5)


def test_thresholds_defaults():
    t = AdaptiveThresholds()
    assert t.threshold(TASK_CLASSIFICATION) == 0.88
    assert t.threshold(TASK_GENERAL) == 0.92
    assert t.threshold(TASK_CREATIVE) == 0.97
    # unknown task types start from the general threshold
    assert t.threshold("other") == 0.92
    assert AdaptiveThresholds(base={"general": 0.8}).threshold("other") == 0.8
    assert AdaptiveThresholds(base={}).threshold("other") == 0.92


def test_thresholds_clamped_and_validated():
    t = AdaptiveThresholds(base={"general": 0.5, "creative": 1.0}, floor=0.7, ceiling=0.99)
    assert t.threshold("general") == 0.7
    assert t.threshold("creative") == 0.99
    with pytest.raises(ValueError):
        AdaptiveThresholds(floor=0.9, ceiling=0.8)
    with pytest.raises(ValueError):
        AdaptiveThresholds(floor=0.0)
    with pytest.raises(ValueError):
        AdaptiveThresholds(ceiling=1.01)
    AdaptiveThresholds(floor=0.8, ceiling=0.8)  # equal is allowed
    AdaptiveThresholds(floor=0.5, ceiling=1.0)


def test_false_hit_raises_threshold_above_it():
    t = AdaptiveThresholds(base={"general": 0.85}, margin=0.01)
    assert t.feedback("general", 0.88, correct=False) == pytest.approx(0.89)
    assert t.threshold("general") == pytest.approx(0.89)
    # a false hit below the current threshold does not lower it
    assert t.feedback("general", 0.80, correct=False) == pytest.approx(0.89)
    # exactly at threshold - margin: similarity + margin == threshold, unchanged
    assert t.feedback("general", 0.88, correct=False) == pytest.approx(0.89)


def test_correct_hits_lower_slowly_but_not_below_worst_false_hit():
    t = AdaptiveThresholds(base={"general": 0.85}, margin=0.01, step=0.01)
    t.feedback("general", 0.86, correct=False)  # -> 0.87, worst 0.86
    assert t.feedback("general", 0.9, correct=True) == pytest.approx(0.87)
    t2 = AdaptiveThresholds(base={"general": 0.85}, step=0.01)
    assert t2.feedback("general", 0.9, correct=True) == pytest.approx(0.84)
    assert t2.feedback("general", 0.9, correct=True) == pytest.approx(0.83)


def test_correct_hits_respect_floor():
    t = AdaptiveThresholds(base={"general": 0.71}, floor=0.7, step=0.05)
    assert t.feedback("general", 0.9, correct=True) == pytest.approx(0.7)


def test_false_hit_respects_ceiling():
    t = AdaptiveThresholds(base={"general": 0.9}, ceiling=0.95, margin=0.01)
    assert t.feedback("general", 0.99, correct=False) == pytest.approx(0.95)


def test_non_adaptive_records_but_does_not_move():
    t = AdaptiveThresholds(base={"general": 0.85}, adaptive=False)
    assert t.feedback("general", 0.95, correct=False) == 0.85
    assert t.feedback("general", 0.95, correct=True) == 0.85
    snap = t.snapshot()["general"]
    assert snap["bad_feedback"] == 1 and snap["good_feedback"] == 1
    assert snap["worst_false_hit"] == 0.95


def test_snapshot_shape_sorted():
    t = AdaptiveThresholds()
    snap = t.snapshot()
    assert list(snap) == sorted(snap)
    assert snap["general"] == {"threshold": 0.92, "worst_false_hit": -1.0, "good_feedback": 0.0, "bad_feedback": 0.0}
    t.feedback("general", 0.9, correct=True)
    t.feedback("general", 0.86, correct=False)
    s = t.snapshot()["general"]
    assert s["good_feedback"] == 1.0 and s["bad_feedback"] == 1.0
    assert s["worst_false_hit"] == 0.86
