import pytest

from semcache.config import Settings
from semcache.upstream import Provider, Router, from_anthropic, to_anthropic


def test_settings_defaults_and_env():
    s = Settings.from_env({})
    assert s.embedder == "hashing" and s.index == "auto" and s.enabled and s.adaptive and s.normalize
    assert s.thresholds == {"classification": 0.88, "general": 0.92, "creative": 0.97}
    env = {
        "SEMCACHE_UPSTREAM_URL": "http://x/v1",
        "SEMCACHE_UPSTREAM_API_KEY": "k",
        "SEMCACHE_EMBEDDER": "minilm",
        "SEMCACHE_MAX_ENTRIES": "5",
        "SEMCACHE_THRESHOLDS": '{"general": 0.9}',
        "SEMCACHE_ADAPTIVE": "false",
        "SEMCACHE_NORMALIZE": "0",
        "SEMCACHE_TTL_SHORT": "60",
        "SEMCACHE_TTL_DEFAULT": "120",
        "SEMCACHE_TTL_LONG": "240",
        "SEMCACHE_ENABLED": "no",
        "SEMCACHE_ADMIN_TOKEN": "t",
        "SEMCACHE_UPSTREAM_TIMEOUT": "5",
        "SEMCACHE_HASHING_DIM": "64",
        "SEMCACHE_NEAR_MISS_WINDOW": "0.1",
        "SEMCACHE_INDEX": "numpy",
    }
    s = Settings.from_env(env)
    assert s.upstream_url == "http://x/v1" and s.upstream_api_key == "k" and s.embedder == "minilm"
    assert s.max_entries == 5 and s.thresholds["general"] == 0.9 and s.thresholds["creative"] == 0.97
    assert not s.adaptive and not s.normalize and not s.enabled
    assert (s.ttl_short, s.ttl_default, s.ttl_long) == (60, 120, 240)
    assert s.upstream_timeout == 5 and s.hashing_dim == 64 and s.near_miss_window == 0.1 and s.index == "numpy"
    red = s.redacted()
    assert red["upstream_api_key"] == "***" and red["admin_token"] == "***"
    assert Settings().redacted()["admin_token"] == ""


@pytest.mark.parametrize(
    "raw,val", [("1", True), ("TRUE", True), ("on", True), ("yes", True), ("0", False), ("off", False)]
)
def test_bool_parsing(raw, val):
    assert Settings.from_env({"SEMCACHE_ADAPTIVE": raw}).adaptive is val


def test_blank_numbers_use_defaults():
    s = Settings.from_env({"SEMCACHE_MAX_ENTRIES": " ", "SEMCACHE_TTL_LONG": ""})
    assert s.max_entries == 10_000 and s.ttl_long == 604800


def test_thresholds_must_be_object():
    with pytest.raises(ValueError):
        Settings.from_env({"SEMCACHE_THRESHOLDS": "[0.9]"})


def test_router_longest_prefix(monkeypatch):
    monkeypatch.setenv("ANTH_KEY", "secret")
    r = Router.from_config(
        "http://default/v1/",
        "dk",
        '[{"prefix": "claude-", "kind": "anthropic", "base_url": "https://a/", "api_key_env": "ANTH_KEY"},'
        ' {"prefix": "claude-3-5", "base_url": "http://b", "name": "b", "timeout": 3},'
        ' {"prefix": "llama", "base_url": "http://ollama:11434/v1"}]',
    )
    assert r.default.base_url == "http://default/v1" and r.default.api_key == "dk"
    assert r.pick("gpt-4o") is r.default
    a = r.pick("claude-3-opus")
    assert a.kind == "anthropic" and a.base_url == "https://a" and a.api_key == "secret" and a.name == "route0"
    b = r.pick("claude-3-5-sonnet")
    assert b.name == "b" and b.kind == "openai" and b.timeout == 3 and b.api_key == ""
    assert r.pick("llama3.1").base_url == "http://ollama:11434/v1"


def test_router_validation():
    with pytest.raises(ValueError):
        Router.from_config("http://d", "", '{"prefix": "x"}')
    with pytest.raises(ValueError):
        Provider("x", "grpc", "http://x")
    assert Router.from_config("http://d", "", "  ").routes == []


def test_to_anthropic():
    body = {
        "model": "claude-3-5-haiku",
        "messages": [
            {"role": "system", "content": "S1"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": "END",
    }
    out = to_anthropic(body)
    assert out["system"] == "S1"
    assert out["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "q"},
    ]
    assert out["max_tokens"] == 1024 and out["temperature"] == 0.2 and out["top_p"] == 0.9
    assert out["stop_sequences"] == ["END"]
    out2 = to_anthropic(
        {"model": "c", "messages": [{"role": "user", "content": "x"}], "max_tokens": 5, "stop": ["a", "b"]}
    )
    assert "system" not in out2 and out2["max_tokens"] == 5 and out2["stop_sequences"] == ["a", "b"]
    assert "temperature" not in out2
    assert to_anthropic({"model": "c", "messages": [], "max_completion_tokens": 7})["max_tokens"] == 7


def test_from_anthropic():
    resp = {
        "id": "msg_1",
        "model": "claude",
        "content": [{"type": "text", "text": "a"}, {"type": "tool_use"}, {"type": "text", "text": "b"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    out = from_anthropic(resp)
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "ab"}
    assert out["choices"][0]["finish_reason"] == "length"
    assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert out["id"] == "msg_1" and out["model"] == "claude"
    assert from_anthropic({"stop_reason": "end_turn"})["choices"][0]["finish_reason"] == "stop"
    assert from_anthropic({"stop_reason": "stop_sequence"})["choices"][0]["finish_reason"] == "stop"
    assert from_anthropic({"stop_reason": "tool_use"})["choices"][0]["finish_reason"] == "tool_calls"
    assert from_anthropic({})["usage"]["total_tokens"] == 0
