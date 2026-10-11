import json

import httpx
import pytest

from semcache.config import Settings
from semcache.embedders import HashingEmbedder
from semcache.mock_upstream import create_mock_app
from semcache.server import create_app
from semcache.upstream import Router, UpstreamClient


def make(settings=None, routes_json=""):
    mock = create_mock_app()
    s = settings or Settings()
    s.upstream_url = "http://mock/v1"
    s.prices_json = '{"mock-gpt": [1.0, 2.0]}'
    s.index = "numpy"
    router = Router.from_config(s.upstream_url, s.upstream_api_key, routes_json)
    upstream = UpstreamClient(router, httpx.AsyncClient(transport=httpx.ASGITransport(app=mock)))
    app = create_app(s, embedder=HashingEmbedder(), upstream=upstream)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://semcache")
    return client, app, mock


def chat(q, model="mock-gpt", system=None, **kw):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return {"model": model, "messages": msgs, **kw}


async def test_miss_then_hit_then_paraphrase_hit():
    client, app, mock = make()
    async with client:
        r1 = await client.post("/v1/chat/completions", json=chat("How do I reverse a list in Python?"))
        assert r1.status_code == 200 and r1.headers["x-semcache"] == "miss"
        assert r1.headers["x-semcache-similarity"] == "none"
        r2 = await client.post("/v1/chat/completions", json=chat("How do I reverse a list in Python?"))
        assert r2.headers["x-semcache"] == "hit"
        assert r2.json() == r1.json()
        assert float(r2.headers["x-semcache-similarity"]) == pytest.approx(1.0, abs=1e-3)
        r3 = await client.post("/v1/chat/completions", json=chat("hey, how do i reverse a list in python please"))
        assert r3.headers["x-semcache"] == "hit"
        assert mock.state.calls == 1
        usage = r1.json()["usage"]
        expected = (usage["prompt_tokens"] * 1.0 + usage["completion_tokens"] * 2.0) / 1e6
        assert float(r3.headers["x-semcache-saved-usd"]) == pytest.approx(expected, abs=1e-6)
        stats = (await client.get("/v1/cache/stats")).json()
        assert stats["hits"] == 2 and stats["entries"] == 1
        assert stats["estimated_saved_usd"] == pytest.approx(2 * expected, abs=1e-6)
        assert stats["estimated_upstream_usd"] == pytest.approx(expected, abs=1e-6)


async def test_different_system_prompt_or_params_never_shares():
    client, _, mock = make()
    async with client:
        await client.post("/v1/chat/completions", json=chat("what is python", system="A"))
        r = await client.post("/v1/chat/completions", json=chat("what is python", system="B"))
        assert r.headers["x-semcache"] == "miss"
        r = await client.post("/v1/chat/completions", json=chat("what is python", system="A", temperature=0.9))
        assert r.headers["x-semcache"] == "miss"
        assert mock.state.calls == 3


async def test_bypass_reasons():
    client, _, mock = make()
    async with client:
        r = await client.post("/v1/chat/completions", json=chat("hi"), headers={"X-Semcache-Bypass": "1"})
        assert r.headers["x-semcache"] == "bypass" and r.headers["x-semcache-reason"] == "header"
        r = await client.post("/v1/chat/completions", json=chat("what is the live score right now"))
        assert r.headers["x-semcache-reason"] == "live_data"
        r = await client.post("/v1/chat/completions", json=chat("write a poem about cats", temperature=1.0))
        assert r.headers["x-semcache-reason"] == "creative_high_temperature"
        r = await client.post("/v1/chat/completions", json=chat("what is go", n=3))
        assert r.headers["x-semcache-reason"] == "uncacheable_request"
        r = await client.post(
            "/v1/chat/completions", json={"model": "mock-gpt", "messages": [{"role": "system", "content": "x"}]}
        )
        assert r.headers["x-semcache-reason"] == "no_user_message"
        assert (await client.get("/v1/cache/stats")).json()["entries"] == 0


async def test_live_data_cached_when_ttl_override_given():
    client, _, _ = make()
    async with client:
        h = {"X-Semcache-TTL": "30"}
        await client.post("/v1/chat/completions", json=chat("btc price right now"), headers=h)
        r = await client.post("/v1/chat/completions", json=chat("btc price right now"), headers=h)
        assert r.headers["x-semcache"] == "hit"


async def test_disabled():
    client, _, _ = make(Settings(enabled=False))
    async with client:
        r = await client.post("/v1/chat/completions", json=chat("what is go"))
        assert r.headers["x-semcache-reason"] == "disabled"


async def test_threshold_header_and_validation():
    client, _, _ = make()
    async with client:
        await client.post("/v1/chat/completions", json=chat("alpha beta gamma"))
        r = await client.post(
            "/v1/chat/completions", json=chat("alpha beta delta"), headers={"X-Semcache-Threshold": "0.2"}
        )
        assert r.headers["x-semcache"] == "hit" and r.headers["x-semcache-threshold"] == "0.2000"
        r = await client.post("/v1/chat/completions", json=chat("x"), headers={"X-Semcache-Threshold": "abc"})
        assert r.status_code == 400
        r = await client.post("/v1/chat/completions", json=chat("x"), headers={"X-Semcache-TTL": "soon"})
        assert r.status_code == 400


async def test_bad_bodies():
    client, _, _ = make()
    async with client:
        r = await client.post("/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"})
        assert r.status_code == 400
        r = await client.post("/v1/chat/completions", json={"model": "m", "messages": "nope"})
        assert r.status_code == 400
        r = await client.post("/v1/chat/completions", json=[1, 2])
        assert r.status_code == 400


async def test_upstream_errors_and_truncation_not_cached():
    client, _, mock = make()
    async with client:
        r = await client.post("/v1/chat/completions", json=chat("please [fail] now"))
        assert r.status_code == 500 and r.headers["x-semcache"] == "miss"
        r = await client.post("/v1/chat/completions", json=chat("story [truncate]"))
        assert r.status_code == 200 and r.json()["choices"][0]["finish_reason"] == "length"
        r = await client.post("/v1/chat/completions", json=chat("story [truncate]"))
        assert r.headers["x-semcache"] == "miss"
        assert (await client.get("/v1/cache/stats")).json()["entries"] == 0
        metrics = (await client.get("/metrics")).text
        assert 'semcache_upstream_errors_total{reason="http_500"} 1.0' in metrics
        assert 'semcache_upstream_errors_total{reason="incomplete"} 2.0' in metrics


async def test_upstream_unreachable_returns_502():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    s = Settings(upstream_url="http://127.0.0.1:9/v1", index="numpy")
    up = UpstreamClient(
        Router.from_config(s.upstream_url, "", ""), httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    )
    app = create_app(s, embedder=HashingEmbedder(), upstream=up)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x") as client:
        r = await client.post("/v1/chat/completions", json=chat("what is go"))
        assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_error"
        r = await client.post("/v1/chat/completions", json=chat("what is go", stream=True))
        assert r.status_code == 502


def parse_sse(text):
    out = []
    for block in text.split("\n\n"):
        if block.startswith("data: "):
            payload = block[6:]
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


async def test_streaming_miss_is_relayed_and_cached_then_hit_streams():
    client, _, mock = make()
    async with client:
        r1 = await client.post("/v1/chat/completions", json=chat("explain python generators", stream=True))
        assert r1.headers["x-semcache"] == "miss" and r1.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r1.text)
        assert events[-1] == "[DONE]"
        streamed = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1] if e.get("choices"))
        r2 = await client.post("/v1/chat/completions", json=chat("explain python generators", stream=True))
        assert r2.headers["x-semcache"] == "hit"
        ev2 = parse_sse(r2.text)
        assert ev2[0]["choices"][0]["delta"]["content"] == streamed
        # a non-stream request is served from the streamed entry too
        r3 = await client.post("/v1/chat/completions", json=chat("explain python generators"))
        assert r3.headers["x-semcache"] == "hit"
        assert r3.json()["choices"][0]["message"]["content"] == streamed
        assert r3.json()["usage"]["estimated"] is True
        assert mock.state.calls == 1


async def test_broken_or_truncated_stream_not_cached():
    client, _, mock = make()
    async with client:
        for q in ("tell me about [broken-stream]", "tell me about [truncate] it"):
            await client.post("/v1/chat/completions", json=chat(q, stream=True))
            r = await client.post("/v1/chat/completions", json=chat(q, stream=True))
            assert r.headers["x-semcache"] == "miss"
        assert mock.state.calls == 4
        metrics = (await client.get("/metrics")).text
        assert 'semcache_upstream_errors_total{reason="stream_aborted"} 2.0' in metrics
        assert 'semcache_upstream_errors_total{reason="incomplete"} 2.0' in metrics


async def test_stream_upstream_error_status_passed_through():
    client, _, _ = make()
    async with client:
        r = await client.post("/v1/chat/completions", json=chat("[fail]", stream=True))
        assert r.status_code == 500 and "mock failure" in r.text


async def test_anthropic_route_translation_and_caching():
    routes = json.dumps([{"prefix": "claude-", "kind": "anthropic", "base_url": "http://mock"}])
    client, _, mock = make(routes_json=routes)
    async with client:
        r = await client.post("/v1/chat/completions", json=chat("what is rust", model="claude-3-5-haiku", system="S"))
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion" and body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["completion_tokens"] > 0
        r2 = await client.post("/v1/chat/completions", json=chat("what is rust", model="claude-3-5-haiku", system="S"))
        assert r2.headers["x-semcache"] == "hit"
        # same words to an OpenAI-routed model never get the Anthropic answer
        r3 = await client.post("/v1/chat/completions", json=chat("what is rust", model="mock-gpt", system="S"))
        assert r3.headers["x-semcache"] == "miss"
        # anthropic + stream: one upstream call replayed as SSE
        r4 = await client.post(
            "/v1/chat/completions", json=chat("explain borrowck", model="claude-3-5-haiku", stream=True)
        )
        assert r4.headers["content-type"].startswith("text/event-stream") and parse_sse(r4.text)[-1] == "[DONE]"
        r5 = await client.post("/v1/chat/completions", json=chat("x [truncate]", model="claude-3-5-haiku"))
        assert r5.json()["choices"][0]["finish_reason"] == "length"
        assert mock.state.calls == 4


async def test_invalidation_endpoints():
    client, _, _ = make()
    async with client:
        await client.post(
            "/v1/chat/completions", json=chat("what is python", system="A"), headers={"X-Semcache-Tags": "docs, faq"}
        )
        await client.post("/v1/chat/completions", json=chat("what is java", system="B"))
        await client.post("/v1/chat/completions", json=chat("explain go", system="B"))
        assert (await client.post("/v1/cache/invalidate", json={})).status_code == 400
        assert (await client.post("/v1/cache/invalidate", json={"tag": "faq"})).json() == {"removed": 1}
        assert (await client.post("/v1/cache/invalidate", json={"system_prompt": "B", "prefix": "explain"})).json() == {
            "removed": 1
        }
        entries = (await client.get("/v1/cache/entries")).json()
        assert entries["total"] == 1 and entries["entries"][0]["prompt"] == "what is java"
        assert (await client.post("/v1/cache/invalidate", json={"all": True})).json() == {"removed": 1}


async def test_feedback_adjusts_threshold_and_near_misses_and_tune():
    client, _, _ = make()
    async with client:
        await client.post("/v1/chat/completions", json=chat("alpha beta gamma"))
        r = await client.post(
            "/v1/chat/completions", json=chat("alpha beta delta"), headers={"X-Semcache-Threshold": "0.3"}
        )
        entry = int(r.headers["x-semcache-entry"])
        sim = float(r.headers["x-semcache-similarity"])
        fb = await client.post(
            "/v1/cache/feedback", json={"entry_id": entry, "similarity": max(sim, 0.95), "correct": False}
        )
        assert fb.json()["task"] == "general" and fb.json()["threshold"] == pytest.approx(max(sim, 0.95) + 0.01)
        assert (
            await client.post("/v1/cache/feedback", json={"entry_id": 999, "similarity": 0.9, "correct": True})
        ).status_code == 404
        bad = await client.post(
            "/v1/cache/feedback", json={"entry_id": 999, "similarity": 0.9, "correct": True, "task": "x"}
        )
        assert bad.status_code == 400
        ok = await client.post(
            "/v1/cache/feedback", json={"entry_id": 999, "similarity": 0.95, "correct": True, "task": "creative"}
        )
        assert ok.json()["threshold"] == pytest.approx(0.968)
        # a lookup just under threshold is a near miss
        await client.post(
            "/v1/chat/completions", json=chat("alpha beta delta"), headers={"X-Semcache-Threshold": f"{sim + 0.02}"}
        )
        nm = (await client.get("/v1/cache/near-misses")).json()["near_misses"]
        assert nm and nm[0]["nearest_prompt"] == "alpha beta gamma"
        hist = (await client.post("/v1/cache/tune", json={"thresholds": [0.1, 0.99]})).json()
        assert hist["source"] == "history" and hist["points"][0]["hit_rate"] == 1.0
        pairs = (
            await client.post(
                "/v1/cache/tune",
                json={
                    "pairs": [{"similarity": 0.9, "same_intent": True}, {"similarity": 0.95, "same_intent": False}],
                    "thresholds": [0.85],
                },
            )
        ).json()
        assert pairs["source"] == "pairs" and pairs["points"][0]["false_hit_rate"] == 0.5
        snap = (await client.get("/v1/cache/stats")).json()["thresholds"]
        assert snap["general"]["bad_feedback"] == 1


async def test_admin_token_required_when_configured():
    client, _, _ = make(Settings(admin_token="s3cret"))
    async with client:
        assert (await client.get("/v1/cache/stats")).status_code == 401
        assert (await client.get("/v1/cache/stats", headers={"authorization": "Bearer nope"})).status_code == 401
        ok = await client.get("/v1/cache/stats", headers={"authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        cfg = (await client.get("/v1/cache/config", headers={"authorization": "Bearer s3cret"})).json()
        assert cfg["admin_token"] == "***" and "s3cret" not in json.dumps(cfg)
        # the proxy itself is not behind the admin token
        assert (await client.post("/v1/chat/completions", json=chat("what is go"))).status_code == 200


async def test_metrics_health_models():
    client, _, _ = make()
    async with client:
        await client.post("/v1/chat/completions", json=chat("what is go"))
        await client.post("/v1/chat/completions", json=chat("what is go"))
        assert (await client.get("/healthz")).json()["entries"] == 1
        text = (await client.get("/metrics")).text
        assert 'semcache_requests_total{model="mock-gpt",result="hit"} 1.0' in text
        assert 'semcache_requests_total{model="mock-gpt",result="miss"} 1.0' in text
        assert "semcache_saved_usd_total" in text and "semcache_entries 1.0" in text
        assert 'semcache_threshold{task="general"} 0.92' in text
        assert "semcache_capacity_entries 10000.0" in text
        assert 'semcache_similarity_count{outcome="hit"} 1.0' in text
        models = await client.get("/v1/models")
        assert models.json()["data"][0]["id"] == "mock-gpt"


async def test_client_authorization_passthrough_and_configured_key():
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]},
        )

    for key, expected in (("", "Bearer app-key"), ("sk-proxy", "Bearer sk-proxy")):
        s = Settings(index="numpy", upstream_url="http://up/v1", upstream_api_key=key)
        up = UpstreamClient(
            Router.from_config(s.upstream_url, key, ""), httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        app = create_app(s, embedder=HashingEmbedder(), upstream=up)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x") as c:
            await c.post("/v1/chat/completions", json=chat("q"), headers={"authorization": "Bearer app-key"})
        assert seen[-1] == expected
