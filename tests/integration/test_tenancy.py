"""Oct 2026 audit: cache isolation per caller, proxy tokens, closed admin API."""

import httpx
import pytest

from semcache.cli import is_loopback, main, unsafe_bind_reason
from semcache.config import Settings, parse_proxy_tokens
from semcache.embedders import HashingEmbedder
from semcache.keys import build_key
from semcache.mock_upstream import create_mock_app
from semcache.server import tenant_for
from semcache.upstream import Router, UpstreamClient


def make(settings: Settings, handler=None):
    s = settings
    s.upstream_url = "http://mock/v1"
    s.index = "numpy"
    router = Router.from_config(s.upstream_url, s.upstream_api_key, "")
    transport = httpx.MockTransport(handler) if handler else httpx.ASGITransport(app=create_mock_app())
    upstream = UpstreamClient(router, httpx.AsyncClient(transport=transport))
    from semcache.server import create_app

    app = create_app(s, embedder=HashingEmbedder(), upstream=upstream)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://semcache")


BODY = {"model": "mock-gpt", "messages": [{"role": "user", "content": "What is the capital of Canada?"}]}


async def test_cached_answer_not_served_to_a_different_key():
    async with make(Settings()) as c:
        a1 = await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer key-A"})
        a2 = await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer key-A"})
        b1 = await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer key-B"})
        anon = await c.post("/v1/chat/completions", json=BODY)
        assert [r.headers["x-semcache"] for r in (a1, a2, b1, anon)] == ["miss", "hit", "miss", "miss"]


async def test_invalid_key_gets_no_cached_answer():
    """A bogus key never sees another caller's cache; upstream rejects it."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != "Bearer good":
            return httpx.Response(401, json={"error": {"message": "bad key"}})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Ottawa"}, "finish_reason": "stop"}
                ]
            },
        )

    async with make(Settings(), handler) as c:
        assert (
            await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer good"})
        ).status_code == 200
        bad = await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer stolen-guess"})
        assert bad.status_code == 401 and bad.headers["x-semcache"] == "miss"
        none = await c.post("/v1/chat/completions", json=BODY)
        assert none.status_code == 401
        again = await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer good"})
        assert again.headers["x-semcache"] == "hit"


async def test_shared_cache_is_opt_in():
    async with make(Settings(shared_cache=True)) as c:
        await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer key-A"})
        r = await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer key-B"})
        assert r.headers["x-semcache"] == "hit"


async def test_proxy_tokens_required_before_cache_lookup():
    s = Settings(proxy_tokens={"tok-a": "alpha", "tok-b": "beta"})
    async with make(s) as c:
        assert (await c.post("/v1/chat/completions", json=BODY, headers={"x-semcache-key": "tok-a"})).headers[
            "x-semcache"
        ] == "miss"
        # cached now; still refused without/with a wrong token
        r = await c.post("/v1/chat/completions", json=BODY)
        assert r.status_code == 401 and "x-semcache" not in r.headers
        assert r.headers["www-authenticate"] == "Bearer"
        assert (await c.post("/v1/chat/completions", json=BODY, headers={"x-semcache-key": "nope"})).status_code == 401
        assert (await c.get("/v1/models")).status_code == 401
        same = await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer tok-a"})
        assert same.headers["x-semcache"] == "hit"
        other = await c.post("/v1/chat/completions", json=BODY, headers={"x-semcache-key": "tok-b"})
        assert other.headers["x-semcache"] == "miss"
        assert (await c.get("/v1/models", headers={"x-semcache-key": "tok-b"})).status_code == 200


async def test_proxy_token_in_authorization_is_not_forwarded_upstream():
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]},
        )

    async with make(Settings(proxy_tokens={"tok-a": "alpha"}), handler) as c:
        await c.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer tok-a"})
        await c.post(
            "/v1/chat/completions",
            json={**BODY, "messages": [{"role": "user", "content": "different question entirely"}]},
            headers={"x-semcache-key": "tok-a", "authorization": "Bearer app-key"},
        )
    assert seen == [None, "Bearer app-key"]


async def test_admin_api_closed_by_default():
    async with make(Settings()) as c:
        for path in ("/v1/cache/stats", "/v1/cache/entries", "/v1/cache/config", "/v1/cache/near-misses"):
            r = await c.get(path)
            assert r.status_code == 403, path
            assert "SEMCACHE_ADMIN_TOKEN" in r.json()["detail"]
        assert (await c.post("/v1/cache/invalidate", json={"all": True})).status_code == 403
        assert (await c.post("/v1/cache/tune", json={})).status_code == 403
        assert (
            await c.post("/v1/cache/feedback", json={"entry_id": 1, "similarity": 1, "correct": True})
        ).status_code == 403
        # proxy path and health are unaffected
        assert (await c.get("/healthz")).status_code == 200
        assert (await c.post("/v1/chat/completions", json=BODY)).status_code == 200


async def test_admin_token_scheme_and_value():
    async with make(Settings(admin_token="adm")) as c:
        assert (await c.get("/v1/cache/stats", headers={"authorization": "adm"})).status_code == 401
        assert (await c.get("/v1/cache/stats", headers={"authorization": "Basic adm"})).status_code == 401
        assert (await c.get("/v1/cache/stats", headers={"authorization": "Bearer adm2"})).status_code == 401
        assert (await c.get("/v1/cache/stats", headers={"authorization": "bearer adm"})).status_code == 200


def test_tenant_for_variants():
    s = Settings()
    assert tenant_for(s, None, None).tenant == "anon"
    a = tenant_for(s, "Bearer x", None)
    assert a.tenant.startswith("auth:") and a.upstream_auth == "Bearer x"
    assert a.tenant != tenant_for(s, "Bearer y", None).tenant
    assert tenant_for(Settings(shared_cache=True), "Bearer x", None).tenant == ""
    p = Settings(proxy_tokens={"t1": "one"})
    assert tenant_for(p, "Bearer app", "t1") == type(a)(tenant="tenant:one", upstream_auth="Bearer app")
    assert tenant_for(p, "Bearer t1", None) == type(a)(tenant="tenant:one", upstream_auth=None)
    assert tenant_for(Settings(proxy_tokens={"t1": "one"}, shared_cache=True), None, "t1").tenant == ""
    from fastapi import HTTPException

    for auth, key in ((None, None), ("Bearer t2", None), (None, ""), ("t1", None)):
        with pytest.raises(HTTPException) as exc:
            tenant_for(p, auth, key)
        assert exc.value.status_code == 401


def test_build_key_namespaces_by_tenant():
    k1 = build_key(BODY, tenant="a")
    k2 = build_key(BODY, tenant="b")
    assert k1 is not None and k2 is not None
    assert k1.namespace != k2.namespace and k1.query == k2.query
    assert build_key(BODY, tenant="a") == k1


def test_parse_proxy_tokens():
    out = parse_proxy_tokens(" alpha:tok1 , tok2,, ")
    assert out["tok1"] == "alpha"
    assert out["tok2"].startswith("t-") and len(out["tok2"]) == 14
    assert parse_proxy_tokens("") == {}
    with pytest.raises(ValueError):
        parse_proxy_tokens("alpha:")
    with pytest.raises(ValueError):
        parse_proxy_tokens(":tok")


def test_settings_from_env_security_fields():
    s = Settings.from_env({"SEMCACHE_PROXY_TOKENS": "a:t1", "SEMCACHE_SHARED_CACHE": "1", "SEMCACHE_ADMIN_OPEN": "yes"})
    assert s.proxy_tokens == {"t1": "a"} and s.shared_cache is True and s.admin_open is True
    d = Settings.from_env({})
    assert d.proxy_tokens == {} and d.shared_cache is False and d.admin_open is False
    red = s.redacted()
    assert red["proxy_tokens"] == ["a"] and "t1" not in str(red)


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("10.1.2.3", False),
        ("box", False),
    ],
)
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


def test_unsafe_bind_reason():
    assert unsafe_bind_reason("127.0.0.1", Settings(upstream_api_key="k"), {}) == ""
    assert unsafe_bind_reason("0.0.0.0", Settings(), {}) == ""  # pass-through keys, cache partitioned
    assert "SEMCACHE_PROXY_TOKENS" in unsafe_bind_reason("0.0.0.0", Settings(upstream_api_key="k"), {})
    assert "SEMCACHE_PROXY_TOKENS" in unsafe_bind_reason(
        "0.0.0.0", Settings(routes_json='[{"prefix":"c","api_key_env":"K"}]'), {}
    )
    assert unsafe_bind_reason("0.0.0.0", Settings(upstream_api_key="k", proxy_tokens={"t": "a"}), {}) == ""
    assert "ADMIN_OPEN" in unsafe_bind_reason("0.0.0.0", Settings(admin_open=True), {})
    assert unsafe_bind_reason("0.0.0.0", Settings(admin_open=True, admin_token="x"), {}) == ""
    assert unsafe_bind_reason("0.0.0.0", Settings(upstream_api_key="k"), {"SEMCACHE_ALLOW_OPEN": "1"}) == ""
    assert unsafe_bind_reason("0.0.0.0", Settings(upstream_api_key="k"), {"SEMCACHE_ALLOW_OPEN": "0"}) != ""


def test_serve_refuses_open_relay(monkeypatch, capsys):
    monkeypatch.setenv("SEMCACHE_UPSTREAM_API_KEY", "sk-server")
    monkeypatch.delenv("SEMCACHE_PROXY_TOKENS", raising=False)
    monkeypatch.delenv("SEMCACHE_ALLOW_OPEN", raising=False)
    assert main(["serve", "--host", "0.0.0.0", "--port", "1"]) == 2
    assert "refusing to listen" in capsys.readouterr().err


def test_serve_default_host_is_loopback(monkeypatch):
    from semcache.cli import build_parser

    monkeypatch.delenv("SEMCACHE_HOST", raising=False)
    assert build_parser().parse_args(["serve"]).host == "127.0.0.1"
