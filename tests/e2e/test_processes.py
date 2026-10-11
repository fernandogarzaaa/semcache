"""End-to-end: real mock-upstream and proxy processes over real sockets."""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

pytestmark = pytest.mark.slow


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(url: str, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(url)


@pytest.fixture
def stack():
    mport, pport = free_port(), free_port()
    env = {**os.environ, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    mock = subprocess.Popen(
        [sys.executable, "-m", "semcache", "mock-upstream", "--port", str(mport), "--latency-ms", "50"], env=env
    )
    proxy_env = {**env, "SEMCACHE_UPSTREAM_URL": f"http://127.0.0.1:{mport}/v1", "SEMCACHE_INDEX": "numpy"}
    proxy = subprocess.Popen(
        [sys.executable, "-m", "semcache", "serve", "--host", "127.0.0.1", "--port", str(pport)], env=proxy_env
    )
    try:
        wait(f"http://127.0.0.1:{mport}/v1/models")
        wait(f"http://127.0.0.1:{pport}/healthz")
        yield f"http://127.0.0.1:{pport}"
    finally:
        for p in (proxy, mock):
            p.terminate()
            p.wait(timeout=10)


def test_openai_shaped_requests_hit_after_first_call(stack):
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "How do I undo the last git commit?"}]}
    with httpx.Client(base_url=stack, trust_env=False, timeout=10) as c:
        t0 = time.perf_counter()
        first = c.post("/v1/chat/completions", json=body)
        miss_s = time.perf_counter() - t0
        assert first.headers["x-semcache"] == "miss"
        para = {**body, "messages": [{"role": "user", "content": "hey, how do I undo the last git commit? thanks"}]}
        t0 = time.perf_counter()
        second = c.post("/v1/chat/completions", json=para)
        hit_s = time.perf_counter() - t0
        assert second.headers["x-semcache"] == "hit"
        assert second.json()["choices"] == first.json()["choices"]
        assert hit_s < miss_s
        streamed = c.post("/v1/chat/completions", json={**body, "stream": True})
        assert streamed.headers["x-semcache"] == "hit" and streamed.text.rstrip().endswith("data: [DONE]")
        assert "semcache_requests_total" in c.get("/metrics").text
