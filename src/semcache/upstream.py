"""Upstream providers and model-based routing.

The cache is provider-agnostic: it only ever sees OpenAI-shaped requests and
responses. Routing picks a provider by model-name prefix; the model name is
part of the cache namespace, so an answer from one provider is never served for
another provider's model.

* ``openai`` kind: any OpenAI-compatible ``/chat/completions`` endpoint
  (OpenAI, Ollama's ``/v1``, vLLM, llama.cpp server, Groq, OpenRouter, the
  user's own llmgate gateway ...). Streams are passed through byte for byte.
* ``anthropic`` kind: translated to the Messages API and back. Text only;
  ``stream=true`` is served by one non-streamed upstream call replayed as SSE
  (documented limitation).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

import httpx

from .keys import message_text


@dataclass(frozen=True)
class Provider:
    name: str
    kind: str  # "openai" | "anthropic"
    base_url: str
    api_key: str = ""
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.kind not in ("openai", "anthropic"):
            raise ValueError(f"unknown provider kind {self.kind!r}")


@dataclass(frozen=True)
class Route:
    prefix: str
    provider: Provider


@dataclass
class Router:
    default: Provider
    routes: List[Route] = field(default_factory=list)

    def pick(self, model: str) -> Provider:
        """Longest matching model prefix wins; otherwise the default provider."""
        best: Optional[Route] = None
        for r in self.routes:
            if model.startswith(r.prefix) and (best is None or len(r.prefix) > len(best.prefix)):
                best = r
        return best.provider if best else self.default

    @classmethod
    def from_config(cls, default_url: str, default_key: str, routes_json: str, timeout: float = 120.0) -> Router:
        """``routes_json``: ``[{"prefix": "claude-", "kind": "anthropic", "base_url": ..., "api_key_env": ...}]``.

        Keys are read from the environment variable named by ``api_key_env``,
        never from the JSON itself, so the routing config is safe to commit.
        """
        default = Provider("default", "openai", default_url.rstrip("/"), default_key, timeout)
        routes: List[Route] = []
        if routes_json.strip():
            data = json.loads(routes_json)
            if not isinstance(data, list):
                raise ValueError("SEMCACHE_ROUTES must be a JSON list")
            for i, item in enumerate(data):
                key_env = item.get("api_key_env", "")
                provider = Provider(
                    name=item.get("name", f"route{i}"),
                    kind=item.get("kind", "openai"),
                    base_url=str(item["base_url"]).rstrip("/"),
                    api_key=os.environ.get(key_env, "") if key_env else "",
                    timeout=float(item.get("timeout", timeout)),
                )
                routes.append(Route(prefix=str(item["prefix"]), provider=provider))
        return cls(default=default, routes=routes)


# ---- Anthropic translation ---------------------------------------------------------------------

_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


def to_anthropic(body: Mapping[str, Any]) -> Dict[str, Any]:
    system = "\n".join(message_text(m) for m in body.get("messages", []) if m.get("role") in ("system", "developer"))
    messages = [
        {"role": m["role"], "content": message_text(m)}
        for m in body.get("messages", [])
        if m.get("role") in ("user", "assistant")
    ]
    out: Dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 1024,
    }
    if system:
        out["system"] = system
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p")):
        if body.get(src) is not None:
            out[dst] = body[src]
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    return out


def from_anthropic(resp: Mapping[str, Any]) -> Dict[str, Any]:
    text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    usage = resp.get("usage") or {}
    pt, ct = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return {
        "id": resp.get("id", "chatcmpl-anthropic"),
        "object": "chat.completion",
        "created": 0,
        "model": resp.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _FINISH.get(str(resp.get("stop_reason")), "stop"),
            }
        ],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


# ---- client ------------------------------------------------------------------------------------


class UpstreamClient:
    """Thin async HTTP client over the router."""

    def __init__(self, router: Router, client: Optional[httpx.AsyncClient] = None) -> None:
        self.router = router
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self, provider: Provider, client_auth: Optional[str]) -> Dict[str, str]:
        if provider.kind == "anthropic":
            h = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
            if provider.api_key:
                h["x-api-key"] = provider.api_key
            return h
        h = {"content-type": "application/json"}
        if provider.api_key:
            h["authorization"] = f"Bearer {provider.api_key}"
        elif client_auth:
            # No key configured: pass the application's own key through untouched.
            h["authorization"] = client_auth
        return h

    async def complete(self, body: Mapping[str, Any], client_auth: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        provider = self.router.pick(str(body.get("model", "")))
        payload = dict(body)
        payload["stream"] = False
        if provider.kind == "anthropic":
            r = await self._client.post(
                f"{provider.base_url}/v1/messages",
                json=to_anthropic(payload),
                headers=self._headers(provider, client_auth),
                timeout=provider.timeout,
            )
            data = _json_or_error(r)
            return r.status_code, (from_anthropic(data) if r.status_code == 200 else data)
        r = await self._client.post(
            f"{provider.base_url}/chat/completions",
            json=payload,
            headers=self._headers(provider, client_auth),
            timeout=provider.timeout,
        )
        return r.status_code, _json_or_error(r)

    def is_native_stream(self, model: str) -> bool:
        return self.router.pick(model).kind == "openai"

    async def stream(
        self, body: Mapping[str, Any], client_auth: Optional[str] = None
    ) -> AsyncIterator[Tuple[int, bytes]]:
        """Yield ``(status, chunk)``. The first item carries the upstream status with an empty chunk."""
        provider = self.router.pick(str(body.get("model", "")))
        payload = dict(body)
        payload["stream"] = True
        async with self._client.stream(
            "POST",
            f"{provider.base_url}/chat/completions",
            json=payload,
            headers=self._headers(provider, client_auth),
            timeout=provider.timeout,
        ) as r:
            yield r.status_code, b""
            async for chunk in r.aiter_bytes():
                yield r.status_code, chunk

    async def models(self, client_auth: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        provider = self.router.default
        r = await self._client.get(f"{provider.base_url}/models", headers=self._headers(provider, client_auth))
        return r.status_code, _json_or_error(r)


def _json_or_error(r: httpx.Response) -> Dict[str, Any]:
    try:
        data = r.json()
    except ValueError:
        return {"error": {"message": r.text[:500], "type": "upstream_error"}}
    return data if isinstance(data, dict) else {"data": data}
