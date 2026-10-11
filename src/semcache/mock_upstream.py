"""A deterministic OpenAI-compatible (and minimal Anthropic) mock LLM for tests and load tests.

The answer, its token count and its latency are all derived from a hash of the
request, so a run is reproducible and costs nothing. Latency is *synthetic*
(``latency_ms`` + up to ``jitter_ms``); it stands in for a real provider's
response time and is reported as such.

Magic markers in the last user message, for failure-path tests:
``[fail]`` -> HTTP 500, ``[truncate]`` -> ``finish_reason: length``,
``[broken-stream]`` -> the stream ends without ``[DONE]``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .keys import last_user_index, message_text

_WORDS = (
    "the cache returns a stored answer when a new prompt means the same thing as an earlier one "
    "which saves the provider call and its latency while a miss forwards the request upstream"
).split()


def _digest(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def mock_answer(prompt: str, min_tokens: int = 120, spread: int = 280) -> tuple[str, int]:
    """Deterministic answer text and its completion token count (one token per word)."""
    h = _digest(prompt)
    n = min_tokens + h % max(1, spread)
    words = [f"[answer:{h % 10_000_000:07d}]"] + [_WORDS[(h + i) % len(_WORDS)] for i in range(n - 1)]
    return " ".join(words), n


def create_mock_app(latency_ms: float = 0.0, jitter_ms: float = 0.0, stream_chunks: int = 8) -> FastAPI:
    app = FastAPI(title="semcache mock upstream")
    app.state.calls = 0

    async def _sleep(seed: str) -> None:
        delay = latency_ms + (jitter_ms * ((_digest(seed + "j") % 1000) / 1000.0) if jitter_ms else 0.0)
        if delay > 0:
            await asyncio.sleep(delay / 1000.0)

    @app.get("/v1/models")
    async def models() -> Dict[str, Any]:
        return {"object": "list", "data": [{"id": "mock-gpt", "object": "model", "owned_by": "semcache"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        app.state.calls += 1
        body = await request.json()
        messages: List[Dict[str, Any]] = body.get("messages") or []
        idx = last_user_index(messages)
        prompt = message_text(messages[idx]) if idx >= 0 else ""
        model = str(body.get("model", "mock-gpt"))
        seed = json.dumps([model, messages, body.get("temperature")], sort_keys=True)
        if "[fail]" in prompt:
            return JSONResponse({"error": {"message": "mock failure", "type": "server_error"}}, status_code=500)
        text, completion_tokens = mock_answer(seed)
        prompt_tokens = max(1, sum(len(message_text(m).split()) for m in messages))
        finish = "length" if "[truncate]" in prompt else "stop"
        cid = f"chatcmpl-mock-{_digest(seed) % 10**12:012d}"
        created = int(time.time())
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if not body.get("stream"):
            await _sleep(seed)
            return JSONResponse(
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
                    ],
                    "usage": usage,
                }
            )

        broken = "[broken-stream]" in prompt
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        async def gen() -> AsyncIterator[bytes]:
            words = text.split(" ")
            per = max(1, len(words) // stream_chunks)
            base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
            first = {
                **base,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            yield b"data: " + json.dumps(first).encode() + b"\n\n"
            pieces = [" ".join(words[i : i + per]) for i in range(0, len(words), per)]
            for i, piece in enumerate(pieces):
                if latency_ms:
                    await asyncio.sleep(latency_ms / 1000.0 / len(pieces))
                content = piece if i == 0 else " " + piece
                chunk = {**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
                yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
            if broken:
                return
            end = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            yield b"data: " + json.dumps(end).encode() + b"\n\n"
            if include_usage:
                yield b"data: " + json.dumps({**base, "choices": [], "usage": usage}).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        """Just enough of the Anthropic Messages API to test the translation layer."""
        app.state.calls += 1
        body = await request.json()
        msgs = body.get("messages") or []
        prompt = str(msgs[-1].get("content", "")) if msgs else ""
        seed = json.dumps([body.get("model"), body.get("system"), msgs], sort_keys=True)
        text, out_tokens = mock_answer(seed)
        await _sleep(seed)
        return JSONResponse(
            {
                "id": f"msg_mock_{_digest(seed) % 10**10:010d}",
                "type": "message",
                "role": "assistant",
                "model": body.get("model"),
                "content": [{"type": "text", "text": text}],
                "stop_reason": "max_tokens" if "[truncate]" in prompt else "end_turn",
                "usage": {"input_tokens": max(1, len(prompt.split())), "output_tokens": out_tokens},
            }
        )

    return app
