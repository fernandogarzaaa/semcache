"""Server-sent-event helpers for streamed chat completions.

``StreamAccumulator`` watches an upstream SSE stream go by (the bytes are
forwarded to the client untouched) and rebuilds the full ``chat.completion``
object. It reports ``complete`` only when the stream ended with ``[DONE]``,
every choice finished with ``stop``, and nothing in it was unparseable or an
error, which is the only case in which a streamed answer is cached.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional


def estimate_tokens(text: str) -> int:
    """~4 characters per token; used only when the upstream stream carries no usage."""
    return max(1, (len(text) + 3) // 4) if text else 0


class StreamAccumulator:
    def __init__(self) -> None:
        self._buf = b""
        self.done = False
        self.broken = False
        self.has_tool_calls = False
        self.id: Optional[str] = None
        self.model: Optional[str] = None
        self.created: Optional[int] = None
        self.content: Dict[int, List[str]] = {}
        self.roles: Dict[int, str] = {}
        self.finish: Dict[int, Optional[str]] = {}
        self.usage: Optional[Dict[str, Any]] = None

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._line(line.strip())

    def close(self) -> None:
        if self._buf.strip():
            self._line(self._buf.strip())
        self._buf = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if data == b"[DONE]":
            self.done = True
            return
        try:
            obj = json.loads(data)
        except ValueError:
            self.broken = True
            return
        if not isinstance(obj, dict) or "error" in obj:
            self.broken = True
            return
        self.id = self.id or obj.get("id")
        self.model = self.model or obj.get("model")
        self.created = self.created or obj.get("created")
        if obj.get("usage"):
            self.usage = dict(obj["usage"])
        for choice in obj.get("choices") or []:
            idx = int(choice.get("index", 0))
            delta = choice.get("delta") or {}
            if delta.get("role"):
                self.roles[idx] = delta["role"]
            if delta.get("content"):
                self.content.setdefault(idx, []).append(delta["content"])
            if delta.get("tool_calls") or delta.get("function_call"):
                self.has_tool_calls = True
            self.finish.setdefault(idx, None)
            if choice.get("finish_reason"):
                self.finish[idx] = choice["finish_reason"]

    @property
    def complete(self) -> bool:
        return (
            self.done
            and not self.broken
            and not self.has_tool_calls
            and bool(self.finish)
            and all(reason == "stop" for reason in self.finish.values())
        )

    def to_completion(self, prompt_text: str = "") -> Dict[str, Any]:
        choices = []
        completion_text = ""
        for idx in sorted(self.finish):
            text = "".join(self.content.get(idx, []))
            completion_text += text
            choices.append(
                {
                    "index": idx,
                    "message": {"role": self.roles.get(idx, "assistant"), "content": text},
                    "finish_reason": self.finish[idx],
                }
            )
        usage = self.usage or {
            "prompt_tokens": estimate_tokens(prompt_text),
            "completion_tokens": estimate_tokens(completion_text),
            "total_tokens": estimate_tokens(prompt_text) + estimate_tokens(completion_text),
            "estimated": True,
        }
        return {
            "id": self.id or "chatcmpl-semcache",
            "object": "chat.completion",
            "created": self.created or int(time.time()),
            "model": self.model or "",
            "choices": choices,
            "usage": usage,
        }


def sse(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n\n"


def completion_to_sse(response: Mapping[str, Any]) -> Iterator[bytes]:
    """Replay a cached ``chat.completion`` as a stream: one delta per choice, usage, then ``[DONE]``."""
    base = {
        "id": response.get("id", "chatcmpl-semcache"),
        "object": "chat.completion.chunk",
        "created": response.get("created", int(time.time())),
        "model": response.get("model", ""),
    }
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        delta: Dict[str, Any] = {"role": message.get("role", "assistant"), "content": message.get("content") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(message["tool_calls"])]
        yield sse({**base, "choices": [{"index": choice.get("index", 0), "delta": delta, "finish_reason": None}]})
        yield sse(
            {
                **base,
                "choices": [
                    {"index": choice.get("index", 0), "delta": {}, "finish_reason": choice.get("finish_reason")}
                ],
            }
        )
    if response.get("usage"):
        yield sse({**base, "choices": [], "usage": response["usage"]})
    yield b"data: [DONE]\n\n"
