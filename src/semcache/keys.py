"""Cache keys: which requests are allowed to share an answer.

A semantic cache has two halves of a key:

* the **namespace**, an exact hash of everything that must match for two
  requests to be interchangeable: the model, the system prompt, the sampling
  parameters, tools/response format, and the conversation *before* the last
  user turn; and
* the **query text**, the last user message, which is embedded and compared by
  cosine similarity *within* a namespace only.

The namespace also includes the caller's **tenant** (see ``server.tenant_for``):
the proxy token's tenant, or a hash of the caller's ``Authorization`` header.
Callers with different keys therefore never share answers, and a cached answer
is only served to a caller presenting the same credential that created it.

Splitting it this way is what stops cross-contamination: "Summarise this" under
a legal-review system prompt can never be answered from the same words under a
marketing system prompt, and a temperature 0 answer is never served to a
temperature 1 request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Request fields that can change the answer and therefore belong in the namespace.
PARAM_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "seed",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "parallel_tool_calls",
)


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace, unicode kept."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def message_text(message: Mapping[str, Any]) -> str:
    """Plain text of a chat message; list-style content keeps only its text parts."""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def has_non_text_content(messages: Sequence[Mapping[str, Any]]) -> bool:
    """True when any message carries images, audio or other non-text parts."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") != "text":
                    return True
    return False


def last_user_index(messages: Sequence[Mapping[str, Any]]) -> int:
    """Index of the last ``user`` message, or -1 when there is none."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def system_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    """All system/developer messages joined, in order."""
    return "\n".join(message_text(m) for m in messages if m.get("role") in ("system", "developer"))


def system_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    """Short hash of the system prompt; the unit for system-prompt invalidation."""
    return sha256_hex(system_prompt(messages))[:16]


@dataclass(frozen=True)
class CacheKey:
    namespace: str
    query: str
    model: str
    system_hash: str


def build_key(body: Mapping[str, Any], tenant: str = "") -> Optional[CacheKey]:
    """Split a chat-completions body into (namespace, query text).

    ``tenant`` partitions the cache: two requests only share a namespace when
    their tenants are equal. Returns ``None`` when the request has no user
    message to key on.
    """
    messages = list(body.get("messages") or [])
    idx = last_user_index(messages)
    if idx < 0:
        return None
    model = str(body.get("model", ""))
    params: Dict[str, Any] = {k: body[k] for k in PARAM_FIELDS if body.get(k) is not None}
    # Everything except the last user turn (system prompt, earlier turns, and any
    # assistant/tool messages after it) is part of the exact namespace.
    context = [
        {"role": m.get("role"), "content": message_text(m), "name": m.get("name")}
        for i, m in enumerate(messages)
        if i != idx
    ]
    payload = {"tenant": tenant, "model": model, "params": params, "context": context}
    return CacheKey(
        namespace=sha256_hex(canonical_json(payload))[:32],
        query=message_text(messages[idx]),
        model=model,
        system_hash=system_hash(messages),
    )


def is_cacheable(body: Mapping[str, Any]) -> bool:
    """Requests we refuse to cache regardless of policy.

    ``n > 1`` asks for several different samples, and non-text inputs (images,
    audio) are not represented by the text embedding.
    """
    n = body.get("n")
    if n is not None and n != 1:
        return False
    return not has_non_text_content(list(body.get("messages") or []))
