"""The drop-in proxy: an OpenAI-compatible ``/v1/chat/completions`` with a semantic cache in front.

Point any OpenAI SDK at ``http://semcache:8080/v1`` instead of the provider and
nothing else changes. Every response carries ``X-Semcache: hit|miss|bypass``.

Request headers an application may send (all optional):

* ``X-Semcache-Bypass: 1``      skip the cache for this request (no lookup, no store)
* ``X-Semcache-Threshold: 0.9`` override the similarity threshold for this request
* ``X-Semcache-TTL: 600``       override the TTL (seconds) of the entry this request creates
* ``X-Semcache-Tags: a,b``      tag the stored entry, for invalidation by tag
* ``X-Semcache-Key: <token>``   proxy token, when ``SEMCACHE_PROXY_TOKENS`` is set
                                 (``Authorization: Bearer <token>`` also works)

Isolation: every cache entry belongs to a tenant. With proxy tokens, the
tenant is the token's tenant; otherwise it is a hash of the caller's
``Authorization`` header. The caller is identified *before* any cache lookup,
so a cached answer is never served to a caller that did not present the same
credential (``SEMCACHE_SHARED_CACHE=1`` opts out, for single-tenant use).
Admin endpoints (``/v1/cache/*``) require ``SEMCACHE_ADMIN_TOKEN`` and are
disabled when it is not set.
"""

from __future__ import annotations

import hmac
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import __version__
from .batching import EmbedBatcher
from .cache import Lookup, SemanticCache
from .config import Settings
from .embedders import Embedder, make_embedder
from .index import make_index
from .keys import build_key, is_cacheable, sha256_hex, system_prompt
from .metrics import Metrics
from .policy import (
    TASK_TYPES,
    TIER_NO_CACHE,
    AdaptiveThresholds,
    TTLPolicy,
    classify_freshness,
    classify_task,
    creative_bypass,
)
from .pricing import PriceTable
from .streaming import StreamAccumulator, completion_to_sse
from .tuner import default_thresholds, sweep_history, sweep_pairs
from .upstream import Router, UpstreamClient


@dataclass
class State:
    settings: Settings
    cache: SemanticCache
    thresholds: AdaptiveThresholds
    ttl: TTLPolicy
    prices: PriceTable
    upstream: UpstreamClient
    metrics: Metrics
    batcher: EmbedBatcher
    saved_usd: float = 0.0
    upstream_usd: float = 0.0


class InvalidateRequest(BaseModel):
    model: Optional[str] = None
    system_prompt: Optional[str] = Field(default=None, description="Full system prompt text; hashed server-side")
    system_hash: Optional[str] = None
    tag: Optional[str] = None
    prefix: Optional[str] = None
    all: bool = False


class FeedbackRequest(BaseModel):
    entry_id: int
    similarity: float
    correct: bool
    task: Optional[str] = None


class PairIn(BaseModel):
    similarity: float
    same_intent: bool


class TuneRequest(BaseModel):
    pairs: Optional[List[PairIn]] = None
    thresholds: Optional[List[float]] = None


def build_state(
    settings: Settings,
    *,
    embedder: Optional[Embedder] = None,
    upstream: Optional[UpstreamClient] = None,
) -> State:
    emb = embedder or make_embedder(settings.embedder, settings.embedder_model, settings.hashing_dim)
    metrics = Metrics()
    cache = SemanticCache(
        emb,
        make_index(settings.index, emb.dim),
        max_entries=settings.max_entries,
        normalize=settings.normalize,
        near_miss_window=settings.near_miss_window,
        on_evict=lambda reason, n: metrics.evictions.labels(reason).inc(n),
    )
    metrics.capacity.set(settings.max_entries)
    thresholds = AdaptiveThresholds(base=dict(settings.thresholds), adaptive=settings.adaptive)
    router = Router.from_config(
        settings.upstream_url, settings.upstream_api_key, settings.routes_json, settings.upstream_timeout
    )
    return State(
        settings=settings,
        cache=cache,
        thresholds=thresholds,
        ttl=TTLPolicy(short=settings.ttl_short, default=settings.ttl_default, long=settings.ttl_long),
        prices=PriceTable.from_json(settings.prices_json),
        upstream=upstream or UpstreamClient(router),
        metrics=metrics,
        batcher=EmbedBatcher(
            lambda texts: cache.embedder.embed([cache.embedding_text(t) for t in texts]),
            max_batch=settings.embed_batch,
            max_wait_ms=settings.embed_wait_ms,
        ),
    )


def _float_header(value: Optional[str], name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise HTTPException(400, f"{name} must be a number") from None


def _bearer(authorization: Optional[str]) -> str:
    if authorization and authorization[:7].lower() == "bearer ":
        return authorization[7:].strip()
    return ""


def _token_equal(a: str, b: str) -> bool:
    return bool(a) and bool(b) and hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


@dataclass(frozen=True)
class Caller:
    tenant: str
    upstream_auth: Optional[str]  # what may be forwarded upstream as Authorization


def tenant_for(settings: Settings, authorization: Optional[str], semcache_key: Optional[str]) -> Caller:
    """Identify the caller before touching the cache.

    * Proxy tokens configured: ``X-Semcache-Key`` or ``Authorization: Bearer``
      must carry a known token (else 401). A token sent via Authorization is
      never forwarded upstream.
    * Otherwise: the tenant is a hash of the caller's Authorization header
      (callers without one share the ``anon`` partition).
    """
    if settings.proxy_tokens:
        presented = (semcache_key or "").strip()
        via_auth = False
        if not presented:
            presented, via_auth = _bearer(authorization), True
        tenant = None
        for token, name in settings.proxy_tokens.items():
            if _token_equal(presented, token):
                tenant = name
        if tenant is None:
            raise HTTPException(401, "a valid semcache proxy token is required", headers={"WWW-Authenticate": "Bearer"})
        return Caller(
            tenant="" if settings.shared_cache else "tenant:" + tenant,
            upstream_auth=None if via_auth else authorization,
        )
    if settings.shared_cache:
        return Caller(tenant="", upstream_auth=authorization)
    if not authorization:
        return Caller(tenant="anon", upstream_auth=None)
    return Caller(tenant="auth:" + sha256_hex(authorization)[:32], upstream_auth=authorization)


def create_app(
    settings: Optional[Settings] = None,
    *,
    embedder: Optional[Embedder] = None,
    upstream: Optional[UpstreamClient] = None,
) -> FastAPI:
    st = build_state(settings or Settings.from_env(), embedder=embedder, upstream=upstream)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await st.batcher.aclose()
        await st.upstream.aclose()

    app = FastAPI(
        title="semcache",
        version=__version__,
        description="Semantic cache for OpenAI-compatible LLM APIs",
        lifespan=lifespan,
    )
    app.state.semcache = st
    m = st.metrics

    def require_admin(authorization: Optional[str] = Header(default=None)) -> None:
        token = st.settings.admin_token
        if not token:
            if st.settings.admin_open:
                return
            raise HTTPException(403, "admin API disabled: set SEMCACHE_ADMIN_TOKEN")
        if not _token_equal(_bearer(authorization), token):
            raise HTTPException(401, "admin token required")

    def _refresh_gauges() -> None:
        m.entries.set(len(st.cache))
        for task, snap in st.thresholds.snapshot().items():
            m.threshold.labels(task).set(snap["threshold"])

    def _record_similarity(lk: Lookup) -> None:
        if lk.similarity < 0:
            return
        outcome = "hit" if lk.hit else ("near_miss" if lk.near_miss else "miss")
        m.similarity.labels(outcome).observe(lk.similarity)

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "version": __version__, "entries": len(st.cache)}

    @app.get("/metrics")
    async def metrics() -> Response:
        _refresh_gauges()
        return Response(generate_latest(m.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def models(
        authorization: Optional[str] = Header(default=None),
        x_semcache_key: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        caller = tenant_for(st.settings, authorization, x_semcache_key)
        status, data = await st.upstream.models(caller.upstream_auth)
        return JSONResponse(data, status_code=status)

    @app.post("/v1/chat/completions")
    async def chat(
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_semcache_bypass: Optional[str] = Header(default=None),
        x_semcache_threshold: Optional[str] = Header(default=None),
        x_semcache_ttl: Optional[str] = Header(default=None),
        x_semcache_tags: Optional[str] = Header(default=None),
        x_semcache_key: Optional[str] = Header(default=None),
    ) -> Response:
        started = time.perf_counter()
        # Authenticate / identify the caller before any cache access.
        caller = tenant_for(st.settings, authorization, x_semcache_key)
        authorization = caller.upstream_auth
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "request body must be JSON") from None
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            raise HTTPException(400, "expected a chat completions body with a 'messages' list")
        model = str(body.get("model", ""))
        stream = bool(body.get("stream"))
        threshold_override = _float_header(x_semcache_threshold, "X-Semcache-Threshold")
        ttl_override = _float_header(x_semcache_ttl, "X-Semcache-TTL")
        tags = [t.strip() for t in (x_semcache_tags or "").split(",") if t.strip()]

        key = build_key(body, tenant=caller.tenant)
        reason = ""
        if not st.settings.enabled:
            reason = "disabled"
        elif (x_semcache_bypass or "").lower() in ("1", "true", "yes"):
            reason = "header"
        elif key is None:
            reason = "no_user_message"
        elif not is_cacheable(body):
            reason = "uncacheable_request"
        tier = task = ""
        if key is not None and not reason:
            normalized = st.cache.embedding_text(key.query)
            tier = classify_freshness(normalized)
            task = classify_task(normalized, system_prompt(body["messages"]))
            if tier == TIER_NO_CACHE and ttl_override is None:
                reason = "live_data"
            elif creative_bypass(task, body.get("temperature")):
                reason = "creative_high_temperature"

        if reason or key is None:
            return await _forward(
                body,
                stream,
                authorization,
                {"X-Semcache": "bypass", "X-Semcache-Reason": reason},
                started,
                "bypass",
                None,
            )

        threshold = threshold_override if threshold_override is not None else st.thresholds.threshold(task)
        vector = await st.batcher.embed(key.query)
        lk = st.cache.lookup(key, threshold, vector)
        _record_similarity(lk)
        sim_header = f"{lk.similarity:.4f}" if lk.similarity >= 0 else "none"
        if lk.hit and lk.entry is not None:
            entry = lk.entry
            saved = st.prices.cost(entry.model, entry.prompt_tokens, entry.completion_tokens)
            st.saved_usd += saved
            m.saved_usd.labels(model).inc(saved)
            m.saved_tokens.labels(model, "prompt").inc(entry.prompt_tokens)
            m.saved_tokens.labels(model, "completion").inc(entry.completion_tokens)
            m.requests.labels(model, "hit").inc()
            headers = {
                "X-Semcache": "hit",
                "X-Semcache-Similarity": sim_header,
                "X-Semcache-Threshold": f"{threshold:.4f}",
                "X-Semcache-Entry": str(entry.id),
                "X-Semcache-Task": task,
                "X-Semcache-Saved-USD": f"{saved:.6f}",
            }
            m.latency.labels("hit").observe(time.perf_counter() - started)
            if stream:
                return StreamingResponse(
                    _iter(completion_to_sse(entry.response)), media_type="text/event-stream", headers=headers
                )
            return JSONResponse(entry.response, headers=headers)

        ttl = ttl_override if ttl_override is not None else st.ttl.ttl_for(tier)
        headers = {
            "X-Semcache": "miss",
            "X-Semcache-Similarity": sim_header,
            "X-Semcache-Threshold": f"{threshold:.4f}",
            "X-Semcache-Task": task,
            "X-Semcache-Tier": tier,
        }

        def on_complete(response: Dict[str, Any]) -> None:
            st.cache.store(key, response, ttl=ttl, vector=vector, tier=tier, task=task, tags=tags)

        return await _forward(body, stream, authorization, headers, started, "miss", on_complete)

    async def _iter(chunks: Any) -> AsyncIterator[bytes]:
        for c in chunks:
            yield c

    def _account_upstream(model: str, response: Dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        cost = st.prices.cost(model, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))
        st.upstream_usd += cost
        m.upstream_usd.labels(model).inc(cost)

    async def _forward(
        body: Dict[str, Any],
        stream: bool,
        authorization: Optional[str],
        headers: Dict[str, str],
        started: float,
        result: str,
        on_complete: Optional[Any],
    ) -> Response:
        model = str(body.get("model", ""))
        m.requests.labels(model, result).inc()
        if stream and st.upstream.is_native_stream(model):
            return await _forward_stream(body, authorization, headers, started, result, on_complete)
        try:
            status, data = await st.upstream.complete(body, authorization)
        except Exception as exc:  # network errors, timeouts
            m.upstream_errors.labels("transport").inc()
            m.latency.labels(result).observe(time.perf_counter() - started)
            return JSONResponse(
                {"error": {"message": f"upstream unreachable: {type(exc).__name__}", "type": "upstream_error"}},
                status_code=502,
                headers=headers,
            )
        if status == 200:
            _account_upstream(model, data)
            if on_complete is not None:
                if _complete_response(data):
                    on_complete(data)
                else:
                    m.upstream_errors.labels("incomplete").inc()
        else:
            m.upstream_errors.labels(f"http_{status}").inc()
        m.latency.labels(result).observe(time.perf_counter() - started)
        if stream and status == 200:
            return StreamingResponse(_iter(completion_to_sse(data)), media_type="text/event-stream", headers=headers)
        return JSONResponse(data, status_code=status, headers=headers)

    async def _forward_stream(
        body: Dict[str, Any],
        authorization: Optional[str],
        headers: Dict[str, str],
        started: float,
        result: str,
        on_complete: Optional[Any],
    ) -> Response:
        gen = st.upstream.stream(body, authorization)
        try:
            status, _ = await gen.__anext__()
        except Exception as exc:
            m.upstream_errors.labels("transport").inc()
            return JSONResponse(
                {"error": {"message": f"upstream unreachable: {type(exc).__name__}", "type": "upstream_error"}},
                status_code=502,
                headers=headers,
            )
        if status != 200:
            raw = b"".join([chunk async for _, chunk in gen])
            m.upstream_errors.labels(f"http_{status}").inc()
            m.latency.labels(result).observe(time.perf_counter() - started)
            return Response(raw, status_code=status, media_type="application/json", headers=headers)

        acc = StreamAccumulator()
        prompt_text = "\n".join(str(msg.get("content", "")) for msg in body.get("messages", []))

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for _, chunk in gen:
                    acc.feed(chunk)
                    yield chunk
            finally:
                acc.close()
                m.latency.labels(result).observe(time.perf_counter() - started)
                if acc.done and not acc.broken:
                    completion = acc.to_completion(prompt_text)
                    _account_upstream(str(body.get("model", "")), completion)
                    if on_complete is not None and acc.complete:
                        on_complete(completion)
                    elif on_complete is not None:
                        m.upstream_errors.labels("incomplete").inc()
                elif on_complete is not None:
                    m.upstream_errors.labels("stream_aborted").inc()

        return StreamingResponse(relay(), media_type="text/event-stream", headers=headers)

    # ---- admin / analytics -------------------------------------------------------------------
    @app.get("/v1/cache/stats", dependencies=[Depends(require_admin)])
    async def stats() -> Dict[str, Any]:
        _refresh_gauges()
        return {
            **st.cache.stats(),
            "thresholds": st.thresholds.snapshot(),
            "estimated_saved_usd": round(st.saved_usd, 6),
            "estimated_upstream_usd": round(st.upstream_usd, 6),
            "embed_batches": st.batcher.batches,
            "embed_mean_batch": round(st.batcher.items / st.batcher.batches, 2) if st.batcher.batches else 0.0,
        }

    @app.get("/v1/cache/entries", dependencies=[Depends(require_admin)])
    async def entries(limit: int = 50) -> Dict[str, Any]:
        items = st.cache.entries()[::-1][: max(0, limit)]
        return {"entries": [e.public() for e in items], "total": len(st.cache)}

    @app.post("/v1/cache/invalidate", dependencies=[Depends(require_admin)])
    async def invalidate(req: InvalidateRequest) -> Dict[str, Any]:
        if req.all:
            return {"removed": st.cache.clear()}
        sys_hash = req.system_hash
        if req.system_prompt is not None:
            sys_hash = sha256_hex(req.system_prompt)[:16]
        try:
            removed = st.cache.invalidate(model=req.model, system_hash=sys_hash, tag=req.tag, prefix=req.prefix)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"removed": removed}

    @app.post("/v1/cache/feedback", dependencies=[Depends(require_admin)])
    async def feedback(req: FeedbackRequest) -> Dict[str, Any]:
        entry = st.cache.get(req.entry_id)
        task = req.task or (entry.task if entry else None)
        if task is None:
            raise HTTPException(404, "unknown entry; pass 'task' explicitly")
        if task not in TASK_TYPES:
            raise HTTPException(400, f"task must be one of {list(TASK_TYPES)}")
        new = st.thresholds.feedback(task, req.similarity, req.correct)
        m.threshold.labels(task).set(new)
        return {"task": task, "threshold": new}

    @app.get("/v1/cache/near-misses", dependencies=[Depends(require_admin)])
    async def near_misses(limit: int = 50) -> Dict[str, Any]:
        return {"near_misses": [n.as_dict() for n in st.cache.near_misses(limit)]}

    @app.post("/v1/cache/tune", dependencies=[Depends(require_admin)])
    async def tune(req: TuneRequest) -> Dict[str, Any]:
        thresholds = req.thresholds or default_thresholds()
        if req.pairs:
            points = sweep_pairs([p.similarity for p in req.pairs], [p.same_intent for p in req.pairs], thresholds)
            return {"source": "pairs", "points": [p.as_dict() for p in points]}
        return {"source": "history", "points": sweep_history(st.cache.history(), thresholds)}

    @app.get("/v1/cache/config", dependencies=[Depends(require_admin)])
    async def config() -> Dict[str, Any]:
        return st.settings.redacted()

    return app


def _complete_response(data: Dict[str, Any]) -> bool:
    """Only cache whole answers: every choice stopped normally (or asked for tools)."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    return all(c.get("finish_reason") in ("stop", "tool_calls") for c in choices)
