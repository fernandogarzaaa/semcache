"""Command line: ``semcache serve | mock-upstream | tune | stats | version``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .config import Settings
    from .server import create_app

    settings = Settings.from_env()
    if args.upstream:
        settings.upstream_url = args.upstream
    if args.embedder:
        settings.embedder = args.embedder
    print(json.dumps({"semcache": __version__, **settings.redacted()}, default=str), file=sys.stderr)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level=args.log_level, access_log=False)
    return 0


def _cmd_mock(args: argparse.Namespace) -> int:
    import uvicorn

    from .mock_upstream import create_mock_app

    app = create_mock_app(latency_ms=args.latency_ms, jitter_ms=args.jitter_ms)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, access_log=False)
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    """Offline threshold sweep: replay a labelled JSONL workload at each threshold."""
    from .embedders import make_embedder
    from .normalize import normalize_query
    from .tuner import default_thresholds, pick_threshold, replay

    with open(args.workload, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    emb = make_embedder(args.embedder, args.model)
    texts = [normalize_query(r["prompt"]) if not args.no_normalize else r["prompt"] for r in rows]
    vecs = emb.embed(texts)
    intents = [r.get("intent") for r in rows]
    results = [replay(vecs, intents, t) for t in default_thresholds(args.lo, args.hi, args.step)]
    best = pick_threshold(results, args.max_false_hit_rate)
    out = {
        "embedder": emb.name,
        "requests": len(rows),
        "max_false_hit_rate": args.max_false_hit_rate,
        "recommended": best.as_dict() if best else None,
        "sweep": [r.as_dict() for r in results],
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    import httpx

    headers = {"authorization": f"Bearer {args.token}"} if args.token else {}
    r = httpx.get(args.url.rstrip("/") + "/v1/cache/stats", headers=headers, timeout=10)
    print(json.dumps(r.json(), indent=2))
    return 0 if r.status_code == 200 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semcache", description=__doc__)
    p.add_argument("--version", action="version", version=f"semcache {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the caching proxy")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--upstream", default="", help="OpenAI-compatible base URL (overrides SEMCACHE_UPSTREAM_URL)")
    s.add_argument("--embedder", default="", help="hashing | minilm (overrides SEMCACHE_EMBEDDER)")
    s.add_argument("--log-level", default="warning")
    s.set_defaults(func=_cmd_serve)

    mk = sub.add_parser("mock-upstream", help="run the deterministic mock LLM")
    mk.add_argument("--host", default="127.0.0.1")
    mk.add_argument("--port", type=int, default=9000)
    mk.add_argument("--latency-ms", type=float, default=0.0)
    mk.add_argument("--jitter-ms", type=float, default=0.0)
    mk.add_argument("--log-level", default="warning")
    mk.set_defaults(func=_cmd_mock)

    t = sub.add_parser("tune", help="offline threshold sweep over a labelled workload (JSONL: prompt, intent)")
    t.add_argument("workload")
    t.add_argument("--embedder", default="minilm")
    t.add_argument("--model", default="")
    t.add_argument("--lo", type=float, default=0.70)
    t.add_argument("--hi", type=float, default=0.99)
    t.add_argument("--step", type=float, default=0.01)
    t.add_argument("--max-false-hit-rate", type=float, default=0.01)
    t.add_argument("--no-normalize", action="store_true")
    t.set_defaults(func=_cmd_tune)

    st = sub.add_parser("stats", help="print a running proxy's cache stats")
    st.add_argument("--url", default="http://localhost:8080")
    st.add_argument("--token", default="")
    st.set_defaults(func=_cmd_stats)

    v = sub.add_parser("version", help="print the version")
    v.set_defaults(func=lambda _a: print(__version__) or 0)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
