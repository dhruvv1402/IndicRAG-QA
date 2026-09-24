"""A small local server for the web interface in `web/`.

`indicrag ask` reloads the dense encoder on every call, which is most of its
~50 s per query. The server loads the system once and keeps it warm, so the
web Playground answers in about a second (extractive) or a minute (with a
generator). It is a demo surface, not a deployment: standard library only,
bound to localhost, one request at a time.

    indicrag serve                      # http://127.0.0.1:8000
    indicrag serve --gguf <model.gguf>  # generated rather than extracted answers

Routes:

    GET  /                 web/index.html and its static files
    GET  /api/health       {"ok": true, "passages": 694, "generator": "..."}
    GET  /api/demo         the rehearsed outputs in docs/demo/, for the page's examples
    POST /api/ask          {"query": "...", "method": "hybrid", "bm25_floor": 18.08}
"""

from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .models import Passage

#: Retrieval methods the page may request; anything else is refused rather
#: than passed through to `retrieve()`.
METHODS = ("hybrid", "hybrid-rrf", "hybrid-weighted", "bm25", "tfidf", "dense")
#: The dev-fitted BM25 abstention floor (paper §VI-H).
DEFAULT_FLOOR = 18.08
MAX_QUERY_CHARS = 500
#: The only files outside web/ the page links to. An allowlist, not a directory.
PAPER_FILES = ("IndicRAG-QA.pdf", "IndicRAG-QA-IEEE.pdf", "IndicRAG-QA-slides.pptx")


@dataclass
class System:
    passages: list[Passage]
    retrievers: Any
    provider: Any = None
    notes: list[str] = field(default_factory=list)


def load_system(passages: list[Passage], *, gguf: str = "", say=None) -> System:
    """Load the lexical index, the primary dense encoder and, optionally, a generator.

    Shared by `ask` and `serve`, so both run the same system. A missing dense
    index is reported, not fatal: `hybrid` then degrades to lexical retrieval,
    and the caller is told so rather than left to notice the scores.
    """
    from .config import get_settings
    from .index.lexical import LexicalIndex
    from .pipeline import Retrievers

    say = say or (lambda _m: None)
    cfg = get_settings()
    retrievers = Retrievers(lexical=LexicalIndex.load(cfg.lex_dir))
    notes: list[str] = []
    try:
        from .index.dense import DenseIndex, Encoder
        from .index.encoders import get

        spec = get(cfg.encoder_primary)
        retrievers.dense = DenseIndex.load(spec, cfg.emb_dir, passages)
        retrievers.encoder = Encoder(spec)
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        note = f"dense index unavailable ({type(exc).__name__}: {exc}); hybrid falls back to lexical"
        notes.append(note)
        say(f"# {note}")

    provider = None
    if gguf:
        from .rag.providers import LlamaCppProvider

        provider = LlamaCppProvider(gguf, n_ctx=cfg.llm_n_ctx, n_threads=cfg.llm_n_threads)
    return System(passages=passages, retrievers=retrievers, provider=provider, notes=notes)


def handle_ask(payload: Any, system: System) -> tuple[int, dict]:
    """Validate a request and answer it. Returns (HTTP status, JSON body)."""
    from .pipeline import answer_query

    if not isinstance(payload, dict):
        return 400, {"error": "expected a JSON object"}
    query = str(payload.get("query") or "").strip()
    if not query:
        return 400, {"error": "query is empty"}
    if len(query) > MAX_QUERY_CHARS:
        return 400, {"error": f"query is longer than {MAX_QUERY_CHARS} characters"}
    method = str(payload.get("method") or "hybrid")
    if method not in METHODS:
        return 400, {"error": f"method must be one of {', '.join(METHODS)}"}
    try:
        floor = float(payload.get("bm25_floor", DEFAULT_FLOOR))
        k = int(payload.get("k", 5))
    except (TypeError, ValueError):
        return 400, {"error": "bm25_floor and k must be numbers"}
    k = max(1, min(k, 10))

    result = answer_query(
        query, system.passages, retrievers=system.retrievers, method=method, k=k,
        bm25_floor=max(0.0, floor), provider=system.provider,
    )
    return 200, result.as_dict()


def _demo_outputs(demo_dir: Path) -> list[dict]:
    out = []
    for path in sorted(demo_dir.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def make_handler(system: System, web_dir: Path, demo_dir: Path, paper_dir: Path | None = None):
    web_root = web_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "indicrag"

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 -- http.server's naming
            path = self.path.split("?", 1)[0]
            if path == "/api/health":
                gen = getattr(system.provider, "name", None) or "extractive"
                self._json(200, {"ok": True, "passages": len(system.passages),
                                 "generator": gen, "notes": system.notes})
                return
            if path == "/api/demo":
                self._json(200, _demo_outputs(demo_dir))
                return
            if paper_dir is not None and path.startswith("/paper/"):
                name = path[len("/paper/"):]
                target = paper_dir / name
                if name in PAPER_FILES and target.is_file():
                    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
                    self._send(200, target.read_bytes(), ctype)
                else:
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (web_root / rel).resolve()
            # Serve files under web/ only; a path that resolves outside it is a 404.
            inside = web_root in target.parents
            if not inside or not target.is_file():
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/ask":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > 10_000:
                self._json(413, {"error": "request too large"})
                return
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "body is not JSON"})
                return
            status, body = handle_ask(payload, system)
            self._json(status, body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"  {self.command} {self.path.split('?', 1)[0]}  {args[1] if len(args) > 1 else ''}")

    return Handler


def serve(
    system: System, *, host: str, port: int, web_dir: Path, demo_dir: Path, paper_dir: Path
) -> None:
    httpd = HTTPServer((host, port), make_handler(system, web_dir, demo_dir, paper_dir))
    print(f"IndicRAG-QA web interface on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
