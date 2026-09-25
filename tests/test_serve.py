"""The web interface's server: validation, answering, and what it will serve."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

from indicrag.index.lexical import LexicalIndex
from indicrag.models import Passage
from indicrag.pipeline import Retrievers
from indicrag.serve import System, handle_ask, make_handler

CORPUS = [
    Passage(passage_id="b#p0", doc_id="s1-en", scheme="s1", lang="en",
            text="Students receive Rs. 12,000 per annum under the merit scholarship. " * 3,
            section_path="Eligibility", token_count=30),
]


def _system() -> System:
    return System(passages=CORPUS, retrievers=Retrievers(lexical=LexicalIndex.build(CORPUS)))


def test_handle_ask_validates_before_answering():
    s = _system()
    assert handle_ask({"query": ""}, s)[0] == 400
    assert handle_ask({"query": "x" * 501}, s)[0] == 400
    assert handle_ask({"query": "ok", "method": "rm -rf"}, s)[0] == 400
    assert handle_ask({"query": "ok", "bm25_floor": "high"}, s)[0] == 400
    assert handle_ask(["not", "an", "object"], s)[0] == 400


def test_handle_ask_answers_with_the_bm25_floor_off():
    status, body = handle_ask(
        {"query": "How much do students receive per annum?", "method": "bm25", "bm25_floor": 0}, _system()
    )
    assert status == 200
    assert body["answerability"] == "ANSWERABLE"
    assert body["citations"][0]["passage_id"] == "b#p0"


def _serve(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<p>hi</p>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "1-x.json").write_text(json.dumps({"query": "q"}), encoding="utf-8")
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(_system(), web, demo))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def test_the_server_serves_web_and_api_and_nothing_outside_web(tmp_path):
    httpd, base = _serve(tmp_path)
    try:
        assert _get(base + "/")[1] == b"<p>hi</p>"
        assert json.loads(_get(base + "/api/demo")[1]) == [{"query": "q"}]
        assert json.loads(_get(base + "/api/health")[1])["passages"] == 1
        assert _get(base + "/../secret.txt")[0] == 404
        assert _get(base + "/%2e%2e/secret.txt")[0] == 404
        req = urllib.request.Request(
            base + "/api/ask", method="POST", headers={"Content-Type": "application/json"},
            data=json.dumps({"query": "How much do students receive?", "bm25_floor": 0,
                             "method": "bm25"}).encode(),
        )
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["answerability"] == "ANSWERABLE"
    finally:
        httpd.shutdown()


def test_the_rate_limiter_enforces_minute_day_and_hosted_budget():
    from indicrag.serve import RateLimiter

    now = [1000.0]
    lim = RateLimiter(per_min=2, per_day=3, api_daily_cap=1, clock=lambda: now[0])
    assert lim.check("a", hosted=False) is None
    assert lim.check("a", hosted=False) is None
    assert "a minute" in lim.check("a", hosted=False)
    assert lim.check("b", hosted=False) is None          # limits are per visitor
    now[0] += 61
    assert lim.check("a", hosted=False) is None           # the minute window slides
    now[0] += 61
    assert "daily limit" in lim.check("a", hosted=False)
    assert lim.check("c", hosted=True) is None
    assert "budget" in lim.check("d", hosted=True)         # the hosted budget is global
    assert lim.check("d", hosted=False) is None            # the extracted answer still works


def test_a_limited_visitor_gets_429_and_proxy_headers_are_honoured_only_when_trusted(tmp_path):
    from indicrag.serve import RateLimiter

    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<p>hi</p>", encoding="utf-8")

    def post(base, ip):
        req = urllib.request.Request(
            base + "/api/ask", method="POST",
            headers={"Content-Type": "application/json", "X-Forwarded-For": ip},
            data=json.dumps({"query": "How much do students receive?", "method": "bm25",
                             "bm25_floor": 0}).encode(),
        )
        try:
            with urllib.request.urlopen(req) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    for trusted, expect_second_ip in ((True, 200), (False, 429)):
        handler = make_handler(_system(), web, tmp_path, limiter=RateLimiter(per_min=1), trust_proxy=trusted)
        httpd = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            assert post(base, "1.1.1.1") == 200
            assert post(base, "1.1.1.1") == 429
            # A different forwarded address is a different visitor only behind a trusted proxy;
            # otherwise the header is ignored and the socket address (still limited) is used.
            assert post(base, "2.2.2.2") == expect_second_ip
        finally:
            httpd.shutdown()


def test_responses_carry_security_headers(tmp_path):
    httpd, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/") as r:
            assert r.headers["X-Content-Type-Options"] == "nosniff"
            assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    finally:
        httpd.shutdown()
