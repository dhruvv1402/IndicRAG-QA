"""Hosted generators (Groq, Gemini) through the OpenAI-compatible provider.

No network: every test hands the provider a fake opener, so what is checked is
the request it builds, how it reads the reply, and how it fails.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from indicrag.models import Passage
from indicrag.query.langid import classify
from indicrag.rag.prompts import REFUSAL
from indicrag.rag.providers import API_PRESETS, APIError, OpenAICompatProvider, grounded_answer

PASSAGES = [
    Passage(passage_id="b#p0", doc_id="s1-en", scheme="s1", lang="en",
            text="Students receive Rs. 12,000 per annum under the merit scholarship.",
            section_path="Eligibility", token_count=10),
]


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _reply(content: str) -> _Resp:
    return _Resp(json.dumps({"choices": [{"message": {"content": content}}]}).encode())


class FakeOpener:
    """Records requests; returns queued replies or raises queued errors."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _http_error(code, body=b"{}", retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("https://x/chat/completions", code, "err", headers, io.BytesIO(body))


def _provider(opener, **kw):
    return OpenAICompatProvider(base_url="https://api.example/v1/", api_key="sk-test", model="m",
                                label="groq", opener=opener, sleep=lambda _s: None, **kw)


def test_the_request_is_an_openai_chat_completion_with_the_key_as_bearer():
    opener = FakeOpener(_reply('{"answerable": true, "answer": "Rs. 12,000", "citation": "b#p0"}'))
    _provider(opener).complete("prompt")
    req = opener.requests[0]
    assert req.full_url == "https://api.example/v1/chat/completions"
    assert req.get_header("Authorization") == "Bearer sk-test"
    body = json.loads(req.data)
    assert body["model"] == "m" and body["temperature"] == 0
    assert body["messages"] == [{"role": "user", "content": "prompt"}]


def test_a_grounded_answer_goes_through_the_shared_parser():
    opener = FakeOpener(_reply(
        'Sure! {"answerable": true, "answer": "Rs. 12,000 per annum", "citation": "b#p0", "confidence": 0.9}'
    ))
    p = _provider(opener)
    out = grounded_answer(p, "How much do students receive?", PASSAGES,
                          lang=classify("How much do students receive?"))
    assert out.answerable and out.text == "Rs. 12,000 per annum"
    assert out.citations == ["b#p0"]
    assert p.name == "groq:m"


def test_a_declined_or_unparseable_reply_becomes_the_refusal_string():
    lang = classify("What is the interest rate in Nagaland?")
    declined = _provider(FakeOpener(_reply('{"answerable": false, "answer": "", "citation": ""}')))
    assert declined.answer("q", PASSAGES, lang=lang).text == REFUSAL
    garbled = _provider(FakeOpener(_reply("I cannot help with that.")))
    out = garbled.answer("q", PASSAGES, lang=lang)
    assert out.text == REFUSAL and garbled.parse_failures == 1


def test_rate_limits_are_retried_and_then_succeed():
    sleeps = []
    opener = FakeOpener(_http_error(429, retry_after="1"), _http_error(503),
                        _reply('{"answerable": true, "answer": "x", "citation": "b#p0"}'))
    p = OpenAICompatProvider(base_url="https://api.example/v1", api_key="k", model="m",
                             opener=opener, sleep=sleeps.append)
    assert p.complete("prompt")
    assert sleeps == [1.0, 2.0]  # Retry-After honoured, then exponential backoff


def test_a_hard_error_names_the_provider_and_the_message_but_never_the_key():
    body = json.dumps({"error": {"message": "Invalid API Key"}}).encode()
    p = _provider(FakeOpener(_http_error(401, body)))
    with pytest.raises(APIError) as err:
        p.complete("prompt")
    assert "HTTP 401" in str(err.value) and "Invalid API Key" in str(err.value)
    assert "sk-test" not in str(err.value)


def test_presets_read_their_own_key_variable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("INDICRAG_LLM_API_KEY", raising=False)
    with pytest.raises(APIError, match="GROQ_API_KEY"):
        OpenAICompatProvider.from_preset("groq")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    gem = OpenAICompatProvider.from_preset("gemini", model="gemini-custom")
    assert gem.base_url == API_PRESETS["gemini"]["base_url"]
    assert gem.name == "gemini:gemini-custom"
    with pytest.raises(APIError, match="unknown API"):
        OpenAICompatProvider.from_preset("openrouter")


def test_the_server_offers_each_loaded_generator_by_name():
    from indicrag.index.lexical import LexicalIndex
    from indicrag.pipeline import Retrievers
    from indicrag.serve import System, handle_ask

    hosted = _provider(FakeOpener(_reply('{"answerable": true, "answer": "Rs. 12,000", "citation": "b#p0"}')))
    system = System(passages=PASSAGES, retrievers=Retrievers(lexical=LexicalIndex.build(PASSAGES)),
                    provider=hosted, providers={"extractive": None, hosted.name: hosted})
    q = {"query": "How much do students receive per annum?", "method": "bm25", "bm25_floor": 0}
    status, body = handle_ask(q, system)
    assert status == 200 and body["model"] == "groq:m" and body["answer"] == "Rs. 12,000"
    status, body = handle_ask({**q, "generator": "extractive"}, system)
    assert status == 200 and body["model"] == "extractive"
    assert handle_ask({**q, "generator": "gpt-9"}, system)[0] == 400


def test_an_unreachable_api_is_a_502_not_a_crash():
    from indicrag.index.lexical import LexicalIndex
    from indicrag.pipeline import Retrievers
    from indicrag.serve import System, handle_ask

    down = _provider(FakeOpener(*[urllib.error.URLError("no route")] * 4))
    system = System(passages=PASSAGES, retrievers=Retrievers(lexical=LexicalIndex.build(PASSAGES)),
                    provider=down, providers={"extractive": None, down.name: down})
    status, body = handle_ask({"query": "How much do students receive?", "method": "bm25",
                               "bm25_floor": 0}, system)
    assert status == 502 and "could not reach" in body["error"]


def test_a_refusal_written_as_the_answer_is_a_refusal_whatever_the_flag_says():
    """Seen live from a Groq-hosted model: answerable true, answer = the refusal."""
    reply = json.dumps({"answerable": True, "answer": REFUSAL, "citation": "b#p0"})
    out = _provider(FakeOpener(_reply(reply))).answer(
        "What is the interest rate in Nagaland?", PASSAGES, lang=classify("What is the interest rate in Nagaland?"))
    assert out.answerable is False and out.text == REFUSAL


def test_every_request_carries_a_user_agent():
    """Groq's edge returns a bare 403 to urllib's default User-Agent."""
    opener = FakeOpener(_reply('{"answerable": true, "answer": "x", "citation": "b#p0"}'))
    _provider(opener).complete("prompt")
    assert opener.requests[0].get_header("User-agent") == "indicrag/0.1"
