"""End-to-end query pipeline and the output contract.

The citation test guards the defect that matters most in a system whose whole
claim is evidence-grounding: an answer displayed beside evidence that does not
contain it. That failure looks exactly like a correct result.
"""

from __future__ import annotations

from indicrag.models import Answer, Passage
from indicrag.pipeline import Retrievers, answer_query


def _p(pid, text, lang="en", scheme="s1"):
    return Passage(
        passage_id=pid, doc_id=f"{scheme}-{lang}", scheme=scheme, lang=lang,
        text=text, section_path="Eligibility", token_count=len(text.split()),
    )


CORPUS = [
    _p("a#p0", "The Jan Dhan scheme opened 417 million accounts by January 2021. " * 4),
    _p("b#p0", "Students receive Rs. 12,000 per annum under the merit scholarship. " * 4),
    _p("c#p0", "मध्याह्न भोजन योजना 1995 में शुरू हुई थी और प्राथमिक कक्षाओं के बच्चों को भोजन देती है। " * 4, lang="hi", scheme="c"),
]


def test_the_leading_citation_is_the_passage_the_answer_came_from():
    """Retrieval order and answer provenance differ whenever the best supporting
    sentence sits in a lower-ranked passage."""
    result = answer_query("How much do students receive per annum?", CORPUS, k=3)
    assert result.answerability == "ANSWERABLE"
    assert result.citations, "an answerable response must carry a citation"
    lead = result.citations[0]
    assert lead["passage_id"] in result.answer or result.answer in _text_of(lead["passage_id"])


def _text_of(pid):
    return next(p.text for p in CORPUS if p.passage_id == pid)


def test_an_answerable_response_always_carries_a_citation():
    """PRD §9.1. An answer with no citation is a bug, not a degraded result."""
    result = answer_query("How much do students receive?", CORPUS, k=3)
    assert result.answerability == "ANSWERABLE"
    assert len(result.citations) >= 1
    assert result.citations[0]["passage_id"]


def test_query_type_is_one_of_the_three_the_brief_names():
    for q in ["How much is the amount?", "राशि कितनी है?", "Amount kitna hai?"]:
        assert answer_query(q, CORPUS, k=2).query_type in {"English", "Indic", "Code-Mixed"}


def test_refusal_uses_the_exact_required_string():
    """PRD FR-11. The string is fixed; a paraphrase is a contract violation."""
    result = answer_query("zzzz qqqq wwww", CORPUS, k=3, tau=1.1)
    assert result.answerability == "UNANSWERABLE"
    assert result.answer == Answer.REFUSAL


def test_a_refusal_still_reports_what_was_retrieved_and_rejected():
    """Useful to an operator auditing why the system declined.

    tau is on the retriever's own scale, which is unbounded for BM25 (~16 here)
    and tiny for RRF (~0.03), so the value is deliberately far above any of them.
    """
    result = answer_query("How much do students receive?", CORPUS, k=3, tau=1e9)
    assert result.answerability == "UNANSWERABLE"
    assert result.retrieval["method"]


def test_the_response_carries_retrieval_metadata():
    meta = answer_query("How much do students receive?", CORPUS, k=3).retrieval
    assert meta["k"] == 3
    assert "top_score" in meta and "margin" in meta


def test_script_map_is_built_when_the_caller_does_not_supply_one():
    from indicrag.index.lexical import LexicalIndex

    r = Retrievers(lexical=LexicalIndex.build(CORPUS))
    answer_query("How much do students receive?", CORPUS, retrievers=r, method="bm25", k=2)
    assert r.script_of and r.script_of["c#p0"] == "deva"


def test_citations_resolve_to_real_passages():
    ids = {p.passage_id for p in CORPUS}
    result = answer_query("How much do students receive?", CORPUS, k=3)
    assert all(c["passage_id"] in ids for c in result.citations)
