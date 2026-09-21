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


# --- the default method must actually be the default method ----------------------


class _StubDense:
    """A dense index that ranks passages in a fixed order."""

    def __init__(self, order):
        self.order = order

    def search_vector(self, _vec, k):
        from indicrag.models import Retrieved

        return [
            Retrieved(passage_id=pid, score=1.0 - 0.01 * i, rank=i + 1)
            for i, pid in enumerate(self.order[:k])
        ]


class _StubEncoder:
    def encode_query(self, _q):
        return [0.0]


def _retrievers(with_dense: bool):
    from indicrag.index.lexical import LexicalIndex

    r = Retrievers(lexical=LexicalIndex.build(CORPUS))
    if with_dense:
        r.dense = _StubDense(["c#p0", "b#p0", "a#p0"])
        r.encoder = _StubEncoder()
    return r


def test_hybrid_without_a_dense_index_degrades_to_lexical_scores():
    """Documented behaviour: a fresh checkout with no embeddings still answers."""
    result = answer_query("scholarship", CORPUS, retrievers=_retrievers(False), method="hybrid")
    assert result.retrieval["top_score"] > 1.0  # BM25 is unbounded


def test_hybrid_with_a_dense_index_actually_fuses():
    """The regression this guards: `ask` built only a lexical index, so the
    default method silently degraded to BM25 -- the one configuration the
    cross-lingual result shows collapsing -- and nothing in the output said so.
    RRF sums reciprocal ranks, so a fused score is well under 1."""
    result = answer_query("scholarship", CORPUS, retrievers=_retrievers(True), method="hybrid")
    assert 0.0 < result.retrieval["top_score"] < 1.0


def test_fusion_can_return_a_passage_the_lexical_retriever_cannot_see():
    """A Devanagari passage for a Latin query is unreachable by BM25."""
    result = answer_query("scholarship", CORPUS, retrievers=_retrievers(True), method="hybrid")
    assert "c#p0" in [c["passage_id"] for c in result.citations]


# --- generated answering on the interactive path ---------------------------------


def _llama_stub(reply: str):
    """A LlamaCppProvider whose `complete` returns `reply`, with no model load."""
    from indicrag.rag.providers import LlamaCppProvider

    p = LlamaCppProvider.__new__(LlamaCppProvider)
    p.calls = 0
    p.parse_failures = 0
    p.retries = 0
    p.complete = lambda prompt, *a, **kw: reply  # type: ignore[method-assign]
    return p


def _lang():
    from indicrag.query.langid import classify

    return classify("what is the scholarship amount")


def test_a_generated_answer_carries_the_model_citation():
    provider = _llama_stub(
        '{"answerable": true, "answer": "Rs. 12,000 per annum", '
        '"citation": "b#p0", "confidence": 0.9}'
    )
    got = provider.answer("how much", CORPUS, lang=_lang())
    assert got.text == "Rs. 12,000 per annum"
    assert got.citations == ["b#p0"]
    assert got.answerable and got.confidence == 0.9


def test_an_invented_citation_does_not_become_the_displayed_evidence():
    """The model naming an id that is not in the context must not put that id
    in front of the user as the source."""
    provider = _llama_stub(
        '{"answerable": true, "answer": "Rs. 12,000", '
        '"citation": "does-not-exist#p9", "confidence": 0.8}'
    )
    got = provider.answer("how much", CORPUS, lang=_lang())
    assert got.citations == [CORPUS[0].passage_id]
    assert "did not itself cite" in got.explanation


def test_a_declined_answer_becomes_the_required_refusal_string():
    from indicrag.rag.prompts import REFUSAL

    provider = _llama_stub('{"answerable": false, "answer": "", "citation": "", "confidence": 0.0}')
    got = provider.answer("how much", CORPUS, lang=_lang())
    assert got.text == REFUSAL
    assert got.answerable is False


def test_unparseable_output_refuses_rather_than_answering_emptily():
    from indicrag.rag.prompts import REFUSAL

    got = _llama_stub("the model rambled").answer("how much", CORPUS, lang=_lang())
    assert got.text == REFUSAL
    assert got.answerable is False


def test_the_demo_and_the_evaluation_share_one_prompt():
    """If these diverged, the demonstrated system would not be the measured
    one."""
    seen = {}
    provider = _llama_stub('{"answerable": true, "answer": "x", "citation": "a#p0", "confidence": 0.5}')
    original = provider.complete
    provider.complete = lambda prompt, *a, **kw: (seen.setdefault("p", prompt), original(prompt))[1]

    provider.answer("how much", CORPUS, lang=_lang())

    from indicrag.rag.prompts import build_answer_prompt

    assert seen["p"] == build_answer_prompt("how much", CORPUS)


# --- the seam between language ID and fusion -------------------------------------


def test_romanized_hinglish_reaches_fusion_as_latin_script():
    """The correction depends on this and nothing pinned it.

    `script_aware_rrf` treats the lexical retriever as eligible for everything
    when the query script is "mixed". Romanized Hinglish -- "Scholarship ke liye
    kya chahiye?" -- is Code-Mixed by *language* but written wholly in Latin, so
    BM25 genuinely cannot reach a Devanagari passage and the correction must
    apply. If classify() reported "mixed" here instead of "latin", the
    correction would quietly switch off for the entire code-mixed slice, which
    is one of the two headline results.
    """
    from indicrag.query.langid import classify

    romanized = classify("Scholarship ke liye minimum eligibility kya hai?")
    assert romanized.query_type == "Code-Mixed"
    assert romanized.script == "latin"

    # A query that really does carry both scripts is the other case: BM25 has
    # tokens in each, so both retrievers are eligible and no correction is due.
    both = classify("मनरेगा ke liye eligibility kya hai?")
    assert both.query_type == "Code-Mixed"
    assert both.script == "mixed"


def test_a_hinglish_query_can_retrieve_a_devanagari_passage():
    """End to end over the seam: Code-Mixed in, cross-script passage out."""
    result = answer_query(
        "Scholarship ke liye kya chahiye?",
        CORPUS,
        retrievers=_retrievers(True),
        method="hybrid",
    )
    assert result.query_type == "Code-Mixed"
    assert "c#p0" in [c["passage_id"] for c in result.citations]
