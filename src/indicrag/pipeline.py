"""End-to-end query pipeline: classify, retrieve, decide answerability, answer.

This is the seam the CLI and the evaluation both sit on. It deliberately does
*not* branch retrieval on the detected query type -- one pipeline serves English,
Hindi and code-mixed queries alike. That is what makes the per-language
comparison in docs/PLAN.md §3.2 meaningful: if the pipeline varied by language,
the table would be measuring the routing logic rather than the retriever.

The generator is pluggable and defaults to the extractive provider, so the whole
system is usable -- and testable, and evaluable on retrieval and answerability --
without a 2 GB model download.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from .index.hybrid import rrf_fusion, weighted_fusion
from .index.lexical import LexicalIndex
from .models import Answer, Passage, Retrieved
from .query.langid import classify


@dataclass
class Retrievers:
    """Whatever indices are available. Any may be None; `retrieve` adapts."""

    lexical: LexicalIndex | None = None
    dense: object | None = None  # DenseIndex, kept loose to avoid importing numpy here
    encoder: object | None = None  # Encoder


def retrieve(
    query: str,
    r: Retrievers,
    *,
    method: str = "bm25",
    k: int = 5,
    candidates: int = 50,
    alpha: float = 0.4,
) -> list[Retrieved]:
    """Run one retrieval method and return the top-k.

    `candidates` is intentionally larger than `k`: fusion needs a deeper pool from
    each component than it finally returns, otherwise a passage ranked 8th by one
    component and 1st by the other never meets its counterpart.
    """
    method = method.lower()

    if method == "bm25":
        if r.lexical is None:
            raise RuntimeError("bm25 requested but no lexical index is loaded")
        return r.lexical.search_bm25(query, k)

    if method == "tfidf":
        if r.lexical is None:
            raise RuntimeError("tfidf requested but no lexical index is loaded")
        return r.lexical.search_tfidf(query, k)

    if method == "dense":
        if r.dense is None or r.encoder is None:
            raise RuntimeError("dense requested but no dense index/encoder is loaded")
        vec = r.encoder.encode_query(query)
        return r.dense.search_vector(vec, k)

    if method in {"hybrid", "hybrid-rrf", "hybrid-weighted"}:
        if r.lexical is None or r.dense is None or r.encoder is None:
            raise RuntimeError("hybrid requires both a lexical and a dense index")
        lex = r.lexical.search_bm25(query, candidates)
        vec = r.encoder.encode_query(query)
        den = r.dense.search_vector(vec, candidates)
        if method == "hybrid-weighted":
            return weighted_fusion(lex, den, alpha=alpha, k=k)
        return rrf_fusion(lex, den, k=k)

    raise ValueError(f"unknown retrieval method {method!r}")


def answer_query(
    query: str,
    passages: Sequence[Passage],
    *,
    retrievers: Retrievers | None = None,
    method: str = "bm25",
    k: int = 5,
    tau: float = 0.0,
    provider=None,
) -> Answer:
    """Produce the full response object described in docs/PRD.md §9."""
    start = time.time()
    by_id = {p.passage_id: p for p in passages}

    lang = classify(query)

    if retrievers is None:
        retrievers = Retrievers(lexical=LexicalIndex.build(passages))

    hits = retrieve(query, retrievers, method=method, k=k)

    top_score = hits[0].score if hits else 0.0
    margin = (hits[0].score - hits[1].score) if len(hits) > 1 else top_score

    citations = [
        {
            "passage_id": h.passage_id,
            "doc_id": by_id[h.passage_id].doc_id,
            "scheme": by_id[h.passage_id].scheme,
            "lang": by_id[h.passage_id].lang,
            "section_path": by_id[h.passage_id].section_path,
            "score": round(h.score, 4),
            "snippet": by_id[h.passage_id].text[:300],
        }
        for h in hits
        if h.passage_id in by_id
    ]

    retrieval_meta = {
        "method": method,
        "k": k,
        "top_score": round(top_score, 4),
        "margin": round(margin, 4),
        "n_candidates": len(hits),
    }

    # Answerability. With tau=0 this abstains only when retrieval returned
    # nothing at all, which is the right default before the threshold has been
    # calibrated on the dev split -- an uncalibrated threshold would silently
    # suppress answers and make the retrieval numbers look worse than they are.
    if not hits or top_score < tau:
        ans = Answer.refusal(
            query=query,
            query_type=lang.query_type,
            query_lang=lang.lang,
            confidence=round(1.0 - min(top_score, 1.0), 4),
            citations=citations,
            retrieval=retrieval_meta,
        )
        ans.latency_ms = int((time.time() - start) * 1000)
        return ans

    if provider is None:
        from .rag.providers import ExtractiveProvider

        provider = ExtractiveProvider()

    generated = provider.answer(query, [by_id[h.passage_id] for h in hits], lang=lang)

    ans = Answer(
        query=query,
        query_type=lang.query_type,
        query_lang=lang.lang,
        answerability="ANSWERABLE",
        answer=generated.text,
        answer_lang=generated.lang,
        confidence=round(float(generated.confidence), 4),
        citations=citations,
        explanation=generated.explanation,
        retrieval=retrieval_meta,
        model=provider.name,
    )
    ans.latency_ms = int((time.time() - start) * 1000)
    return ans
