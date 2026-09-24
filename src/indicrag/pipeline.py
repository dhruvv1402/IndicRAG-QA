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

from .index.hybrid import rrf_fusion, script_aware_rrf, weighted_fusion
from .index.lexical import LexicalIndex
from .models import Answer, Passage, Retrieved
from .query.langid import classify


@dataclass
class Retrievers:
    """Whatever indices are available. Any may be None; `retrieve` adapts."""

    lexical: LexicalIndex | None = None
    dense: object | None = None  # DenseIndex, kept loose to avoid importing numpy here
    encoder: object | None = None  # Encoder
    #: passage_id -> "deva" | "latin". Required by script-aware fusion, which
    #: needs to know which passages the lexical retriever could even reach.
    script_of: dict[str, str] | None = None


def retrieve(
    query: str,
    r: Retrievers,
    *,
    method: str = "hybrid",
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

    if method in {"hybrid", "hybrid-script", "hybrid-rrf", "hybrid-weighted"}:
        if r.lexical is None:
            raise RuntimeError("hybrid requires a lexical index")
        # Degrade to lexical rather than raise when no dense index is loaded.
        # `hybrid` is the default, and a default that fails on the simple path --
        # no embeddings built yet, a unit test, a fresh checkout -- is a default
        # that makes the library hard to pick up. An explicitly requested
        # `dense` still raises, because there is nothing sensible to substitute.
        if r.dense is None or r.encoder is None:
            return r.lexical.search_bm25(query, k)
        lex = r.lexical.search_bm25(query, candidates)
        vec = r.encoder.encode_query(query)
        den = r.dense.search_vector(vec, candidates)

        if method == "hybrid-weighted":
            return weighted_fusion(lex, den, alpha=alpha, k=k)
        if method == "hybrid-rrf":
            return rrf_fusion(lex, den, k=k)

        # "hybrid" means script-aware, because plain RRF measurably harms the
        # cross-lingual and code-mixed cases this system exists to serve:
        # 0.022 against 0.153 and 0.028 against 0.173 on Recall@5. Shipping the
        # worse fusion as the default while the finding sits in the paper would
        # be indefensible. Plain RRF stays reachable as "hybrid-rrf" so the
        # comparison remains runnable.
        if r.script_of is None:
            return rrf_fusion(lex, den, k=k)
        return script_aware_rrf(
            lex, den, script_of=r.script_of, query_script=classify(query).script, k=k
        )

    raise ValueError(f"unknown retrieval method {method!r}")


def answer_query(
    query: str,
    passages: Sequence[Passage],
    *,
    retrievers: Retrievers | None = None,
    method: str = "hybrid",
    k: int = 5,
    tau: float = 0.0,
    bm25_floor: float = 0.0,
    provider=None,
) -> Answer:
    """Produce the full response object described in docs/PRD.md §9."""
    start = time.time()
    by_id = {p.passage_id: p for p in passages}

    lang = classify(query)

    if retrievers is None:
        retrievers = Retrievers(lexical=LexicalIndex.build(passages))
    if retrievers.script_of is None:
        retrievers.script_of = {
            p.passage_id: ("deva" if p.lang == "hi" else "latin") for p in passages
        }

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

    # The abstention gate the paper recommends (§VI-H): the BM25 top score,
    # read whatever retriever supplies the evidence. Rank fusion discards score
    # magnitude, so the fused score cannot be thresholded usefully; the lexical
    # score is computed anyway and does separate the classes on verified
    # questions. It catches questions about topics the corpus lacks, and misses
    # about half of near-misses, which only a reader of the passage can see.
    below_floor = False
    if bm25_floor > 0 and retrievers.lexical is not None:
        lexical = retrievers.lexical.search_bm25(query, 1)
        bm25_top = lexical[0].score if lexical else 0.0
        retrieval_meta["bm25_top"] = round(bm25_top, 4)
        retrieval_meta["bm25_floor"] = bm25_floor
        below_floor = bm25_top < bm25_floor

    # Answerability. With tau=0 this abstains only when retrieval returned
    # nothing at all, which is the right default before the threshold has been
    # calibrated on the dev split -- an uncalibrated threshold would silently
    # suppress answers and make the retrieval numbers look worse than they are.
    #
    # NOTE: tau is compared against the retriever's RAW score, and those scales
    # differ by orders of magnitude -- BM25 is unbounded and runs around 16 here,
    # RRF sums reciprocal ranks and runs around 0.03. A tau tuned for one method
    # is meaningless for another. `answerability.ThresholdSignal` normalises by
    # an observed scale for exactly this reason and is what the evaluation uses;
    # this field is the simple interactive knob, and its default of 0 avoids the
    # trap entirely.
    if not hits or top_score < tau or below_floor:
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

    # Lead with the passage the answer actually came from, not with whatever
    # retrieval ranked first. Those differ whenever the best supporting sentence
    # sits in a lower-ranked passage, and the result is an answer displayed
    # beside evidence that does not contain it -- observed in the demo, where an
    # English answer was shown citing a Hindi passage. For a system whose whole
    # claim is evidence-grounding, a citation that does not support its answer is
    # the worst possible defect: it looks exactly like a correct one.
    cited_ids = [pid for pid in (generated.citations or []) if pid in by_id]
    if cited_ids:
        primary = cited_ids[0]
        citations.sort(key=lambda c: c["passage_id"] != primary)

    # The generator's own verdict decides the label. This used to be hard-coded
    # to ANSWERABLE, so a generator that declined -- the answerability signal
    # that works best on this corpus -- printed the refusal sentence under an
    # ANSWERABLE label. Module 4 reads the verdict directly and was unaffected.
    ans = Answer(
        query=query,
        query_type=lang.query_type,
        query_lang=lang.lang,
        answerability="ANSWERABLE" if generated.answerable else "UNANSWERABLE",
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
