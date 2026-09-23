"""Hybrid fusion of lexical and dense retrieval.

Two methods, compared rather than blended into a single "hybrid" number, because
they fail differently and reporting which one wins is itself a result.

**Weighted score fusion** is the brief's formula, `alpha*lex + (1-alpha)*dense`,
applied after normalising each component over the candidate pool. Normalisation
is min-max rather than z-score. Both are implemented and the ablation has been
run (`scripts/check-fusion-normalisation.py`): they are indistinguishable,
-0.003 Recall@5 for z-score at p=1.00, and no point of the alpha sweep separates
them by more than 0.013.

The skew argument for min-max -- BM25 pools are right-skewed on 97.7% of
queries, so z-score should let lexical outliers dominate -- is true in its
premise and too small in its effect to reach the metric. What actually differs
is the imputation. A passage only one component returned contributes 0 from the
other, and 0 is the floor of a min-max range but the *mean* of a z-scored one:
an absent dense component ranks above 63% of the candidates it is compared
against under z-score and above none of them under min-max. Min-max is kept
because a retriever's silence should not be scored as an average opinion, which
is an argument about what the fusion means, not about the number it produces.

**Reciprocal Rank Fusion** uses ranks only, so it needs no normalisation at all
and is immune to that skew. `k=60` is the standard value and is deliberately left
untuned, to keep it an honest baseline against a tuned alpha.

The alpha sweep doubles as the evidence for hypothesis H2: alpha=1 and alpha=0
recover pure lexical and pure dense retrieval, so if no interior alpha beats both
endpoints on the dev split, H2 is false and the paper must say so.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from ..models import Retrieved

RRF_K = 60
Normalisation = Literal["minmax", "zscore"]


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        # Every candidate scored alike: carrying that through as 1.0 would let a
        # degenerate component outvote a discriminating one. 0.5 is neutral.
        return dict.fromkeys(scores, 0.5)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _zscore(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    sd = var**0.5
    if sd < 1e-12:
        return dict.fromkeys(scores, 0.0)
    return {k: (v - mean) / sd for k, v in scores.items()}


def weighted_fusion(
    lexical: Sequence[Retrieved],
    dense: Sequence[Retrieved],
    *,
    alpha: float = 0.4,
    k: int = 10,
    normalisation: Normalisation = "minmax",
) -> list[Retrieved]:
    """Score(p) = alpha * norm(lexical) + (1 - alpha) * norm(dense).

    A passage found by only one component keeps that component's normalised score
    weighted by its coefficient and contributes 0 from the other. That is the
    intended behaviour: a passage no dense retriever surfaced should not be
    rewarded for its absence.
    """
    norm = _minmax if normalisation == "minmax" else _zscore
    lex_scores = norm({r.passage_id: r.score for r in lexical})
    den_scores = norm({r.passage_id: r.score for r in dense})

    fused: dict[str, float] = defaultdict(float)
    for pid, s in lex_scores.items():
        fused[pid] += alpha * s
    for pid, s in den_scores.items():
        fused[pid] += (1.0 - alpha) * s

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        Retrieved(
            passage_id=pid,
            score=float(score),
            rank=rank,
            method=f"hybrid-weighted(a={alpha:g},{normalisation})",
            component_scores={
                "lexical": float(lex_scores.get(pid, 0.0)),
                "dense": float(den_scores.get(pid, 0.0)),
            },
        )
        for rank, (pid, score) in enumerate(ranked, start=1)
    ]


def script_aware_rrf(
    lexical: Sequence[Retrieved],
    dense: Sequence[Retrieved],
    *,
    script_of: dict[str, str],
    query_script: str,
    k: int = 10,
    rrf_k: int = RRF_K,
    count_observed_votes: bool = False,
) -> list[Retrieved]:
    """RRF that does not penalise a passage the lexical retriever could never see.

    **This fixes a structural bug in plain RRF on a bilingual corpus**, found by
    reading the Module 6 cases rather than by looking at a metric. Of 20 hybrid
    failures, e5-base alone had retrieved the gold passage in 8 -- and plain RRF
    lost all 8.

    The mechanism: BM25 can only return passages sharing a script with the query,
    because a Romanized or Devanagari query and a passage in the other script
    share essentially no tokens. In plain RRF a passage found by both systems
    collects two contributions and a passage found only by the dense retriever
    collects one, so cross-script passages are penalised roughly 2:1 for the
    retriever's blindness rather than for their own irrelevance. The lexical
    retriever's silence about a Devanagari passage, given a Latin query, is not
    evidence against that passage. It is no evidence at all.

    So each passage is scored over the systems that were *eligible* to retrieve
    it. A cross-script candidate is judged on the dense contribution alone rather
    than being docked for a vote that was never possible.

    `count_observed_votes` corrects a defect found in the test-split error
    analysis (paper §VII). Eligibility is decided by script, but BM25 does
    sometimes return a cross-script passage -- a Hindi passage that quotes
    "Soil Health Card" in Latin letters, or shares a digit string -- and the
    doubling meant for a missing vote then doubles a vote that was cast, which
    can lift an irrelevant passage to rank 1. With the flag, a passage the
    lexical retriever actually returned counts as eligible. It stays off: on the
    dev split it is worse, not better (-0.026 Recall@5 overall, -0.093
    cross-lingual, neither significant; evals/report-observed-votes.txt).
    Over dev and test it changes 6 queries' hit@5: 5 lost, 1 gained. In 4 of
    the 5 losses BM25 reached the cross-script gold through a year or number in
    the question, in the fifth through a Latin acronym (ICDS) -- so a cast
    cross-script vote is usually real evidence, and doubling it usually helps.
    """
    eligible_both = query_script in ("mixed", "unknown")

    contributions: dict[str, dict[str, float]] = defaultdict(dict)
    for label, run in (("lexical", lexical), ("dense", dense)):
        for r in run:
            contributions[r.passage_id][label] = 1.0 / (rrf_k + r.rank)

    fused: dict[str, float] = {}
    for pid, parts in contributions.items():
        passage_script = script_of.get(pid, "unknown")
        lexical_could_see = eligible_both or passage_script == query_script
        if count_observed_votes and "lexical" in parts:
            lexical_could_see = True
        n_eligible = 2 if lexical_could_see else 1
        fused[pid] = sum(parts.values()) * (2.0 / n_eligible)

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        Retrieved(
            passage_id=pid,
            score=float(score),
            rank=rank,
            method=f"hybrid-rrf-script(k={rrf_k})",
            component_scores=contributions[pid],
        )
        for rank, (pid, score) in enumerate(ranked, start=1)
    ]


def rrf_fusion(
    *runs: Sequence[Retrieved],
    k: int = 10,
    rrf_k: int = RRF_K,
) -> list[Retrieved]:
    """RRF(p) = sum over runs of 1 / (rrf_k + rank).

    Accepts any number of runs, so adding a third retriever later needs no change
    here.
    """
    fused: dict[str, float] = defaultdict(float)
    parts: dict[str, dict[str, float]] = defaultdict(dict)

    for i, run in enumerate(runs):
        label = run[0].method if run else f"run{i}"
        for r in run:
            contribution = 1.0 / (rrf_k + r.rank)
            fused[r.passage_id] += contribution
            parts[r.passage_id][label] = contribution

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        Retrieved(
            passage_id=pid,
            score=float(score),
            rank=rank,
            method=f"hybrid-rrf(k={rrf_k})",
            component_scores=parts[pid],
        )
        for rank, (pid, score) in enumerate(ranked, start=1)
    ]
