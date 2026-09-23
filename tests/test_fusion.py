"""Weighted fusion, and what each normaliser does to a missing component.

§IV-E claims the normalisation choice is ablatable. It is only ablatable while
`normalisation="zscore"` stays reachable and keeps its own semantics, and no
caller in the package passes it -- the ablation runs from
`scripts/check-fusion-normalisation.py`. These tests are what stops the
unused branch rotting into a copy of the used one.

The measured difference between the two is not the skew argument the section
used to give. It is imputation: a passage only one component returned scores 0
from the other, and 0 is the floor of a min-max range but the mean of a z-scored
one. On the gold set that puts an absent dense component above 63% of the
candidates it is ranked against.
"""

from __future__ import annotations

from indicrag.index.hybrid import _minmax, _zscore, weighted_fusion
from indicrag.models import Retrieved


def _run(method: str, scores: list[tuple[str, float]]) -> list[Retrieved]:
    return [
        Retrieved(passage_id=pid, score=s, rank=i, method=method)
        for i, (pid, s) in enumerate(scores, start=1)
    ]


def test_an_absent_component_is_the_floor_under_minmax_and_the_mean_under_zscore():
    """The asymmetry that actually separates the two normalisers."""
    scores = {"a": 10.0, "b": 2.0, "c": 1.0}
    assert min(_minmax(scores).values()) == 0.0  # 0 is attainable: it is the floor
    assert min(_zscore(scores).values()) < 0.0  # 0 is interior: it is the mean

    # So a passage missing from this component, contributing 0, outranks nothing
    # under min-max and outranks the below-average candidates under z-score.
    assert sum(1 for v in _minmax(scores).values() if v < 0.0) == 0
    assert sum(1 for v in _zscore(scores).values() if v < 0.0) == 2


def test_zscore_can_reorder_a_ranking_that_minmax_leaves_alone():
    """The branch is live: it is not an alias for the default."""
    lexical = _run("bm25", [("p1", 30.0), ("p2", 4.0), ("p3", 3.0)])
    dense = _run("e5", [("p2", 0.90), ("p3", 0.88), ("p4", 0.86)])

    top = {
        norm: weighted_fusion(lexical, dense, alpha=0.5, k=4, normalisation=norm)[0].passage_id
        for norm in ("minmax", "zscore")
    }
    assert top["minmax"] != top["zscore"]


def test_a_lexical_outlier_carries_further_under_zscore():
    """The direction §IV-E predicted, at the scale it turned out to have.

    One runaway BM25 score compresses every other lexical candidate towards the
    floor under min-max. Under z-score the outlier inflates the standard
    deviation instead, which pulls it back towards the pack -- so the outlier
    itself is *less* dominant, while the mid-pack lexical candidates it used to
    flatten keep more of their score. Either way the fused top-1 draws a larger
    share from the lexical side, which is what the gold-set composition shows:
    29.0% against 34.4%.
    """
    lexical = _run("bm25", [("p1", 40.0), ("p2", 3.0), ("p3", 2.5)])
    dense = _run("e5", [("p1", 0.70), ("p2", 0.69), ("p3", 0.68)])

    shares = {}
    for norm in ("minmax", "zscore"):
        fused = weighted_fusion(lexical, dense, alpha=0.5, k=3, normalisation=norm)
        by_id = {r.passage_id: r for r in fused}
        lex = abs(by_id["p2"].component_scores["lexical"])
        den = abs(by_id["p2"].component_scores["dense"])
        shares[norm] = lex / (lex + den) if lex + den else 0.0

    assert shares["zscore"] > shares["minmax"]


def test_a_degenerate_component_is_neutral_rather_than_unanimous():
    """Every candidate scoring alike is no information, under either normaliser.

    Min-max says 0.5 and z-score says 0.0, and both are the neutral point of
    their own scale -- a component that cannot discriminate must not outvote one
    that can.
    """
    flat = {"a": 7.0, "b": 7.0, "c": 7.0}
    assert set(_minmax(flat).values()) == {0.5}
    assert set(_zscore(flat).values()) == {0.0}


def test_fusing_with_no_candidates_on_one_side_keeps_the_other_side_ordered():
    dense = _run("e5", [("p1", 0.9), ("p2", 0.5)])
    fused = weighted_fusion([], dense, alpha=0.4, k=2)
    assert [r.passage_id for r in fused] == ["p1", "p2"]


def test_a_cast_lexical_vote_is_not_doubled_when_observed_votes_count():
    """A Latin query can reach a Hindi passage through a Latin word it quotes.
    Doubling that passage's score for a 'missing' vote that was in fact cast
    lifted an irrelevant passage to rank 1 in the test-split error analysis."""
    from indicrag.index.hybrid import script_aware_rrf
    from indicrag.models import Retrieved

    def run(ids):
        return [Retrieved(passage_id=p, score=1.0, rank=r, method="x") for r, p in enumerate(ids, 1)]

    script_of = {"en-good": "latin", "hi-quotes-card": "deva"}
    lexical = run(["en-good", "hi-quotes-card"])
    dense = run(["en-good", "hi-quotes-card"])

    old = script_aware_rrf(lexical, dense, script_of=script_of, query_script="latin", k=2)
    new = script_aware_rrf(
        lexical, dense, script_of=script_of, query_script="latin", k=2, count_observed_votes=True
    )
    assert old[0].passage_id == "hi-quotes-card"  # the defect, pinned
    assert new[0].passage_id == "en-good"
