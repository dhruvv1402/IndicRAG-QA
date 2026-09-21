"""Metrics, bootstrap intervals and the paired test.

A quiet bug in this module would shift every number in the paper without
anything looking wrong, so the metrics are pinned against hand-computed values
rather than against whatever the code currently returns.
"""

from __future__ import annotations

from indicrag.evaluation.retrieval import Outcome, RetrievalReport
from indicrag.evaluation.run import alpha_verdict
from indicrag.evaluation.stats import bootstrap_ci, paired_randomization_test


def _o(item_id, gold, retrieved, slice_key="EN->EN", group="monolingual"):
    return Outcome(
        item_id=item_id,
        slice_key=slice_key,
        language_group=group,
        gold=gold,
        retrieved=retrieved,
    )


def test_hit_recall_and_precision_match_hand_computed_values():
    o = _o("q1", gold=["a", "b"], retrieved=["x", "a", "y", "b", "z"])
    assert o.hit_at(1) is False
    assert o.hit_at(2) is True
    assert o.recall_at(2) == 0.5  # one of two gold passages
    assert o.recall_at(4) == 1.0
    assert o.precision_at(4) == 0.5  # two hits in four slots
    assert o.reciprocal_rank() == 0.5  # first gold at rank 2


def test_a_query_with_no_hit_scores_zero_reciprocal_rank():
    assert _o("q", gold=["a"], retrieved=["x", "y"]).reciprocal_rank() == 0.0


def test_ndcg_rewards_a_higher_rank():
    high = _o("q", gold=["a"], retrieved=["a", "x", "y"]).ndcg_at(3)
    low = _o("q", gold=["a"], retrieved=["x", "y", "a"]).ndcg_at(3)
    assert high == 1.0
    assert 0 < low < high


def test_recall_counts_any_gold_passage_not_a_specific_one():
    """Both language versions of a fact are correct evidence (PRD §6.4).

    Scoring against one designated passage would penalise a cross-lingual
    retriever for the exact behaviour being measured.
    """
    o = _o("q", gold=["en#p1", "hi#p1"], retrieved=["hi#p1"])
    assert o.hit_at(1) is True


def test_slices_partition_the_report():
    report = RetrievalReport(
        system="s",
        outcomes=[
            _o("a", ["g"], ["g"], "EN->EN", "monolingual"),
            _o("b", ["g"], ["x"], "HI->EN", "cross-lingual"),
        ],
    )
    assert report.n() == 2
    assert report.n("monolingual") == 1
    assert report.recall_at(1, "monolingual") == 1.0
    assert report.recall_at(1, "cross-lingual") == 0.0
    assert report.recall_at(1) == 0.5


def test_bootstrap_interval_brackets_the_point_estimate():
    outcomes = [_o(f"q{i}", ["g"], ["g"] if i % 2 == 0 else ["x"]) for i in range(40)]
    ci = bootstrap_ci(outcomes, lambda o: o.recall_at(1), resamples=400)
    assert ci.low <= ci.point <= ci.high
    assert abs(ci.point - 0.5) < 1e-9
    assert ci.width() > 0


def test_bootstrap_on_a_constant_metric_has_zero_width():
    outcomes = [_o(f"q{i}", ["g"], ["g"]) for i in range(20)]
    ci = bootstrap_ci(outcomes, lambda o: o.recall_at(1), resamples=200)
    assert ci.point == 1.0 and ci.width() == 0.0


def test_identical_systems_are_not_significantly_different():
    a = [_o(f"q{i}", ["g"], ["g"] if i % 3 else ["x"]) for i in range(30)]
    b = [_o(f"q{i}", ["g"], ["g"] if i % 3 else ["x"]) for i in range(30)]
    res = paired_randomization_test(a, b, lambda o: o.recall_at(1))
    assert res.delta == 0.0
    assert not res.significant()


def test_a_uniformly_better_system_is_significant():
    a = [_o(f"q{i}", ["g"], ["g"]) for i in range(40)]
    b = [_o(f"q{i}", ["g"], ["x"]) for i in range(40)]
    res = paired_randomization_test(a, b, lambda o: o.recall_at(1), trials=2000)
    assert res.delta == 1.0
    assert res.significant()


def test_the_paired_test_matches_on_item_id_not_position():
    """Two systems may emit outcomes in different orders."""
    a = [_o("q1", ["g"], ["g"]), _o("q2", ["g"], ["x"])]
    b = [_o("q2", ["g"], ["x"]), _o("q1", ["g"], ["g"])]
    res = paired_randomization_test(a, b, lambda o: o.recall_at(1))
    assert res.n == 2
    assert res.delta == 0.0


def test_alpha_verdict_refuses_to_call_a_win_inside_the_noise():
    """The naive test called H2 supported on a 0.001 difference."""
    sweep = [(0.0, 0.303), (0.5, 0.487), (0.9, 0.500), (1.0, 0.499)]
    assert "NOT SUPPORTED" in alpha_verdict(sweep)


def test_alpha_verdict_accepts_a_clear_win():
    sweep = [(0.0, 0.30), (0.5, 0.62), (1.0, 0.50)]
    assert "SUPPORTED" in alpha_verdict(sweep)
    assert "NOT SUPPORTED" not in alpha_verdict(sweep)


# --- calibration of the statistical machinery ------------------------------------
#
# The tests above check that the functions behave sensibly on hand-built cases.
# These check the property that actually makes their output trustworthy: that a
# 95% interval covers the truth about 95% of the time, and that a test at the
# 0.05 level rejects a true null about 5% of the time. Every interval and every
# p-value in the paper rests on these holding, and a wrong percentile index or a
# mis-built null would leave every hand-built case passing.


def _bernoulli_outcomes(rng, n, p):
    """n queries, each a hit with probability p -- the shape of recall@1."""
    return [
        _o(f"q{i}", ["g"], ["g"] if rng.random() < p else ["x"]) for i in range(n)
    ]


def test_the_bootstrap_interval_covers_the_truth_about_95_percent_of_the_time():
    import random

    true_p, n, sims = 0.6, 60, 200
    covered = 0
    for s in range(sims):
        rng = random.Random(1000 + s)
        outcomes = _bernoulli_outcomes(rng, n, true_p)
        ci = bootstrap_ci(
            outcomes, lambda o: o.recall_at(1), resamples=200, seed=2000 + s
        )
        if ci.low <= true_p <= ci.high:
            covered += 1

    rate = covered / sims
    # Binomial noise on 200 draws at 0.95 is about +-0.03; the band is wide
    # enough not to flake and narrow enough that a wrong percentile index --
    # which would land near 0.50 or 1.00 -- fails it.
    assert 0.88 <= rate <= 0.99, f"coverage {rate:.3f} over {sims} simulations"


def test_the_paired_test_rejects_a_true_null_about_five_percent_of_the_time():
    """Two systems drawn from the same distribution differ only by noise. A test
    that reports significance far more often than 5% would make the paper's
    p-values meaningless, and every hand-built case above would still pass."""
    import random

    n, sims = 40, 120
    rejected = 0
    for s in range(sims):
        rng = random.Random(3000 + s)
        a = _bernoulli_outcomes(rng, n, 0.5)
        b = _bernoulli_outcomes(rng, n, 0.5)
        result = paired_randomization_test(
            a, b, lambda o: o.recall_at(1), trials=300, seed=4000 + s
        )
        if result.significant():
            rejected += 1

    rate = rejected / sims
    assert rate <= 0.15, f"false positive rate {rate:.3f} over {sims} simulations"


def test_the_paired_test_finds_a_real_but_modest_difference():
    """The other side of the same coin: a test tuned to never reject would pass
    the check above and be useless."""
    import random

    rng = random.Random(7)
    a = _bernoulli_outcomes(rng, 200, 0.75)
    b = _bernoulli_outcomes(rng, 200, 0.45)
    result = paired_randomization_test(a, b, lambda o: o.recall_at(1), trials=2000)
    assert result.significant()
    assert result.delta > 0.15


def test_a_p_value_is_never_reported_as_exactly_zero():
    """With `trials` samples the evidence supports 'below 1/trials', not zero,
    and a paper quoting p=0.0000 would be claiming more than it measured."""
    import random

    rng = random.Random(11)
    a = _bernoulli_outcomes(rng, 80, 1.0)
    b = _bernoulli_outcomes(rng, 80, 0.0)
    result = paired_randomization_test(a, b, lambda o: o.recall_at(1), trials=500)
    assert result.p_value > 0.0
    assert result.p_value <= 1.0 / 500 + 1e-9
