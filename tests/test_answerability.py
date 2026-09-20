"""Answerability signals and metrics.

The per-class breakdown is the point of this module, so it is what gets pinned:
an aggregate F1 can look respectable while the system fails completely on the
near-miss and false-premise classes, which is exactly what the first measured run
showed.
"""

from __future__ import annotations

from indicrag.answerability.signals import (
    Features,
    ThresholdSignal,
    extract_features,
    unanswerable_f1,
)
from indicrag.evaluation.answerability import (
    AnswerabilityOutcome,
    AnswerabilityReport,
)
from indicrag.models import Retrieved


def _hits(scores, schemes=None):
    return [
        Retrieved(passage_id=f"p{i}", score=s, rank=i + 1, method="test")
        for i, s in enumerate(scores)
    ]


# --- features -------------------------------------------------------------------


def test_margin_separates_a_clear_winner_from_a_flat_field():
    """A flat field is the signature of a near-miss: the question matches the
    scheme's subject without matching any particular statement."""
    clear = extract_features(_hits([0.9, 0.3, 0.2, 0.1, 0.05]))
    flat = extract_features(_hits([0.9, 0.88, 0.87, 0.86, 0.85]))
    assert clear.max_score == flat.max_score
    assert clear.margin > flat.margin
    assert clear.score_spread > flat.score_spread


def test_scheme_agreement_counts_the_dominant_scheme_in_top_k():
    hits = _hits([0.9, 0.8, 0.7, 0.6])
    scheme_of = {"p0": "a", "p1": "a", "p2": "a", "p3": "b"}
    assert extract_features(hits, scheme_of=scheme_of, k=4).scheme_agreement == 0.75


def test_features_on_an_empty_result_do_not_raise():
    f = extract_features([])
    assert f.max_score == 0.0 and f.n_candidates == 0


def test_feature_vector_length_is_stable():
    assert len(Features().as_vector()) == 6


# --- threshold ------------------------------------------------------------------


def test_threshold_fit_maximises_unanswerable_f1_not_accuracy():
    """Classes are deliberately imbalanced (PRD §6.1). Accuracy is maximised by
    never abstaining, which is the failure being guarded against."""
    feats = [Features(max_score=s) for s in [0.9] * 18 + [0.1] * 2]
    labels = [True] * 18 + [False] * 2
    signal, f1 = ThresholdSignal.fit(feats, labels)
    preds = [signal.predict_answerable(f) for f in feats]
    assert f1 > 0.0
    assert not preds[-1], "the low-scoring unanswerable item must be caught"


def test_threshold_scale_normalises_unbounded_scores():
    """BM25 is unbounded and its scale varies with query length; a raw tau tuned
    on short queries over-abstains on long ones."""
    feats = [Features(max_score=s) for s in [40.0, 30.0, 2.0, 1.0]]
    signal, _ = ThresholdSignal.fit(feats, [True, True, False, False])
    assert signal.scale == 40.0
    assert 0.0 <= signal.score(feats[0]) <= 1.0


def test_unanswerable_f1_treats_unanswerable_as_positive():
    # pred, gold (True == answerable)
    assert unanswerable_f1([False, True], [False, True]) == 1.0
    assert unanswerable_f1([True, True], [False, True]) == 0.0


def test_a_system_that_never_abstains_scores_zero_f1():
    assert unanswerable_f1([True] * 5, [True, True, False, False, False]) == 0.0


# --- report ---------------------------------------------------------------------


def _o(gold, pred, *, qt="English", klass=None, i=0):
    return AnswerabilityOutcome(
        item_id=f"q{i}",
        query_type=qt,
        gold_answerable=gold,
        pred_answerable=pred,
        unanswerable_class=klass,
    )


def test_confusion_matrix_counts_each_quadrant():
    r = AnswerabilityReport(
        "s",
        [
            _o(False, False, i=0),  # tp
            _o(True, False, i=1),  # fp -- over-abstention
            _o(False, True, i=2),  # fn -- hallucination risk
            _o(True, True, i=3),  # tn
        ],
    )
    cm = r.confusion()
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (1, 1, 1, 1)
    assert cm.accuracy == 0.5
    assert cm.precision == 0.5 and cm.recall == 0.5


def test_per_class_recall_exposes_a_failure_the_aggregate_hides():
    """The measured run: out-of-scope caught, false-premise missed entirely."""
    outcomes = [_o(False, False, klass="out-of-scope", i=i) for i in range(8)]
    outcomes += [_o(False, True, klass="false-premise", i=100 + i) for i in range(8)]
    r = AnswerabilityReport("s", outcomes)
    by_class = r.recall_by_class()
    assert by_class["out-of-scope"][0] == 1.0
    assert by_class["false-premise"][0] == 0.0
    # The aggregate looks like a partial success while one class is total failure.
    assert r.confusion().recall == 0.5


def test_abstention_by_query_type_separates_over_abstention():
    r = AnswerabilityReport(
        "s",
        [
            _o(True, False, qt="Code-Mixed", i=0),  # answerable, refused
            _o(False, False, qt="Code-Mixed", i=1),  # correctly refused
            _o(True, True, qt="English", i=2),
        ],
    )
    by_qt = r.abstention_by_query_type()
    assert by_qt["Code-Mixed"] == (1.0, 1.0, 2)
    assert by_qt["English"] == (0.0, 0.0, 1)


def test_empty_report_does_not_divide_by_zero():
    r = AnswerabilityReport("s", [])
    assert r.confusion().accuracy == 0.0
    assert r.recall_by_class() == {}
