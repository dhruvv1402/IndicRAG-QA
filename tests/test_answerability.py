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


# --- generator self-report (ARCHITECTURE §12 signal 2) ---------------------------


def _reports():
    from indicrag.answerability.signals import SelfReport

    return [
        SelfReport(answerable=True, confidence=0.9),   # answerable, confident
        SelfReport(answerable=True, confidence=0.2),   # answerable, hedged
        SelfReport(answerable=False, confidence=0.0),  # abstained
        SelfReport(answerable=False, confidence=0.9),  # confidently abstained
    ]


def test_an_abstention_scores_zero_however_confident_it_was():
    """A model abstaining with confidence 0.9 is confidently saying it cannot
    answer -- that is not a confident answer."""
    from indicrag.answerability.signals import SelfReport, SelfReportSignal

    signal = SelfReportSignal()
    assert signal.score(SelfReport(answerable=False, confidence=0.9)) == 0.0
    assert signal.predict_answerable(SelfReport(answerable=False, confidence=0.9)) is False


def test_confidence_gates_an_answer_the_model_did_give():
    from indicrag.answerability.signals import SelfReport, SelfReportSignal

    signal = SelfReportSignal(min_confidence=0.5)
    assert signal.predict_answerable(SelfReport(True, 0.9)) is True
    assert signal.predict_answerable(SelfReport(True, 0.2)) is False


def test_fitting_maximises_f1_on_the_unanswerable_class():
    from indicrag.answerability.signals import SelfReportSignal

    reports = _reports()
    signal, f1 = SelfReportSignal.fit(reports, [True, False, False, False])
    assert f1 == 1.0
    assert [signal.predict_answerable(r) for r in reports] == [True, False, False, False]


def test_a_zero_threshold_is_a_real_outcome_not_a_failed_fit():
    """If the flag carries the decision and confidence adds nothing, 0.0 is the
    right answer and means something."""
    from indicrag.answerability.signals import SelfReport, SelfReportSignal

    reports = [SelfReport(True, 0.9), SelfReport(False, 0.9)]
    signal, f1 = SelfReportSignal.fit(reports, [True, False])
    assert signal.min_confidence == 0.0
    assert f1 == 1.0


def test_fitting_on_nothing_does_not_raise():
    from indicrag.answerability.signals import SelfReportSignal

    signal, f1 = SelfReportSignal.fit([], [])
    assert f1 == 0.0 and signal.min_confidence == 0.0


def test_it_reads_a_generation_without_importing_one():
    from indicrag.rag.arms import Generation
    from indicrag.answerability.signals import SelfReport

    gen = Generation(
        item_id="i", arm="B", question="q", answerable=True,
        answer="Rs. 12,000 per annum", citation="p1", confidence=0.8,
    )
    report = SelfReport.from_generation(gen)
    assert report.answerable and report.confidence == 0.8
    assert report.answer_length == 4


# --- the dev/test split for fitting a signal -------------------------------------


def _mixed_set():
    from indicrag.models import QAItem

    items = [
        QAItem(id=f"qa-en-en-{i:03d}", question=f"q{i}", query_lang="en",
               passage_lang="en", answerable=True, gold_passage_ids=["p1"])
        for i in range(320)
    ]
    for klass, n in [
        ("out-of-scope", 25), ("near-miss", 30),
        ("false-premise", 15), ("under-specified", 10),
    ]:
        items += [
            QAItem(id=f"un-{klass}-{i:03d}", question=f"u{i}", query_lang="en",
                   passage_lang="", answerable=False, unanswerable_class=klass)
            for i in range(n)
        ]
    return items


def test_both_halves_of_the_split_contain_unanswerable_items():
    """Two ways this broke. Sorting by id put every `un-*` item in test, so the
    fit saw no positive examples and returned a threshold that never abstains.
    Round-robin then put every one of them in dev. Either way one half is
    single-class and the numbers look like a finding."""
    from indicrag.evaluation.answerability import stratified_by_class as _stratified_by_class

    items = _mixed_set()
    dev = _stratified_by_class(items, 120)
    dev_ids = {i.id for i in dev}
    test = [i for i in items if i.id not in dev_ids]

    assert any(not i.answerable for i in dev)
    assert any(not i.answerable for i in test)
    assert any(i.answerable for i in dev)
    assert any(i.answerable for i in test)


def test_every_unanswerable_class_survives_into_both_halves():
    from indicrag.evaluation.answerability import stratified_by_class as _stratified_by_class

    items = _mixed_set()
    dev = _stratified_by_class(items, 120)
    dev_ids = {i.id for i in dev}
    test = [i for i in items if i.id not in dev_ids]

    classes = {"out-of-scope", "near-miss", "false-premise", "under-specified"}
    assert {i.unanswerable_class for i in dev if not i.answerable} == classes
    assert {i.unanswerable_class for i in test if not i.answerable} == classes


def test_the_draw_is_proportional_rather_than_even():
    """Even cells would drain the small unanswerable classes into dev."""
    from indicrag.evaluation.answerability import stratified_by_class as _stratified_by_class

    dev = _stratified_by_class(_mixed_set(), 120)
    answerable = sum(1 for i in dev if i.answerable)
    assert answerable > 80  # 320/400 of the draw, not 1/5 of it


def test_the_draw_is_deterministic():
    from indicrag.evaluation.answerability import stratified_by_class as _stratified_by_class

    items = _mixed_set()
    assert [i.id for i in _stratified_by_class(items, 120)] == [
        i.id for i in _stratified_by_class(items, 120)
    ]


# --- score separation reporting --------------------------------------------------
#
# These exist because the threshold result was twice written up wrongly: once as
# 0.000 recall on false-premise, once as 0.93, from the same flat curve. The
# separation table and the sweep are what make that visible, so their semantics
# are pinned rather than left to the formatter.


def test_separation_is_empty_when_one_class_is_missing():
    """A table comparing two classes when only one is present says nothing."""
    from indicrag.evaluation.answerability import format_separation

    assert format_separation([0.1, 0.2], [True, True]) == []
    assert format_separation([0.1, 0.2], [False, False]) == []


def test_separation_warns_when_unanswerable_score_at_least_as_high():
    """The direction a threshold assumes is answerable-scores-higher. When it
    does not hold, the table has to say so -- that is the whole finding."""
    from indicrag.evaluation.answerability import format_separation

    text = "\n".join(format_separation([0.1, 0.3], [True, False]))
    assert "opposite of the direction" in text
    assert "not weak here, it is absent" in text


def test_separation_stays_quiet_when_the_signal_points_the_right_way():
    from indicrag.evaluation.answerability import format_separation

    text = "\n".join(format_separation([0.9, 0.1], [True, False]))
    assert "opposite of the direction" not in text


def test_separation_reports_the_base_rate():
    """Precision at the base rate is how you recognise a useless threshold."""
    from indicrag.evaluation.answerability import format_separation

    text = "\n".join(format_separation([0.1] * 8 + [0.2] * 2, [True] * 8 + [False] * 2))
    assert "0.200" in text


# --- the threshold sweep ---------------------------------------------------------


def test_the_sweep_covers_the_whole_range():
    from indicrag.answerability.signals import Features
    from indicrag.evaluation.answerability import tau_sweep

    feats = [Features(max_score=s) for s in (0.1, 0.5, 0.9)]
    rows = tau_sweep(feats, [True, False, True], grid=10)
    assert len(rows) == 11
    assert rows[0][0] == 0.0 and rows[-1][0] == 1.0


def test_a_perfectly_separable_signal_reaches_f1_one():
    """Guards the sweep arithmetic itself: if it cannot find a clean split when
    one exists, a flat result proves nothing."""
    from indicrag.answerability.signals import Features
    from indicrag.evaluation.answerability import tau_sweep

    feats = [Features(max_score=s) for s in (1.0, 0.9, 0.1, 0.05)]
    rows = tau_sweep(feats, [True, True, False, False], grid=20)
    assert max(r[3] for r in rows) == 1.0


def test_an_uninformative_signal_never_beats_the_base_rate_on_precision():
    """Identical scores in both classes: no threshold can do better than
    abstaining at random, so precision tops out at the prevalence."""
    from indicrag.answerability.signals import Features
    from indicrag.evaluation.answerability import tau_sweep

    labels = [True] * 8 + [False] * 2
    feats = [Features(max_score=0.5) for _ in labels]
    rows = tau_sweep(feats, labels, grid=20)
    assert max(r[1] for r in rows) <= 0.2 + 1e-9


def test_abstention_rises_monotonically_with_tau():
    from indicrag.answerability.signals import Features
    from indicrag.evaluation.answerability import tau_sweep

    feats = [Features(max_score=s / 10) for s in range(1, 11)]
    rates = [r[4] for r in tau_sweep(feats, [True] * 5 + [False] * 5, grid=20)]
    assert rates == sorted(rates)


def test_the_sweep_report_marks_the_best_row_and_its_cost():
    from indicrag.answerability.signals import Features
    from indicrag.evaluation.answerability import format_tau_sweep, tau_sweep

    feats = [Features(max_score=s) for s in (1.0, 0.9, 0.1, 0.05)]
    text = "\n".join(format_tau_sweep(tau_sweep(feats, [True, True, False, False], grid=10)))
    assert "best F1" in text
    assert "abstention" in text
    assert "it is silence" in text
