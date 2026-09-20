"""NLI entailment scoring and the calibrated combination.

The model itself is not loaded here -- these run against a stub, so the caching,
label handling and fitting logic are tested without a 1.1 GB download or CPU
inference. `NLIScorer.available()` is what the harness uses to decide whether the
real model can run.

The label-order test is the one that would otherwise bite silently: a transposed
entailment/contradiction map inverts every verdict while still producing numbers
that look entirely reasonable.
"""

from __future__ import annotations

from indicrag.answerability.calibrate import (
    FEATURE_NAMES,
    CalibratedSignal,
    CombinedFeatures,
    fit,
    format_weights,
)
from indicrag.answerability.nli import NLIScorer, NLIVerdict, _key
from indicrag.answerability.signals import Features

# --- verdict ---------------------------------------------------------------------


def test_verdict_label_picks_the_largest_probability():
    assert NLIVerdict(0.8, 0.1, 0.1).label == "entailment"
    assert NLIVerdict(0.1, 0.8, 0.1).label == "neutral"
    assert NLIVerdict(0.1, 0.1, 0.8).label == "contradiction"


def test_verdict_round_trips_through_its_dict_form():
    v = NLIVerdict(0.75, 0.2, 0.05)
    d = v.as_dict()
    assert set(d) == {"entailment", "neutral", "contradiction"}
    assert abs(d["entailment"] - 0.75) < 1e-6


# --- cache -----------------------------------------------------------------------


def test_cache_key_depends_on_both_premise_and_hypothesis():
    assert _key("a", "b") != _key("b", "a")
    assert _key("a", "b") == _key("a", "b")


def test_empty_input_is_scored_neutral_without_loading_the_model(tmp_path):
    """A missing citation must not trigger a 1.1 GB model load."""
    scorer = NLIScorer(cache_path=tmp_path / "nli.jsonl")
    out = scorer.score_pairs([("", "something"), ("passage", "  ")])
    assert [v.label for v in out] == ["neutral", "neutral"]
    assert scorer._model is None


def test_cached_pairs_are_served_without_loading_the_model(tmp_path):
    path = tmp_path / "nli.jsonl"
    warm = NLIScorer(cache_path=path)
    warm._remember("the scheme gives Rs 12000", "it gives Rs 12000", NLIVerdict(0.9, 0.07, 0.03))

    cold = NLIScorer(cache_path=path)
    out = cold.score_pairs([("the scheme gives Rs 12000", "it gives Rs 12000")])
    assert out[0].entailment == 0.9
    assert cold._model is None, "a cache hit must not load the model"


def test_availability_check_does_not_raise_when_weights_are_absent():
    assert NLIScorer.available("definitely/not-a-real-model-id") is False


# --- calibration -----------------------------------------------------------------


def _cf(max_score=0.5, margin=0.2, entail=0.5, gen_ans=1.0, conf=0.5, length=10):
    return CombinedFeatures(
        retrieval=Features(
            max_score=max_score, margin=margin, mean_top_k=max_score * 0.8,
            score_spread=margin, scheme_agreement=0.8, n_candidates=5,
        ),
        generator_answerable=gen_ans,
        generator_confidence=conf,
        entailment=entail,
        answer_length=length,
    )


def test_feature_vector_matches_the_declared_names():
    assert len(_cf().as_vector()) == len(FEATURE_NAMES)


def test_answer_length_is_bounded_so_one_long_answer_cannot_dominate():
    short = _cf(length=10).as_vector()[-1]
    long = _cf(length=500).as_vector()[-1]
    assert long == 1.0 and short < 1.0


def test_fit_separates_a_clean_signal():
    """Entailment alone is made decisive; the fit should find it."""
    feats, labels = [], []
    for i in range(40):
        answerable = i % 2 == 0
        feats.append(_cf(entail=0.9 if answerable else 0.05))
        labels.append(answerable)
    signal, f1 = fit(feats, labels)
    assert f1 > 0.8
    preds = [signal.predict_answerable(f) for f in feats]
    assert preds == labels


def test_the_dominant_feature_is_reported_first():
    feats, labels = [], []
    for i in range(40):
        answerable = i % 2 == 0
        feats.append(_cf(entail=0.95 if answerable else 0.02, max_score=0.5, margin=0.2))
        labels.append(answerable)
    signal, _ = fit(feats, labels)
    assert signal.weights()[0][0] == "entailment"


def test_fit_on_single_class_labels_does_not_raise():
    feats = [_cf() for _ in range(10)]
    signal, f1 = fit(feats, [True] * 10)
    assert f1 == 0.0
    assert signal.predict_answerable(feats[0]) in (True, False)


def test_probability_is_bounded():
    signal = CalibratedSignal([50.0] * len(FEATURE_NAMES), 100.0)
    p = signal.probability(_cf())
    assert 0.0 <= p <= 1.0


def test_weights_format_lists_every_feature():
    signal = CalibratedSignal([0.1 * i for i in range(len(FEATURE_NAMES))], 0.0)
    text = "\n".join(format_weights(signal))
    for name in FEATURE_NAMES:
        assert name in text
