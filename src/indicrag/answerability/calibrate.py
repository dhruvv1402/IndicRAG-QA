"""Combine the answerability signals by fitting a linear model on dev.

`docs/ARCHITECTURE.md` §12 specifies four signals: retrieval threshold, generator
self-report, NLI entailment, and a calibrated combination of all of them. This is
the combination.

**Logistic regression, not something larger.** There are 120 dev items. Anything
with more capacity memorises them, and the coefficients of a linear model are
directly reportable -- *which signal carries the decision* is itself a finding,
and the measured Module 5 result makes it an interesting one: if the entailment
coefficient dominates, that is the quantitative form of "a retrieval score cannot
see whether the asserted fact exists".

Features are deliberately few and each has a reason to be there:

- `max_score`, `mean_top_k` -- how good the evidence looks.
- `margin` (top1 − top2) -- whether anything *else* looked equally good. A
  near-miss unanswerable question retrieves several passages at similar scores,
  because it matches a scheme's subject without matching any statement in it; an
  answerable question usually has one clear winner. High max with low margin is
  the signature of the hardest unanswerable class, and `max_score` alone is blind
  to it.
- `scheme_agreement` -- whether the top-k agree on which scheme they are about.
- `generator_answerable`, `generator_confidence` -- the model's self-report.
- `entailment` -- whether the evidence entails what was answered.
- `answer_length` -- fabricated answers in this corpus run long and hedged.

Fitting maximises F1 on the UNANSWERABLE class rather than accuracy, because the
classes are deliberately imbalanced (80 of 400) and accuracy is maximised by a
system that never abstains -- the exact failure being guarded against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .signals import Features, unanswerable_f1

FEATURE_NAMES = (
    "max_score",
    "margin",
    "mean_top_k",
    "score_spread",
    "scheme_agreement",
    "generator_answerable",
    "generator_confidence",
    "entailment",
    "answer_length",
)


@dataclass
class CombinedFeatures:
    """Everything the calibrated signal sees for one query."""

    retrieval: Features = field(default_factory=Features)
    generator_answerable: float = 1.0
    generator_confidence: float = 0.0
    entailment: float = 0.0
    answer_length: int = 0

    def as_vector(self) -> list[float]:
        r = self.retrieval
        return [
            r.max_score,
            r.margin,
            r.mean_top_k,
            r.score_spread,
            r.scheme_agreement,
            self.generator_answerable,
            self.generator_confidence,
            self.entailment,
            # Bounded so one long answer cannot dominate the fit.
            min(self.answer_length, 60) / 60.0,
        ]


@dataclass
class CalibratedSignal:
    """A fitted logistic model predicting ANSWERABLE."""

    coefficients: list[float]
    intercept: float
    threshold: float = 0.5
    means: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)

    def _standardise(self, vec: Sequence[float]) -> list[float]:
        if not self.means:
            return list(vec)
        return [
            (v - m) / (s if s > 1e-9 else 1.0)
            for v, m, s in zip(vec, self.means, self.scales, strict=True)
        ]

    def probability(self, features: CombinedFeatures) -> float:
        import math

        vec = self._standardise(features.as_vector())
        z = self.intercept + sum(c * v for c, v in zip(self.coefficients, vec, strict=True))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_answerable(self, features: CombinedFeatures) -> bool:
        return self.probability(features) >= self.threshold

    def weights(self) -> list[tuple[str, float]]:
        """Feature name to coefficient, largest absolute first.

        Reported in the paper: which signal carries the decision is a finding.
        """
        pairs = list(zip(FEATURE_NAMES, self.coefficients, strict=True))
        return sorted(pairs, key=lambda kv: -abs(kv[1]))


def fit(
    features: Sequence[CombinedFeatures],
    labels: Sequence[bool],
    *,
    grid: int = 40,
    seed: int = 20260922,
) -> tuple[CalibratedSignal, float]:
    """Fit on dev and pick the decision threshold by unanswerable-class F1.

    Returns (signal, dev F1). Falls back to a degenerate always-answerable signal
    when the labels are single-class, which happens on tiny slices and should not
    raise.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.array([f.as_vector() for f in features], dtype=float)
    y = np.array([1 if lab else 0 for lab in labels], dtype=int)

    if len(set(y.tolist())) < 2 or len(y) < 4:
        return CalibratedSignal([0.0] * len(FEATURE_NAMES), 1.0, 0.5), 0.0

    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales[scales < 1e-9] = 1.0
    Xs = (X - means) / scales

    model = LogisticRegression(
        max_iter=2000,
        # Strong regularisation: nine features on ~120 items will otherwise fit
        # noise, and the coefficients are meant to be interpretable.
        C=0.5,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(Xs, y)

    signal = CalibratedSignal(
        coefficients=[float(c) for c in model.coef_[0]],
        intercept=float(model.intercept_[0]),
        threshold=0.5,
        means=[float(m) for m in means],
        scales=[float(s) for s in scales],
    )

    best = (0.5, -1.0)
    for i in range(1, grid):
        tau = i / grid
        signal.threshold = tau
        preds = [signal.predict_answerable(f) for f in features]
        f1 = unanswerable_f1(preds, labels)
        if f1 > best[1]:
            best = (tau, f1)
    signal.threshold = best[0]
    return signal, best[1]


def format_weights(signal: CalibratedSignal) -> list[str]:
    out = [
        "CALIBRATED SIGNAL -- fitted coefficients",
        "-" * 78,
        "  Positive weight pushes towards ANSWERABLE. Features are standardised,",
        "  so magnitudes are comparable across rows.",
        "",
    ]
    for name, coef in signal.weights():
        bar = "#" * min(40, int(abs(coef) * 20))
        out.append(f"  {name:<24}{coef:>8.3f}  {bar}")
    out += ["", f"  decision threshold: {signal.threshold:.2f}"]
    return out
