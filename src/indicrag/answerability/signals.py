"""Answerability signals and their calibration.

Four signals, per `docs/ARCHITECTURE.md` §12. Two of them need no generator and
are therefore usable now: the retrieval-score threshold and the score-margin
feature. The generator self-report and the NLI entailment check arrive with P4.

**The margin feature is the interesting one.** `max_sim` alone says how good the
best passage looks; it does not say whether anything *else* looked equally good.
A near-miss unanswerable question -- the topic and scheme are in the corpus, the
specific fact is not -- characteristically retrieves several passages at similar
scores, because the question matches the scheme's general subject matter without
matching any particular statement. A genuinely answerable question usually has
one clear winner. So a high `max_sim` with a *small* `top1 - top2` margin is the
signature of the hardest unanswerable class, and a threshold on `max_sim` alone
cannot see it.

Scores are min-max normalised per query before thresholding, because BM25 scores
are unbounded and their scale varies with query length: a raw threshold tuned on
short queries silently abstains on long ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from ..models import Retrieved


@dataclass
class Features:
    """The feature vector the calibrated combination is fit over."""

    max_score: float = 0.0
    margin: float = 0.0  # top1 - top2, on the raw scale
    mean_top_k: float = 0.0
    n_candidates: int = 0
    score_spread: float = 0.0  # top1 - topK
    scheme_agreement: float = 0.0  # share of top-k from the winning scheme

    def as_dict(self) -> dict:
        return asdict(self)

    def as_vector(self) -> list[float]:
        return [
            self.max_score,
            self.margin,
            self.mean_top_k,
            float(self.n_candidates),
            self.score_spread,
            self.scheme_agreement,
        ]


def extract_features(
    hits: Sequence[Retrieved], *, scheme_of: dict[str, str] | None = None, k: int = 5
) -> Features:
    """Derive answerability features from one query's retrieval result."""
    top = list(hits[:k])
    if not top:
        return Features()

    scores = [h.score for h in top]
    max_score = scores[0]
    margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    spread = scores[0] - scores[-1] if len(scores) > 1 else 0.0

    agreement = 0.0
    if scheme_of:
        schemes = [scheme_of.get(h.passage_id, "") for h in top]
        if schemes:
            winner = max(set(schemes), key=schemes.count)
            agreement = schemes.count(winner) / len(schemes)

    return Features(
        max_score=float(max_score),
        margin=float(margin),
        mean_top_k=float(sum(scores) / len(scores)),
        n_candidates=len(hits),
        score_spread=float(spread),
        scheme_agreement=float(agreement),
    )


# --- threshold signal ------------------------------------------------------------


@dataclass
class ThresholdSignal:
    """Abstain when the best retrieved passage scores below tau.

    `normalise` divides by the corpus-wide score scale observed at fit time.
    Without it, tau is tied to BM25's unbounded scale, which varies with query
    length -- a value tuned on short queries abstains far too often on long ones,
    and that error falls hardest on the Hindi and code-mixed slices where queries
    run longer.
    """

    tau: float = 0.0
    scale: float = 1.0

    def score(self, features: Features) -> float:
        """Confidence that the question is ANSWERABLE, in [0, 1]."""
        if self.scale <= 0:
            return 0.0
        return max(0.0, min(1.0, features.max_score / self.scale))

    def predict_answerable(self, features: Features) -> bool:
        return self.score(features) >= self.tau

    @classmethod
    def fit(
        cls,
        features: Sequence[Features],
        labels: Sequence[bool],
        *,
        grid: int = 40,
    ) -> tuple[ThresholdSignal, float]:
        """Choose tau by maximising F1 on the UNANSWERABLE class.

        F1 on the *unanswerable* class, not on accuracy: the classes are
        deliberately imbalanced (80 of 400, PRD §6.1), and accuracy is maximised
        by a system that never abstains at all, which is precisely the failure
        being guarded against.
        """
        if not features:
            return cls(), 0.0
        scale = max((f.max_score for f in features), default=1.0) or 1.0
        best = (cls(tau=0.0, scale=scale), -1.0)
        for i in range(grid + 1):
            tau = i / grid
            signal = cls(tau=tau, scale=scale)
            preds = [signal.predict_answerable(f) for f in features]
            f1 = unanswerable_f1(preds, labels)
            if f1 > best[1]:
                best = (signal, f1)
        return best


def unanswerable_f1(pred_answerable: Sequence[bool], gold_answerable: Sequence[bool]) -> float:
    """F1 treating UNANSWERABLE as the positive class."""
    tp = sum(1 for p, g in zip(pred_answerable, gold_answerable, strict=True) if not p and not g)
    fp = sum(1 for p, g in zip(pred_answerable, gold_answerable, strict=True) if not p and g)
    fn = sum(1 for p, g in zip(pred_answerable, gold_answerable, strict=True) if p and not g)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


@dataclass
class SelfReport:
    """What the generator claimed about one question.

    Two separate claims, deliberately kept apart. `answerable` is the model's
    verdict; `confidence` is how sure it says it is. A model that abstains with
    confidence 0.9 is making a confident claim that it cannot answer, which is
    not the same as answering with low confidence, and collapsing them loses the
    distinction that makes this signal worth having.
    """

    answerable: bool = True
    confidence: float = 0.0
    answer_length: int = 0

    @classmethod
    def from_generation(cls, gen) -> SelfReport:
        """Build from a `rag.arms.Generation` without importing it."""
        return cls(
            answerable=bool(gen.answerable),
            confidence=float(gen.confidence or 0.0),
            answer_length=len((gen.answer or "").split()),
        )


@dataclass
class SelfReportSignal:
    """Abstain when the generator says it cannot answer, or says so weakly.

    ARCHITECTURE §12 signal 2, and the only one of the four that sees the
    passage *text* rather than a retrieval score. That is the whole reason it is
    evaluated separately rather than folded into the calibrated combination: the
    threshold signal scores 0.000 on false-premise questions because such a
    question retrieves confidently -- the scheme it names is real -- and no
    function of retrieval scores can separate it from an answerable one. A
    signal that reads the passage can, in principle, notice that the asserted
    fact is absent.

    Whether it does is an empirical question, and it is the one this signal
    exists to answer. It is free: the fields are already generated.
    """

    #: Minimum self-reported confidence to accept an answer the model did give.
    min_confidence: float = 0.0

    def score(self, report: SelfReport) -> float:
        """Confidence that the question is ANSWERABLE, in [0, 1]."""
        if not report.answerable:
            return 0.0
        return max(0.0, min(1.0, report.confidence))

    def predict_answerable(self, report: SelfReport) -> bool:
        return report.answerable and report.confidence >= self.min_confidence

    @classmethod
    def fit(
        cls,
        reports: Sequence[SelfReport],
        labels: Sequence[bool],
        *,
        grid: int = 20,
    ) -> tuple[SelfReportSignal, float]:
        """Choose `min_confidence` by maximising F1 on the UNANSWERABLE class.

        Same objective as the threshold signal, for the same reason: accuracy is
        maximised by never abstaining, which is the failure being guarded
        against.

        Note that a fitted threshold of 0.0 is a real and informative outcome,
        not a failure to fit. It means the model's own `answerable` flag carries
        the decision and its confidence adds nothing -- which is what one would
        expect from a small model that reports 0.9 for almost everything it
        answers.
        """
        if not reports:
            return cls(), 0.0
        best = (cls(), -1.0)
        for i in range(grid + 1):
            signal = cls(min_confidence=i / grid)
            preds = [signal.predict_answerable(r) for r in reports]
            f1 = unanswerable_f1(preds, labels)
            if f1 > best[1]:
                best = (signal, f1)
        return best
