"""Answerability metrics: confusion matrix, per-class recall, abstention by language.

Three breakdowns, and the aggregate F1 is the least informative of them.

**Per unanswerable class** is where the real result lives. A single F1 can look
respectable while the system fails completely on near-miss questions, because
out-of-scope questions are trivially rejected and there are 25 of them. Reporting
only the aggregate would hide exactly the failure Module 5 exists to expose.

**Abstention rate per query type** guards a specific and likely failure named in
`docs/ARCHITECTURE.md` §12.3: if code-mixed queries retrieve at systematically
lower scores, a single global threshold abstains on them disproportionately. The
system would then refuse to answer Hinglish users not because evidence is missing
but because their scores run lower -- a fairness failure that an aggregate F1
cannot see. If it appears, a per-language-type threshold is the obvious remedy,
and the fact that one is needed is itself worth reporting.

**Over-abstention** is the cost side. A system that abstains on everything scores
perfect recall on the unanswerable class, so recall alone must never be quoted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnswerabilityOutcome:
    item_id: str
    query_type: str  # English | Indic | Code-Mixed
    gold_answerable: bool
    pred_answerable: bool
    confidence: float = 0.0
    unanswerable_class: str | None = None


@dataclass
class ConfusionMatrix:
    """Rows are gold, columns predicted. UNANSWERABLE is the positive class."""

    tp: int = 0  # gold unanswerable, predicted unanswerable
    fp: int = 0  # gold answerable, predicted unanswerable  (over-abstention)
    fn: int = 0  # gold unanswerable, predicted answerable  (hallucination risk)
    tn: int = 0  # gold answerable, predicted answerable

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def rows(self) -> list[str]:
        return [
            "                        predicted",
            "                  UNANSWERABLE   ANSWERABLE",
            f"  gold UNANSWERABLE {self.tp:>10}   {self.fn:>10}",
            f"  gold ANSWERABLE   {self.fp:>10}   {self.tn:>10}",
        ]


@dataclass
class AnswerabilityReport:
    system: str
    outcomes: list[AnswerabilityOutcome] = field(default_factory=list)

    def _sel(self, query_type: str | None) -> list[AnswerabilityOutcome]:
        if query_type in (None, "all"):
            return self.outcomes
        return [o for o in self.outcomes if o.query_type == query_type]

    def confusion(self, query_type: str | None = None) -> ConfusionMatrix:
        cm = ConfusionMatrix()
        for o in self._sel(query_type):
            if not o.gold_answerable and not o.pred_answerable:
                cm.tp += 1
            elif o.gold_answerable and not o.pred_answerable:
                cm.fp += 1
            elif not o.gold_answerable and o.pred_answerable:
                cm.fn += 1
            else:
                cm.tn += 1
        return cm

    def recall_by_class(self) -> dict[str, tuple[float, int]]:
        """Recall on each unanswerable class. The table that matters most."""
        out: dict[str, tuple[float, int]] = {}
        classes = {o.unanswerable_class for o in self.outcomes if o.unanswerable_class}
        for klass in sorted(classes):
            sel = [o for o in self.outcomes if o.unanswerable_class == klass]
            caught = sum(1 for o in sel if not o.pred_answerable)
            out[klass] = (caught / len(sel) if sel else 0.0, len(sel))
        return out

    def abstention_by_query_type(self) -> dict[str, tuple[float, float, int]]:
        """query_type -> (abstention rate, over-abstention rate, n)."""
        out: dict[str, tuple[float, float, int]] = {}
        for qt in sorted({o.query_type for o in self.outcomes}):
            sel = self._sel(qt)
            abstained = sum(1 for o in sel if not o.pred_answerable)
            answerable = [o for o in sel if o.gold_answerable]
            over = sum(1 for o in answerable if not o.pred_answerable)
            out[qt] = (
                abstained / len(sel) if sel else 0.0,
                over / len(answerable) if answerable else 0.0,
                len(sel),
            )
        return out


def format_answerability(report: AnswerabilityReport) -> list[str]:
    rule = "-" * 78
    cm = report.confusion()
    out = [
        f"ANSWERABILITY -- {report.system}",
        rule,
        f"  n = {cm.total}   (UNANSWERABLE is the positive class)",
        "",
        f"  accuracy   {cm.accuracy:.3f}",
        f"  precision  {cm.precision:.3f}",
        f"  recall     {cm.recall:.3f}",
        f"  F1         {cm.f1:.3f}",
        "",
        *cm.rows(),
        "",
        "  fn = gold UNANSWERABLE answered anyway -- the hallucination risk.",
        "  fp = gold ANSWERABLE refused -- the cost of a conservative threshold.",
    ]

    by_class = report.recall_by_class()
    if by_class:
        out += ["", "RECALL BY UNANSWERABLE CLASS", rule]
        out.append(f"  {'class':<20}{'recall':>10}{'n':>6}")
        for klass, (recall, n) in by_class.items():
            note = "  <-- the real test" if klass == "near-miss" else ""
            out.append(f"  {klass:<20}{recall:>10.3f}{n:>6}{note}")
        out += [
            "",
            "  Out-of-scope questions are trivially rejected; near-miss is where a",
            "  threshold either works or does not. A good aggregate F1 can hide a",
            "  complete failure on this row.",
        ]

    by_qt = report.abstention_by_query_type()
    if by_qt:
        out += ["", "ABSTENTION BY QUERY TYPE", rule]
        out.append(f"  {'query type':<16}{'abstained':>12}{'over-abstained':>16}{'n':>6}")
        for qt, (rate, over, n) in by_qt.items():
            out.append(f"  {qt:<16}{rate:>12.3f}{over:>16.3f}{n:>6}")
        out += [
            "",
            "  A global threshold that abstains more on one query type is refusing",
            "  users for their language rather than for missing evidence.",
        ]
    return out
