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


# --- running the evaluation ------------------------------------------------------
#
# This lives here rather than in the CLI so that `eval answerability` and
# `eval all` share one implementation. It previously sat in the CLI, which is
# precisely why `eval all` -- the command PRD NFR-6 points at for regenerating
# every committed table -- had no answerability stage and this report could only
# be produced by remembering to run a second command.


def stratified_by_class(items, n: int, *, seed: int = 20260922) -> list:
    """Seeded draw of `n` items taking the same *share* of each class.

    Proportional, and stratified on the label rather than on anything else,
    because both obvious alternatives were observed to fail. Fitting on a prefix
    sorted by item id put every unanswerable item in the test half -- ids are
    prefixed by kind -- so the fit saw no positive examples, every threshold
    scored F1 = 0, and the search returned a threshold that never abstains.
    Round-robin across strata produced the mirror image: the unanswerable
    classes are the small cells, so it drained all of them into the fitting half
    and left the reported half entirely answerable.

    Either way one half is single-class and the resulting table reads like a
    finding about the signal.
    """
    import random
    from collections import defaultdict

    if n <= 0 or n >= len(items):
        return list(items)

    cells: dict[str, list] = defaultdict(list)
    for item in items:
        cells["answerable" if item.answerable else (item.unanswerable_class or "other")].append(
            item
        )

    rng = random.Random(seed)
    fraction = n / len(items)
    out: list = []
    for key in sorted(cells):
        group = sorted(cells[key], key=lambda i: i.id)
        rng.shuffle(group)
        take = min(len(group), max(1, round(len(group) * fraction)))
        out.extend(group[:take])
    return sorted(out, key=lambda i: i.id)


def run_answerability(
    passages,
    items,
    *,
    method: str = "hybrid",
    k: int = 5,
    dev_fraction: float = 0.3,
    progress=None,
):
    """Fit the retrieval-score threshold and report it on held-out items.

    Returns `(report, meta)` where meta carries the fitted signal, the dev F1
    and the index lists, so a caller that also wants the generator self-report
    can reuse the same split rather than drawing its own.
    """
    from ..answerability.signals import ThresholdSignal, extract_features
    from ..config import get_settings
    from ..index.lexical import LexicalIndex
    from ..models import Passage  # noqa: F401 -- documents the expected type
    from ..pipeline import Retrievers, retrieve
    from ..query.langid import classify

    say = progress or (lambda _m: None)
    cfg = get_settings()

    retrievers = Retrievers(lexical=LexicalIndex.load(cfg.lex_dir))
    try:
        from ..index.dense import DenseIndex, Encoder
        from ..index.encoders import get

        spec = get(cfg.encoder_primary)
        retrievers.dense = DenseIndex.load(spec, cfg.emb_dir, passages)
        retrievers.encoder = Encoder(spec)
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        say(f"  dense index unavailable ({exc}); '{method}' falls back to lexical")

    retrievers.script_of = {
        p.passage_id: ("deva" if p.lang == "hi" else "latin") for p in passages
    }
    scheme_of = {p.passage_id: p.scheme for p in passages}

    say(f"  retrieving for {len(items)} items ({method}, k={k})")
    hits = [retrieve(i.question, retrievers, method=method, k=k) for i in items]
    features = [extract_features(h, scheme_of=scheme_of, k=k) for h in hits]
    labels = [i.answerable for i in items]

    dev_ids = {
        i.id for i in stratified_by_class(items, max(1, int(len(items) * dev_fraction)))
    }
    dev = [n for n, i in enumerate(items) if i.id in dev_ids]
    test = [n for n, i in enumerate(items) if i.id not in dev_ids]

    signal, dev_f1 = ThresholdSignal.fit([features[n] for n in dev], [labels[n] for n in dev])
    report = AnswerabilityReport(system=f"threshold tau={signal.tau:.3f} ({method})")
    for n in test:
        item = items[n]
        report.outcomes.append(
            AnswerabilityOutcome(
                item_id=item.id,
                gold_answerable=item.answerable,
                pred_answerable=signal.predict_answerable(features[n]),
                query_type=classify(item.question).query_type,
                unanswerable_class=item.unanswerable_class or "",
            )
        )

    return report, {
        "signal": signal,
        "dev_f1": dev_f1,
        "dev": dev,
        "test": test,
        "hits": hits,
        "labels": labels,
    }
