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


def enriched_sample(items, n: int, *, seed: int = 20260922) -> list:
    """Every unanswerable item, filled to `n` with answerable ones.

    Proportional sampling is right for a split and wrong for this. The
    unanswerable classes are 80 of 400 and false-premise is 15 of those, so a
    proportional sample of any affordable size leaves two or three
    false-premise items in the reported half -- and per-class recall is the
    Module 5 result, not the aggregate. At n=120 proportional, the test split
    holds 3 false-premise and 2 under-specified items, which measures nothing.

    The cost is that the class balance no longer matches the full set, so the
    base rate changes and precision is not comparable to a proportional run.
    That is a statement the report has to make, and it is a better trade than a
    per-class table nobody can read.
    """
    import random

    unanswerable = [i for i in items if not i.answerable]
    answerable = sorted((i for i in items if i.answerable), key=lambda i: i.id)
    random.Random(seed).shuffle(answerable)
    take = max(0, n - len(unanswerable))
    return sorted(unanswerable + answerable[:take], key=lambda i: i.id)


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
    from ..pipeline import Retrievers, retrieve
    from ..query.langid import classify

    say = progress or (lambda _m: None)
    cfg = get_settings()

    retrievers = Retrievers(lexical=LexicalIndex.load(cfg.lex_dir))

    # Only load the encoder when the method actually needs it. A dense model is
    # roughly a gigabyte resident, and `--method bm25` is the configuration that
    # runs when memory is short -- loading a model it will never call defeats
    # the point of offering it.
    if method.lower() not in {"bm25", "tfidf"}:
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
        "features": features,
        "scores": [f.max_score for f in features],
        "method": method,
    }


def format_separation(scores, labels, *, method: str = "") -> list[str]:
    """Do answerable and unanswerable questions even score differently?

    This table belongs beside the confusion matrix because without it the
    threshold result is unreadable. A fitted threshold can be made to report
    0.000 recall on every unanswerable class or 0.9+ on all of them, depending
    only on where it lands, and neither number says anything about the signal if
    the two score distributions are the same. Reporting an operating point
    without the separation behind it is how a flat curve gets written up as a
    finding about one class.
    """
    import statistics

    rule = "-" * 78
    ans = [s for s, lab in zip(scores, labels, strict=True) if lab]
    una = [s for s, lab in zip(scores, labels, strict=True) if not lab]
    if not ans or not una:
        return []

    a_med, u_med = statistics.median(ans), statistics.median(una)
    base_rate = len(una) / len(scores)

    out = [
        f"SCORE SEPARATION{f' -- {method}' if method else ''}",
        rule,
        "  Top retrieval score, by gold label. If these two rows agree, no",
        "  threshold over this score can separate the classes at any setting.",
        "",
        f"  {'':<16}{'median':>10}{'mean':>10}{'n':>7}",
        f"  {'answerable':<16}{a_med:>10.4f}{statistics.fmean(ans):>10.4f}{len(ans):>7}",
        f"  {'UNanswerable':<16}{u_med:>10.4f}{statistics.fmean(una):>10.4f}{len(una):>7}",
        "",
        f"  difference in medians: {a_med - u_med:+.4f}",
    ]
    if u_med >= a_med:
        out += [
            "",
            "  The unanswerable questions score AT LEAST AS HIGH as the answerable",
            "  ones, which is the opposite of the direction a threshold assumes.",
            "  This follows from how the taxonomy is built: near-miss and",
            "  false-premise questions are deliberately about schemes that are IN",
            "  the corpus, so they retrieve just as well. Only out-of-scope items",
            "  are about absent topics. The signal is not weak here, it is absent.",
        ]

    out += [
        "",
        f"  Base rate of UNANSWERABLE: {base_rate:.3f}. A precision at or near this",
        "  value means the threshold is abstaining about as usefully as chance.",
    ]
    return out


def tau_sweep(features, labels, *, grid: int = 20) -> list[tuple[float, float, float, float, float]]:
    """(tau, precision, recall, F1, abstention rate) across the threshold range.

    The sweep is the honest artefact. A single fitted tau reports one point on
    this curve and cannot show whether the curve is flat.
    """
    from ..answerability.signals import ThresholdSignal

    scale = max((f.max_score for f in features), default=1.0) or 1.0
    rows = []
    for i in range(grid + 1):
        tau = i / grid
        signal = ThresholdSignal(tau=tau, scale=scale)
        preds = [signal.predict_answerable(f) for f in features]
        tp = sum(1 for p, g in zip(preds, labels, strict=True) if not p and not g)
        fp = sum(1 for p, g in zip(preds, labels, strict=True) if not p and g)
        fn = sum(1 for p, g in zip(preds, labels, strict=True) if p and not g)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((tau, precision, recall, f1, sum(1 for p in preds if not p) / len(preds)))
    return rows


def format_tau_sweep(rows) -> list[str]:
    rule = "-" * 78
    out = [
        "THRESHOLD SWEEP",
        rule,
        "  Every operating point, so the shape of the trade-off is visible rather",
        "  than one fitted value standing in for it.",
        "",
        f"  {'tau':>6}{'precision':>11}{'recall':>9}{'F1':>8}{'abstains':>10}",
    ]
    best = max(rows, key=lambda r: r[3]) if rows else None
    for tau, precision, recall, f1, abstain in rows:
        mark = "  <-- best F1" if best and f1 == best[3] and f1 > 0 else ""
        out.append(
            f"  {tau:>6.3f}{precision:>11.3f}{recall:>9.3f}{f1:>8.3f}{abstain:>10.3f}{mark}"
        )
    if best:
        out += [
            "",
            f"  Best F1 {best[3]:.3f} is reached at {best[4]:.1%} abstention. Read those two",
            "  together: a high recall on the unanswerable class bought by refusing",
            "  most answerable questions is not detection, it is silence.",
        ]
    return out


def separation_across_methods(passages, items, *, methods=("bm25", "tfidf"), k: int = 5):
    """Score separation under several retrievers, not just the configured one.

    The claim in the paper is that *retrieval scores* carry no answerability
    signal, not that one particular fusion does. That is a claim about more than
    one retriever, so the report has to show more than one. BM25 and TF-IDF cost
    nothing extra here -- they need no encoder -- and they are the two where the
    unanswerable questions score visibly higher, which the fused score obscures
    by being nearly constant.

    Returns {method: (answerable_median, unanswerable_median, best_f1, precision)}.
    """
    import statistics

    from ..answerability.signals import extract_features
    from ..config import get_settings
    from ..index.lexical import LexicalIndex
    from ..pipeline import Retrievers, retrieve

    cfg = get_settings()
    retrievers = Retrievers(lexical=LexicalIndex.load(cfg.lex_dir))
    scheme_of = {p.passage_id: p.scheme for p in passages}
    labels = [i.answerable for i in items]

    out: dict[str, tuple[float, float, float, float]] = {}
    for method in methods:
        features = [
            extract_features(
                retrieve(i.question, retrievers, method=method, k=k), scheme_of=scheme_of, k=k
            )
            for i in items
        ]
        scores = [f.max_score for f in features]
        ans = [s for s, lab in zip(scores, labels, strict=True) if lab]
        una = [s for s, lab in zip(scores, labels, strict=True) if not lab]
        if not ans or not una:
            continue
        best = max(tau_sweep(features, labels), key=lambda r: r[3])
        out[method] = (statistics.median(ans), statistics.median(una), best[3], best[1])
    return out


def format_separation_across_methods(table) -> list[str]:
    rule = "-" * 78
    if not table:
        return []
    out = [
        "SCORE SEPARATION ACROSS RETRIEVERS",
        rule,
        "  The claim is about retrieval scores in general, not one fusion, so it",
        "  is shown for more than one retriever.",
        "",
        f"  {'method':<12}{'answerable':>12}{'UNanswerable':>14}{'best F1':>10}{'precision':>11}",
    ]
    for method, (a_med, u_med, f1, precision) in table.items():
        flag = "  <-- UNanswerable higher" if u_med > a_med else ""
        out.append(
            f"  {method:<12}{a_med:>12.4f}{u_med:>14.4f}{f1:>10.3f}{precision:>11.3f}{flag}"
        )
    return out


def auc(positive, negative) -> float:
    """P(a random positive ranks above a random negative), ties counted as half.

    The Mann-Whitney U statistic, normalised. Reported instead of a difference
    in medians because medians can agree while the distributions differ, and can
    differ while the distributions overlap almost entirely -- both of which
    happen in this data. 0.5 is chance.
    """
    merged = sorted([(v, 0) for v in positive] + [(v, 1) for v in negative])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        shared = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[t] = shared
        i = j + 1

    n_pos, n_neg = len(positive), len(negative)
    if not n_pos or not n_neg:
        return 0.5
    rank_sum = sum(ranks[t] for t, (_v, group) in enumerate(merged) if group == 0)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


#: The retrieval-side features the calibrated signal is fit over.
FEATURE_FIELDS = ("max_score", "margin", "mean_top_k", "score_spread", "scheme_agreement")


def feature_separation(features, labels) -> dict[str, tuple[float, float, float]]:
    """{feature: (answerable median, unanswerable median, AUC)}.

    Every feature, not only the one the threshold uses. ARCHITECTURE §12.2
    justifies including the top1-top2 margin on the reasoning that a near-miss
    question retrieves several similarly-scoring passages and so shows a small
    margin. That is a testable claim, and the calibrated combination in signal 4
    rests on it, so it is measured here rather than assumed.
    """
    import statistics

    out: dict[str, tuple[float, float, float]] = {}
    for field_name in FEATURE_FIELDS:
        ans = [getattr(f, field_name) for f, lab in zip(features, labels, strict=True) if lab]
        una = [getattr(f, field_name) for f, lab in zip(features, labels, strict=True) if not lab]
        if not ans or not una:
            continue
        out[field_name] = (statistics.median(ans), statistics.median(una), auc(ans, una))
    return out


def format_feature_separation(table, *, method: str = "") -> list[str]:
    rule = "-" * 78
    if not table:
        return []
    out = [
        f"FEATURE SEPARATION{f' -- {method}' if method else ''}",
        rule,
        "  AUC is P(a random answerable question scores above a random",
        "  unanswerable one). 0.500 is chance. This is what the calibrated",
        "  signal has to work with: a logistic regression cannot extract more",
        "  information than its features carry.",
        "",
        f"  {'feature':<20}{'answerable':>12}{'UNanswerable':>14}{'AUC':>8}",
    ]
    for name, (a_med, u_med, area) in table.items():
        note = "  <-- separates" if abs(area - 0.5) > 0.10 else ""
        out.append(f"  {name:<20}{a_med:>12.4f}{u_med:>14.4f}{area:>8.3f}{note}")

    best = max(table.values(), key=lambda v: abs(v[2] - 0.5))
    out += [
        "",
        f"  Best single feature reaches AUC {best[2]:.3f}. Combining them recovers",
        "  somewhat more than any one of them -- see the calibrated section below --",
        "  but a combination is bounded by what its inputs carry, and every column",
        "  here is closer to chance than to a usable signal.",
    ]
    return out


def run_calibrated(features, labels, dev, test, *, items=None):
    """ARCHITECTURE §12 signal 4, over the retrieval features alone.

    Fit on the same split as the threshold so the two are comparable. The
    generator and NLI columns of the feature vector are left at zero here: this
    is the retrieval-side ceiling, answering what the best possible combination
    of score geometry can do before any model reads a passage.

    Returns `(report, meta)` in the shape `run_answerability` uses.
    """
    from ..answerability.calibrate import CombinedFeatures, fit
    from ..answerability.signals import unanswerable_f1
    from ..query.langid import classify

    combined = [CombinedFeatures(retrieval=f) for f in features]
    signal, dev_f1 = fit([combined[n] for n in dev], [labels[n] for n in dev])

    report = AnswerabilityReport(system=f"calibrated, retrieval features (p>={signal.threshold:.2f})")
    for n in test:
        item = items[n] if items else None
        report.outcomes.append(
            AnswerabilityOutcome(
                item_id=item.id if item else str(n),
                gold_answerable=labels[n],
                pred_answerable=signal.predict_answerable(combined[n]),
                query_type=classify(item.question).query_type if item else "",
                unanswerable_class=(item.unanswerable_class or "") if item else "",
            )
        )

    probabilities = [signal.probability(combined[n]) for n in test]
    test_labels = [labels[n] for n in test]
    return report, {
        "signal": signal,
        "dev_f1": dev_f1,
        "auc": auc(
            [p for p, lab in zip(probabilities, test_labels, strict=True) if lab],
            [p for p, lab in zip(probabilities, test_labels, strict=True) if not lab],
        ),
        "test_f1": unanswerable_f1(
            [signal.predict_answerable(combined[n]) for n in test], test_labels
        ),
    }


def format_calibrated(report, meta, *, best_single_auc: float | None = None) -> list[str]:
    from ..answerability.calibrate import format_weights

    rule = "-" * 78
    out = [
        "CALIBRATED COMBINATION (signal 4, retrieval features only)",
        rule,
        "  Fit on the same split as the threshold, so the two are comparable.",
        "  The generator and NLI columns are zero here: this is the ceiling for",
        "  score geometry alone, before any model reads a passage.",
        "",
        f"  dev F1              {meta['dev_f1']:.3f}",
        f"  test F1             {meta['test_f1']:.3f}",
        f"  AUC of p(answerable){meta['auc']:>8.3f}",
    ]
    if best_single_auc is not None:
        gain = meta["auc"] - best_single_auc
        out += [
            f"  best single feature {best_single_auc:>8.3f}   (combination gains {gain:+.3f})",
        ]
    out += ["", *format_weights(meta["signal"])]
    return out
