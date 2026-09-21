"""Module 6: structured error analysis over a fixed taxonomy.

**Cases are selected by rule, not by hand.** Picking examples after seeing which
ones look interesting is how an error analysis becomes a set of anecdotes that
happen to support whatever the author already believed. So the taxonomy is fixed
in advance (`docs/PLAN.md` §4), each category has a detector, and within a
category the selection rule is deterministic: take the failure where the system
was **most confident and most wrong**.

That rule is deliberate. A borderline miss at rank 6 tells you the ranking is
noisy, which you already knew. A confident miss -- high score, wrong passage --
tells you the model has a systematic blind spot, and it is also the case a real
user would actually be harmed by, because nothing in the output would warn them.

`failing_component` includes `annotation` as a real option. Some cases turn out
to be gold-label mistakes rather than system failures, and recording that
honestly is worth more than quietly fixing the label and moving on: the rate of
annotation error is itself a number the paper should report.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from ..corpus.normalize import canonical_amounts
from ..models import Passage, QAItem
from ..query.langid import classify
from .retrieval import Outcome

#: The fixed taxonomy. Order is the reporting order.
CATEGORIES = [
    "code-mixing",
    "transliteration-variation",
    "named-entity",
    "amount",
    "date",
    "ambiguity",
    "near-duplicate-schemes",
    "cross-lingual-gap",
    "long-tail-scheme",
    "short-query",
]

COMPONENTS = (
    "extraction",
    "segmentation",
    "query_processing",
    "retrieval",
    "generation",
    "answerability",
    "annotation",
)

_DATE_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|january|february|march|april|may|june|july|august|september|october|november|december"
    r"|तिथि|वर्ष|माह|दिनांक",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b[A-Z]{2,}\b|\b(?:Yojana|Abhiyan|Mission|Scheme|योजना|अभियान|मिशन)\b")


@dataclass
class ErrorCase:
    """One recorded failure, with everything needed to reproduce and diagnose it."""

    case_id: str
    category: str
    query: str
    query_type: str
    gold_passage_ids: list[str]
    retrieved: dict[str, list[str]] = field(default_factory=dict)
    top_score: float = 0.0
    gold_rank: int | None = None  # None means not retrieved at all
    diagnosis: str = ""
    failing_component: str = "retrieval"
    would_fix: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> ErrorCase:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def categorise(item: QAItem, passages: dict[str, Passage], scheme_sizes: dict[str, int]) -> list[str]:
    """Every category an item belongs to. Items can carry several."""
    tags: list[str] = []
    lang = classify(item.question)
    q = item.question

    if lang.query_type == "Code-Mixed":
        tags.append("code-mixing")
        if lang.script == "latin":
            tags.append("transliteration-variation")

    if item.query_lang != item.passage_lang and item.passage_lang:
        tags.append("cross-lingual-gap")

    if _ENTITY_RE.search(q):
        tags.append("named-entity")
    if canonical_amounts(q):
        tags.append("amount")
    if _DATE_RE.search(q):
        tags.append("date")
    if len(q.split()) <= 5:
        tags.append("short-query")
        tags.append("ambiguity")

    if item.scheme and scheme_sizes.get(item.scheme, 0) <= 6:
        tags.append("long-tail-scheme")

    # Several schemes describing the same kind of benefit make near-duplicate
    # passages, which is where a retriever picks the right clause from the wrong
    # scheme -- the hardest failure to notice, because the answer looks right.
    gold_schemes = {passages[p].scheme for p in item.gold_passage_ids if p in passages}
    if gold_schemes and len(gold_schemes) == 1:
        section = {
            passages[p].section_path.split(" > ")[-1].lower()
            for p in item.gold_passage_ids
            if p in passages
        }
        if section & {"eligibility", "benefits", "features", "पात्रता", "लाभ"}:
            tags.append("near-duplicate-schemes")

    return tags or ["ambiguity"]


def collect_cases(
    outcomes: Sequence[Outcome],
    items: Sequence[QAItem],
    passages: Sequence[Passage],
    *,
    k: int = 5,
    per_category: int = 2,
    other_runs: dict[str, dict[str, list[str]]] | None = None,
) -> list[ErrorCase]:
    """Select the most-confident failure in each taxonomy category.

    `other_runs` maps a system name to {item_id: retrieved ids}, so a case shows
    what every system did rather than only the one being analysed. A miss that
    every system shares points at the corpus or the question; a miss only one
    system makes points at that system.
    """
    by_id = {p.passage_id: p for p in passages}
    items_by_id = {i.id: i for i in items}
    scheme_sizes: dict[str, int] = {}
    for p in passages:
        scheme_sizes[p.scheme] = scheme_sizes.get(p.scheme, 0) + 1

    failures = [o for o in outcomes if not o.hit_at(k)]
    buckets: dict[str, list[tuple[float, Outcome]]] = {c: [] for c in CATEGORIES}

    for o in failures:
        item = items_by_id.get(o.item_id)
        if item is None:
            continue
        confidence = o.scores[0] if o.scores else 0.0
        for tag in categorise(item, by_id, scheme_sizes):
            if tag in buckets:
                buckets[tag].append((confidence, o))

    cases: list[ErrorCase] = []
    seen: set[str] = set()
    for category in CATEGORIES:
        # Most confident first: a confident miss is a systematic blind spot, and
        # the case a user would actually be harmed by.
        ranked = sorted(buckets[category], key=lambda t: -t[0])
        taken = 0
        for confidence, o in ranked:
            if o.item_id in seen or taken >= per_category:
                continue
            item = items_by_id[o.item_id]
            gold_rank = next(
                (i for i, pid in enumerate(o.retrieved, start=1) if pid in o.gold), None
            )
            runs = {"primary": list(o.retrieved[:k])}
            for system, mapping in (other_runs or {}).items():
                if o.item_id in mapping:
                    runs[system] = mapping[o.item_id][:k]

            cases.append(
                ErrorCase(
                    case_id=f"E{len(cases) + 1:02d}",
                    category=category,
                    query=item.question,
                    query_type=classify(item.question).query_type,
                    gold_passage_ids=list(o.gold),
                    retrieved=runs,
                    top_score=float(confidence),
                    gold_rank=gold_rank,
                    diagnosis=_diagnose(category, o, gold_rank, by_id),
                    failing_component="retrieval",
                    would_fix=_remedy(category),
                )
            )
            seen.add(o.item_id)
            taken += 1
    return cases


def _diagnose(category: str, outcome: Outcome, gold_rank: int | None, by_id) -> str:
    got = by_id.get(outcome.retrieved[0]) if outcome.retrieved else None
    gold = by_id.get(outcome.gold[0]) if outcome.gold else None
    where = (
        "gold passage not returned at all"
        if gold_rank is None
        else f"gold passage returned at rank {gold_rank}, below the cut"
    )
    detail = ""
    if got is not None and gold is not None:
        if got.scheme != gold.scheme:
            detail = f" Top hit is from a different scheme ({got.scheme} vs {gold.scheme})."
        elif got.lang != gold.lang:
            detail = f" Top hit is the {got.lang} side where the gold is {gold.lang}."
        else:
            detail = f" Top hit is the same scheme but a different section ({got.section_path})."
    return f"{where}.{detail}"


_REMEDIES = {
    "code-mixing": "Transliterate the query to Devanagari before lexical retrieval (H5 ablation).",
    "transliteration-variation": "Extend the scheme alias table with Romanized spelling variants.",
    "named-entity": "Boost exact entity matches in fusion, or add an entity-aware reranking pass.",
    "amount": "Index canonical amount keys alongside the text so numeric answers are matchable.",
    "date": "Normalise date expressions to a canonical form on both sides.",
    "ambiguity": "Detect under-specification and ask for the scheme rather than guessing.",
    "near-duplicate-schemes": "Condition retrieval on a detected scheme, or rerank by scheme match.",
    "cross-lingual-gap": "A stronger cross-lingual encoder; this is the H1 case.",
    "long-tail-scheme": "Corpus imbalance: the scheme has too few passages to rank reliably.",
    "short-query": "Expand short queries with scheme context before retrieval.",
}


def _remedy(category: str) -> str:
    return _REMEDIES.get(category, "")


def _recovery_summary(cases: Sequence[ErrorCase]) -> list[str]:
    """How many of these failures a single component would have got right.

    This is the number the paper's origin story rests on -- that of the hybrid
    failures, the dense retriever alone had found the gold passage in several,
    and fusion lost every one of them. It was previously hand-written prose in
    the committed report, so regenerating the cases could have changed the
    figure while the sentence quoting it stayed put. It is computed here so the
    report cannot disagree with its own data.
    """
    systems: set[str] = set()
    for case in cases:
        systems.update(k for k in (case.retrieved or {}) if k != "primary")
    if not systems:
        return []

    rule = "-" * 78
    out = ["WHAT A SINGLE COMPONENT WOULD HAVE RECOVERED", rule]
    for name in sorted(systems):
        gold_found = [
            c for c in cases
            if set(c.gold_passage_ids) & set((c.retrieved or {}).get(name, []))
        ]
        lost_by_primary = [
            c for c in gold_found
            if not (set(c.gold_passage_ids) & set((c.retrieved or {}).get("primary", [])))
        ]
        out.append(
            f"  {name:<10} found the gold in {len(gold_found):>2} of {len(cases)}; "
            f"the fused system lost {len(lost_by_primary)} of those"
        )
    out += [
        "",
        "  A fusion that discards what its stronger component got right is not a",
        "  method with a disappointing coefficient. This table is where the",
        "  script-aware correction came from.",
        "",
    ]
    return out


def format_errors(cases: Sequence[ErrorCase], passages: Sequence[Passage]) -> list[str]:
    by_id = {p.passage_id: p for p in passages}
    rule = "-" * 78
    out = [
        "MODULE 6 -- ERROR ANALYSIS",
        "=" * 78,
        f"{len(cases)} cases across {len({c.category for c in cases})} taxonomy categories.",
        "",
        "Cases are selected by rule, not by hand: within each category, the failure",
        "where the system was most confident and most wrong. A confident miss is a",
        "systematic blind spot and the case a user would actually be harmed by,",
        "because nothing in the output warns them.",
        "",
    ]
    out += _recovery_summary(cases)
    for case in cases:
        out += [rule, f"[{case.case_id}] {case.category}   ({case.query_type})", ""]
        out.append(f"  query      {case.query}")
        gold_desc = []
        for pid in case.gold_passage_ids[:2]:
            p = by_id.get(pid)
            gold_desc.append(f"{pid} ({p.scheme}/{p.lang})" if p else pid)
        out.append(f"  gold       {', '.join(gold_desc)}")
        out.append(f"  gold rank  {case.gold_rank if case.gold_rank else 'not retrieved'}")
        out.append(f"  top score  {case.top_score:.4f}")
        for system, ids in case.retrieved.items():
            shown = []
            for pid in ids[:3]:
                p = by_id.get(pid)
                shown.append(f"{p.scheme}/{p.lang}" if p else pid)
            out.append(f"  {system:<10} {' | '.join(shown)}")
        out.append(f"  diagnosis  {case.diagnosis}")
        out.append(f"  component  {case.failing_component}")
        out.append(f"  would fix  {case.would_fix}")
        out.append("")
    return out
