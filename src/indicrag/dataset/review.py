"""Mechanical pre-checks for the gold set, run before a human verifies it.

`dataset verify` puts one item at a time in front of an annotator, and PRD §6.5
requires a person to confirm every one. That requirement stands: nothing here
sets `verified`, and nothing here should be read as having checked an item. What
it does is make the human pass cheaper and more consistent, by finding the
defects a rule can find before anyone has to notice them by eye four hundred
times in a row:

    not-standalone      the question leans on "the passage", "this scheme", ...
    language-mismatch   the question is not in the language its cell says
    answer-unsupported  too little of the gold answer appears in the evidence
    answer-truncated    unbalanced brackets or a dangling final word
    answer-verbose      a sentence restating the question, not a span
    missing-hindi       Hindi evidence but no gold answer in Devanagari
    placeholder         the question is a stub, not a question
    high-overlap        the question copies its evidence (PRD §6.6 leakage)
    duplicate           the same question, normalised, appears twice. The
                        unanswerable scaffold cycles its stems, so 25
                        out-of-scope items hold 9 questions and 10
                        under-specified hold 3; per-class figures on them
                        count repeats as independent evidence
    inconsistent        an unanswerable item that still cites evidence, or
                        an answerable one that cites none

Each finding carries the measured value, so a reviewer can disagree with the
threshold rather than with an opaque flag.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..evaluation.probes import token_overlap
from ..models import Passage, QAItem
from ..query.langid import classify

#: Phrases that only make sense to someone looking at the source passage. A
#: retrieval question has no passage in front of it, so "according to the
#: passage" leaves it unanswerable as asked -- and it was the bootstrapping
#: prompt's habit, so it recurs.
_DEICTIC = re.compile(
    r"\b(according to (the|this) (passage|text|article|document)|(the|this) passage|"
    r"(the|this) text|mentioned (above|here|in the)|the above|given (passage|text)|"
    r"passage (ke|me|mein|men) |is passage)\b"
    r"|गद्यांश|अनुच्छेद के अनुसार|इस पाठ|उपरोक्त",
    re.IGNORECASE,
)

#: A gold answer this long is almost always a sentence restating the question.
VERBOSE_TOKENS = 25

#: Share of the answer's tokens that must appear in the evidence.
SUPPORTED = 0.6

#: Jaccard overlap above which a same-script question reads as copied.
HIGH_OVERLAP = 0.6

_EXPECTED_LANG = {"en": "en", "hi": "hi", "hinglish": "hi-en"}
_TOKEN = re.compile(r"[\wऀ-ॿ]+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_DANGLING = {"the", "a", "an", "and", "or", "of", "to", "with", "for", "in", "by", "के", "की", "का", "और"}


@dataclass
class Finding:
    code: str
    detail: str


@dataclass
class ItemReview:
    item_id: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def codes(self) -> list[str]:
        return [f.code for f in self.findings]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.item_id, "findings": [asdict(f) for f in self.findings]}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _coverage(answer: str, evidence: str) -> float:
    toks = _tokens(answer)
    if not toks:
        return 0.0
    have = set(_tokens(evidence))
    return sum(1 for t in toks if t in have) / len(toks)


def _truncated(answer: str) -> str | None:
    if answer.count("(") != answer.count(")"):
        return "unbalanced parentheses"
    if answer.count('"') % 2:
        return "unbalanced quotation marks"
    words = answer.rstrip(" .,;:।").split()
    if words and words[-1].lower() in _DANGLING:
        return f"ends on '{words[-1]}'"
    return None


def _norm_question(q: str) -> str:
    return " ".join(_tokens(q))


def review_item(item: QAItem, passages: dict[str, Passage]) -> ItemReview:
    r = ItemReview(item.id)
    add = r.findings.append

    if len(_tokens(item.question)) <= 2:
        add(Finding("placeholder", f"question is {item.question!r}"))

    if _DEICTIC.search(item.question):
        add(Finding("not-standalone", f"refers to its source: {_DEICTIC.search(item.question).group(0)!r}"))

    want = _EXPECTED_LANG.get(item.query_lang)
    got = classify(item.question)
    if want and got.lang != want:
        add(Finding("language-mismatch", f"cell says {item.query_lang}, classifier reads {got.lang} ({got.reason})"))

    if not item.answerable:
        if item.gold_passage_ids or item.answer_gold:
            add(Finding("inconsistent", "unanswerable, but cites evidence or carries an answer"))
        return r
    if not item.gold_passage_ids:
        add(Finding("inconsistent", "answerable, but cites no evidence"))
        return r

    evidence = [passages[p] for p in item.gold_passage_ids if p in passages]
    if len(evidence) < len(item.gold_passage_ids):
        add(Finding("inconsistent", "cites a passage that is not in the corpus"))
    text = " ".join(p.text for p in evidence)

    hindi_evidence = any(p.lang == "hi" for p in evidence)
    # The scorer takes the best of both answer fields (evaluation/qa_run.py), so
    # a Hindi answer in answer_gold is fine; what fails is Hindi evidence with
    # an English-worded answer and no Devanagari one, which Exact Match against
    # a Hindi generation can never credit. Bare numbers need neither script.
    answers = item.answer_gold + " " + item.answer_gold_hi
    if hindi_evidence and _LATIN_WORD.search(answers) and not _DEVANAGARI.search(answers):
        add(Finding("missing-hindi", "evidence is Hindi, the gold answer is worded only in English"))

    candidates = [a for a in (item.answer_gold, item.answer_gold_hi) if a.strip()]
    if not candidates:
        add(Finding("answer-unsupported", "no gold answer at all"))
    else:
        best = max(_coverage(a, text) for a in candidates)
        if best < SUPPORTED:
            add(Finding("answer-unsupported", f"{best:.0%} of the answer's tokens are in the evidence"))
        for a in candidates:
            why = _truncated(a)
            if why:
                add(Finding("answer-truncated", why))
                break
        if len(_tokens(item.answer_gold)) > VERBOSE_TOKENS:
            add(Finding("answer-verbose", f"{len(_tokens(item.answer_gold))} tokens"))

    same_script = (item.query_lang == "hi") == hindi_evidence and item.query_lang != "hinglish"
    if same_script:
        overlap = token_overlap(item.question, text)
        if overlap >= HIGH_OVERLAP:
            add(Finding("high-overlap", f"question-evidence Jaccard {overlap:.2f}"))
    return r


def review(items: Sequence[QAItem], passages: Sequence[Passage]) -> list[ItemReview]:
    by_id = {p.passage_id: p for p in passages}
    out = [review_item(i, by_id) for i in items]

    seen: dict[str, list[str]] = defaultdict(list)
    for i in items:
        seen[_norm_question(i.question)].append(i.id)
    index = {r.item_id: r for r in out}
    for ids in seen.values():
        if len(ids) > 1:
            for i in ids:
                others = ", ".join(x for x in ids if x != i)
                index[i].findings.append(Finding("duplicate", f"same question as {others}"))
    return out


# --- merging with a reading pass ---------------------------------------------------
#
# The rules above find what a pattern can find. A reading pass -- a reviewer, human
# or model, comparing each item with its evidence -- finds the rest: answers that
# quote the wrong sentence, questions that name no scheme, near-misses the corpus
# actually answers. Its notes come in as JSONL, one object per item with at least
# `id` and `verdict`, and are merged here with the rule findings so the annotator
# sees both in one place. `reviewed_by` travels with every record, because a note
# from a model pass is advice and must never be mistaken for a verification.

VERDICTS = ("accept", "fix", "relabel-unanswerable", "relabel-answerable", "reject", "unsure")


def merge(
    rule_reviews: Sequence[ItemReview],
    notes: dict[str, dict[str, Any]],
    *,
    reviewed_by: str,
) -> list[dict[str, Any]]:
    """One record per item: rule findings, plus the reading pass's note if any."""
    out = []
    for r in rule_reviews:
        rec: dict[str, Any] = {"id": r.item_id, "rules": [asdict(f) for f in r.findings]}
        note = notes.get(r.item_id)
        if note is not None:
            if note.get("verdict") not in VERDICTS:
                raise ValueError(f"{r.item_id}: verdict {note.get('verdict')!r} not in {VERDICTS}")
            rec["review"] = {k: v for k, v in note.items() if k != "id" and v not in ("", [], None)}
            rec["reviewed_by"] = reviewed_by
        out.append(rec)
    return out


def format_review(records: Sequence[dict[str, Any]], items: Sequence[QAItem]) -> list[str]:
    from collections import Counter

    by_id = {i.id: i for i in items}
    lines = [
        "PRE-VERIFICATION REVIEW",
        "-" * 78,
        "  Advice for the human pass in `dataset verify`, not a verification: no",
        "  item's `verified` field is touched, and every number in evals/ is still",
        "  computed on unverified items.",
        "",
    ]
    reviewed = [r for r in records if "review" in r]
    if reviewed:
        by = sorted({r["reviewed_by"] for r in reviewed})
        lines.append(f"  reading pass: {len(reviewed)}/{len(records)} items, by {', '.join(by)}")
        lines.append("")
        lines.append("  verdict                  all    answerable  unanswerable")
        for v in VERDICTS:
            rows = [r for r in reviewed if r["review"]["verdict"] == v]
            ans = sum(1 for r in rows if by_id[r["id"]].answerable)
            lines.append(f"  {v:<22}{len(rows):>5}{ans:>14}{len(rows) - ans:>14}")
        lines.append("")
        issues = Counter(i for r in reviewed for i in r["review"].get("issues", []))
        lines.append("  issues raised by the reading pass")
        for code, n in issues.most_common():
            lines.append(f"    {code:<28}{n:>5}")
        lines.append("")
        cells = Counter()
        clean = Counter()
        for r in reviewed:
            it = by_id[r["id"]]
            cell = f"{it.query_lang}->{it.passage_lang or 'unans'}"
            cells[cell] += 1
            clean[cell] += r["review"]["verdict"] == "accept"
        lines.append("  accepted as-is, by cell")
        for cell in sorted(cells):
            lines.append(f"    {cell:<18}{clean[cell]:>4} / {cells[cell]:<4}")
        lines.append("")
    rules = Counter(f["code"] for r in records for f in r["rules"])
    lines.append(f"  rule findings: {sum(1 for r in records if r['rules'])} items flagged")
    for code, n in rules.most_common():
        lines.append(f"    {code:<28}{n:>5}")
    return lines
