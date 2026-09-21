"""Stratified dev/test split, and coverage against the PRD matrix.

**Dev is 120, test is 280, and the test split is sealed at the end of P3.** Every
threshold, alpha, k and prompt is tuned on dev; test is scored once, at P5.
`docs/PLAN.md` risk R13 calls contamination here fatal to the paper's
credibility, and it is the one mistake that cannot be undone after the fact --
once a number has been looked at, it cannot be unlooked at.

Stratification is over (query language x answerability x scheme). Without it a
random split leaves some cell of the §6.2 matrix with three test items, and a
per-language table built on three items reports noise with a straight face.

`coverage` exists because annotation drifts. The matrix is a target, not an
outcome: rejected candidates, empty model replies and items reclassified as
unanswerable all pull cells away from their quota. Checking at item 150 is cheap;
discovering it at item 380 means regenerating a cell from scratch.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import QAItem
from .generate import MATRIX

DEV_SIZE = 120
TEST_SIZE = 280


@dataclass
class Coverage:
    target: dict[tuple[str, str], int]
    actual: dict[tuple[str, str], int]
    unanswerable_target: dict[str, int]
    unanswerable_actual: dict[str, int]

    def deltas(self) -> dict[tuple[str, str], int]:
        return {k: self.actual.get(k, 0) - v for k, v in self.target.items()}

    def within(self, tolerance: int = 3) -> bool:
        return all(abs(d) <= tolerance for d in self.deltas().values())


#: PRD §6.3.
UNANSWERABLE_TARGET = {
    "out-of-scope": 25,
    "near-miss": 30,
    "false-premise": 15,
    "under-specified": 10,
}


def coverage(items: Sequence[QAItem]) -> Coverage:
    answerable = Counter(
        (i.query_lang, i.passage_lang) for i in items if i.answerable and not _rejected(i)
    )
    unanswerable = Counter(
        i.unanswerable_class or "unclassified"
        for i in items
        if not i.answerable and not _rejected(i)
    )
    return Coverage(
        target=dict(MATRIX),
        actual=dict(answerable),
        unanswerable_target=dict(UNANSWERABLE_TARGET),
        unanswerable_actual=dict(unanswerable),
    )


def _rejected(item: QAItem) -> bool:
    return item.notes.startswith("REJECTED")


def stratified_split(
    items: Sequence[QAItem],
    *,
    dev_size: int = DEV_SIZE,
    seed: int = 20260922,
) -> tuple[list[QAItem], list[QAItem]]:
    """Split verified items into dev and test, stratified and seeded.

    Only verified, non-rejected items are split. An unverified item must not
    reach either side: it would be scored in P5 without ever having been checked.
    """
    usable = [i for i in items if i.verified and not _rejected(i)]
    rng = random.Random(seed)

    strata: dict[tuple, list[QAItem]] = defaultdict(list)
    for item in usable:
        key = (item.query_lang, item.answerable, item.scheme)
        strata[key].append(item)

    dev: list[QAItem] = []
    test: list[QAItem] = []
    fraction = dev_size / max(1, len(usable))

    for key in sorted(strata, key=lambda k: (str(k[0]), str(k[1]), str(k[2]))):
        group = strata[key]
        rng.shuffle(group)
        # Round rather than floor: flooring every stratum systematically
        # under-fills dev, and with ~40 schemes the shortfall compounds.
        n_dev = int(round(len(group) * fraction))
        dev.extend(group[:n_dev])
        test.extend(group[n_dev:])

    for item in dev:
        item.split = "dev"
    for item in test:
        item.split = "test"
    return dev, test


def format_coverage(cov: Coverage) -> list[str]:
    out = ["DATASET COVERAGE vs PRD §6.2 / §6.3", "-" * 78]
    out.append(f"  {'cell':<22}{'target':>8}{'actual':>8}{'delta':>8}")
    for key in sorted(cov.target, key=lambda k: (k[0], k[1])):
        target = cov.target[key]
        actual = cov.actual.get(key, 0)
        flag = "" if abs(actual - target) <= 3 else "  <-- off target"
        out.append(
            f"  {key[0] + '->' + key[1]:<22}{target:>8}{actual:>8}{actual - target:>+8}{flag}"
        )
    total_t = sum(cov.target.values())
    total_a = sum(cov.actual.values())
    out.append(f"  {'TOTAL answerable':<22}{total_t:>8}{total_a:>8}{total_a - total_t:>+8}")

    out += ["", f"  {'unanswerable class':<22}{'target':>8}{'actual':>8}{'delta':>8}"]
    for key in sorted(cov.unanswerable_target):
        target = cov.unanswerable_target[key]
        actual = cov.unanswerable_actual.get(key, 0)
        out.append(f"  {key:<22}{target:>8}{actual:>8}{actual - target:>+8}")
    ut, ua = sum(cov.unanswerable_target.values()), sum(cov.unanswerable_actual.values())
    out.append(f"  {'TOTAL unanswerable':<22}{ut:>8}{ua:>8}{ua - ut:>+8}")
    out += ["", f"  GRAND TOTAL{'':<11}{total_t + ut:>8}{total_a + ua:>8}"]
    return out


# --- annotation burden -----------------------------------------------------------


def answer_provenance(items, passages) -> dict[str, int]:
    """How far each proposed answer sits from its own evidence.

    Verification effort is not uniform across items, and knowing the split before
    starting is the difference between planning a session and discovering halfway
    through that a third of it needs rewriting. An answer present verbatim is an
    accept; a close paraphrase is a trim; one sharing little with its passage has
    to be written from the passage by hand.
    """
    from ..evaluation.grounding import rouge_l_precision

    by_id = {p.passage_id: p for p in passages}
    counts = {"verbatim": 0, "near": 0, "rewrite": 0, "no_evidence": 0}
    for item in items:
        if not item.answerable or not item.answer_gold:
            continue
        evidence = " ".join(
            by_id[pid].text for pid in item.gold_passage_ids if pid in by_id
        )
        if not evidence:
            counts["no_evidence"] += 1
        elif item.answer_gold.strip() in evidence:
            counts["verbatim"] += 1
        elif rouge_l_precision(item.answer_gold, evidence) >= 0.8:
            counts["near"] += 1
        else:
            counts["rewrite"] += 1
    return counts


def format_provenance(counts: dict[str, int]) -> list[str]:
    total = sum(counts.values())
    if not total:
        return []
    rows = [
        ("verbatim in the passage", "verbatim", "accept as-is"),
        ("close paraphrase", "near", "light edit"),
        ("paraphrased or invented", "rewrite", "write from the passage"),
        ("gold passage missing", "no_evidence", "fix the reference first"),
    ]
    out = ["", "ANNOTATION BURDEN", "-" * 78]
    for label, key, action in rows:
        n = counts.get(key, 0)
        if not n:
            continue
        out.append(f"  {label:<26}{n:>5}  ({n / total:>4.0%})  {action}")
    out += [
        "",
        "  Answers were bootstrapped by a 3B model, so a proposed answer that does",
        "  not appear in its own passage is expected rather than alarming -- it is",
        "  what the verification pass exists to correct.",
    ]
    return out
