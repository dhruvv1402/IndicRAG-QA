"""Second-pass re-labelling and Cohen's kappa.

`docs/PRD.md` §6.5 step 5 requires a 15% sample to be independently re-labelled
for the `answerable` field, with kappa reported, and §10.2 makes **kappa >= 0.70
a gate**: below it the answerable/unanswerable boundary is not well enough defined
for any Module 5 number to mean anything, and the guidelines need tightening
before the main results can be trusted.

The design problem is anchoring. If the re-labeller can see the first pass's
verdict, they will agree with it, and the resulting kappa measures compliance
rather than agreement. So the second pass hides the original label, the answer,
and the `unanswerable_class` -- everything that encodes the first decision -- and
shows only the question and its evidence.

Sampling is seeded and stratified over the four unanswerable classes plus the
answerable pool, because an unstratified 15% draw from a set that is 20%
unanswerable can easily contain two false-premise items, and kappa computed
across a sample missing the hard classes says nothing about the boundary that
actually matters.

The kappa itself lives in `verify.py` next to the other annotation helpers; this
module handles drawing the sample, storing the independent labels, and comparing
them.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..models import QAItem, read_jsonl
from .verify import cohens_kappa

KAPPA_GATE = 0.70
DEFAULT_FRACTION = 0.15


@dataclass
class Agreement:
    """Comparison of two independent labellings of the same items."""

    kappa: float
    n: int
    agreed: int
    first_answerable: int
    second_answerable: int
    disagreements: list[tuple[str, bool, bool]]

    @property
    def observed(self) -> float:
        return self.agreed / self.n if self.n else 0.0

    @property
    def passes_gate(self) -> bool:
        return self.kappa >= KAPPA_GATE


def draw_sample(
    items: Sequence[QAItem],
    *,
    fraction: float = DEFAULT_FRACTION,
    seed: int = 20260922,
) -> list[QAItem]:
    """Stratified, seeded sample of verified items for re-labelling.

    Strata are the four unanswerable classes and the answerable pool. A flat
    sample would routinely miss false-premise items -- 15 of 400 -- and kappa is
    only interesting where the boundary is hard.
    """
    verified = [i for i in items if i.verified and not i.notes.startswith("REJECTED")]
    if not verified:
        return []

    strata: dict[str, list[QAItem]] = defaultdict(list)
    for item in verified:
        strata["answerable" if item.answerable else (item.unanswerable_class or "other")].append(
            item
        )

    rng = random.Random(seed)
    sample: list[QAItem] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda i: i.id)
        rng.shuffle(group)
        # At least one from every stratum, so no class is silently absent.
        take = max(1, round(len(group) * fraction))
        sample.extend(group[:take])
    return sorted(sample, key=lambda i: i.id)


def blind(item: QAItem) -> dict:
    """The item as the second labeller should see it.

    Strips the first pass's verdict, the gold answer and the unanswerable class.
    Keeping any of them turns the exercise into confirmation: a re-labeller shown
    `answerable: false` will agree, and the kappa then measures compliance.
    """
    return {
        "id": item.id,
        "question": item.question,
        "query_lang": item.query_lang,
        "scheme": item.scheme,
        "gold_passage_ids": list(item.gold_passage_ids),
        "answerable": None,
    }


def write_blind_sample(sample: Sequence[QAItem], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in sample:
            fh.write(json.dumps(blind(item), ensure_ascii=False) + "\n")
    return len(sample)


def compare(first: Sequence[QAItem], second_path: Path) -> Agreement:
    """Compare the original labels against the independent second pass."""
    by_id = {i.id: i for i in first}
    pairs: list[tuple[str, bool, bool]] = []

    for raw in read_jsonl(second_path):
        if not isinstance(raw, dict):
            continue
        item_id = raw.get("id")
        label = raw.get("answerable")
        if item_id not in by_id or label is None:
            continue
        pairs.append((item_id, by_id[item_id].answerable, bool(label)))

    if not pairs:
        return Agreement(0.0, 0, 0, 0, 0, [])

    a = [p[1] for p in pairs]
    b = [p[2] for p in pairs]
    return Agreement(
        kappa=cohens_kappa(a, b),
        n=len(pairs),
        agreed=sum(1 for x, y in zip(a, b, strict=True) if x == y),
        first_answerable=sum(a),
        second_answerable=sum(b),
        disagreements=[p for p in pairs if p[1] != p[2]],
    )


def format_agreement(agreement: Agreement) -> list[str]:
    rule = "-" * 78
    out = [
        "SECOND-PASS AGREEMENT (PRD §6.5 step 5)",
        rule,
        f"  items re-labelled     {agreement.n}",
        f"  observed agreement    {agreement.observed:.3f}  ({agreement.agreed}/{agreement.n})",
        f"  answerable, pass 1    {agreement.first_answerable}",
        f"  answerable, pass 2    {agreement.second_answerable}",
        "",
        f"  Cohen's kappa         {agreement.kappa:.3f}",
        "",
    ]
    if agreement.n == 0:
        out.append("  No labels found. Fill in the `answerable` field of the blind sample first.")
        return out

    if agreement.passes_gate:
        out.append(f"  PASSES the {KAPPA_GATE:.2f} gate. Module 5 results can be trusted.")
    else:
        out += [
            f"  BELOW the {KAPPA_GATE:.2f} gate.",
            "",
            "  Do not report Module 5 numbers yet. Kappa this low means the",
            "  answerable/unanswerable boundary is not defined well enough for the",
            "  labels to mean the same thing twice -- the answerability metrics would",
            "  be measuring annotator inconsistency rather than system behaviour.",
            "  Tighten the guideline (usually the near-miss / under-specified line),",
            "  re-label, and re-measure.",
        ]

    if agreement.disagreements:
        out += ["", "DISAGREEMENTS", rule]
        for item_id, first, second in agreement.disagreements[:15]:
            out.append(
                f"  {item_id:<28} pass1={'ANS' if first else 'UNANS':<5} "
                f"pass2={'ANS' if second else 'UNANS'}"
            )
        out += [
            "",
            "  These are the items to read together. A disagreement is usually a",
            "  guideline gap, not a mistake by either labeller.",
        ]
    return out
