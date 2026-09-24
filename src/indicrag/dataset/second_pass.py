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
    #: Who produced the second labels, from each row's `labelled_by`.
    labelled_by: tuple[str, ...] = ()

    @property
    def by_model(self) -> bool:
        return any(w.startswith("model:") for w in self.labelled_by)

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


#: Passages shown per blinded item. Fixed, because a count that varied with the
#: label -- gold passages added on top of retrieval for answerable items only --
#: would give the label away as surely as the old empty list did.
EVIDENCE_SIZE = 5


def blind(
    item: QAItem, retrieved: Sequence[str] = (), *, seed: int = 20260922, size: int = EVIDENCE_SIZE
) -> dict:
    """The item as the second labeller should see it.

    Strips the first pass's verdict, the gold answer and the unanswerable class.
    Keeping any of them turns the exercise into confirmation: a re-labeller shown
    `answerable: false` will agree, and the kappa then measures compliance.

    The evidence has to be blinded too. This used to pass `gold_passage_ids`
    through, and an unanswerable item has none -- so an empty evidence list
    *was* the label, and every unanswerable item in the sample was agreed on by
    construction. Every item now carries the same kind of evidence: what the
    retriever returns for its question (`retrieved`), with the gold passages
    mixed in for answerable items, shuffled, and nothing marking which is which.
    """
    ids = list(dict.fromkeys([*item.gold_passage_ids, *retrieved]))[:size]
    random.Random(f"{seed}:{item.id}").shuffle(ids)
    return {
        "id": item.id,
        "question": item.question,
        "query_lang": item.query_lang,
        "scheme": item.scheme,
        "evidence_ids": ids,
        "answerable": None,
    }


def write_blind_sample(
    sample: Sequence[QAItem], path: Path, retrieved: dict[str, Sequence[str]] | None = None
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in sample:
            view = blind(item, (retrieved or {}).get(item.id, ()))
            fh.write(json.dumps(view, ensure_ascii=False) + "\n")
    return len(sample)


def compare(first: Sequence[QAItem], second_path: Path) -> Agreement:
    """Compare the original labels against the independent second pass."""
    by_id = {i.id: i for i in first}
    pairs: list[tuple[str, bool, bool]] = []
    who: set[str] = set()

    for raw in read_jsonl(second_path):
        if not isinstance(raw, dict):
            continue
        item_id = raw.get("id")
        label = raw.get("answerable")
        if item_id not in by_id or label is None:
            continue
        pairs.append((item_id, by_id[item_id].answerable, bool(label)))
        if raw.get("labelled_by"):
            who.add(str(raw["labelled_by"]))

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
        labelled_by=tuple(sorted(who)),
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

    if agreement.labelled_by:
        out.append(f"  second labels by: {', '.join(agreement.labelled_by)}")
        out.append("")
    if agreement.by_model:
        # The gate was written for two people. Two instances of one model share
        # its blind spots, and when the first pass also filtered out the items
        # its verifier disagreed with, agreement is close to guaranteed. The
        # number is reported, but it is not the check the gate describes.
        out += [
            f"  kappa {agreement.kappa:.3f} is agreement between model passes, not between",
            "  annotators. It shows the boundary is applied consistently by one",
            "  model; it cannot show the boundary is the one a person would draw,",
            "  which is what the gate is for. Not a pass of PRD §6.5 step 5.",
        ]
    elif agreement.passes_gate:
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


def label_blind(path: Path, passages, *, labeller: str, ask, say) -> tuple[int, int]:
    """Interactive blind labelling of a sample written by `write_blind_sample`.

    Shows each unlabelled row's question and its evidence passages -- nothing
    else: no gold answer, no first-pass verdict, no pre-review note. Records
    `answerable` and `labelled_by`, and rewrites the file after every decision
    so a sitting can stop and resume anywhere. Returns (labelled, total).
    """
    path = Path(path)
    rows = [r for r in read_jsonl(path) if isinstance(r, dict)]
    by_id = {p.passage_id: p for p in passages}

    def save() -> None:
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    say("  Read the passages under each question. Press y only if one of them states")
    say("  the answer. If none does -- even when the question is sensible, or the")
    say("  passages are on a related topic -- press n.")
    say("")
    say("  [y] a passage states the answer    [n] no passage states it")
    say("  [s] skip    [w] save and quit")
    todo = [r for r in rows if r.get("answerable") is None]
    for n, row in enumerate(todo, start=1):
        say("")
        say("=" * 78)
        say(f"[{n}/{len(todo)}]  {row['question']}")
        for k, pid in enumerate(row.get("evidence_ids", []), start=1):
            p = by_id.get(pid)
            say(f"  --- passage {k} ({pid})")
            say(f"  {p.text if p else '(passage missing from corpus)'}")
        while True:
            choice = (ask("  answerable? [y/n/s/w]> ") or "").strip().lower()
            if choice in {"y", "n"}:
                row["answerable"] = choice == "y"
                row["labelled_by"] = labeller
                save()
                break
            if choice == "s":
                break
            if choice == "w":
                done = sum(1 for r in rows if r.get("answerable") is not None)
                return done, len(rows)
    done = sum(1 for r in rows if r.get("answerable") is not None)
    return done, len(rows)
