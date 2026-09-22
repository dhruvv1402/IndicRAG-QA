"""Re-anchor gold citations from one segmentation version to another.

Passage IDs are positional (`doc#p0007` is the eighth passage of `doc`), so a
change to segmentation changes what an ID *means* without breaking it. Moving
from the frozen corpus (v1) to the corrected one (v2) keeps every cited ID
resolvable while 312 of them come to name different text, and 123 of the 320
answerable gold items cite one. Nothing fails; a third of the gold set simply
starts pointing at the wrong passage. PLAN §10.1a records that as the reason the
corpus is frozen.

This module is the other half of un-freezing it. For every cited passage it
finds where that passage's *text* went under the new segmentation, and says how
sure it is:

    unchanged      same ID, same text -- nothing to do
    moved          identical text under a different ID
    contained      the old text sits (almost) whole inside one new passage
    split-answer   the old text was divided; exactly one piece carries the answer
    review         divided, and the answer does not single out one piece

Only the first four are rewritten. `review` items keep their old citation and are
listed, because guessing there is exactly the silent re-pointing this exists to
prevent. Nothing here writes `evals/gold.jsonl`: the result is a proposed file
and a report, and adopting it is a decision for whoever verifies the gold set.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from ..models import Passage, QAItem

#: Share of the old passage's tokens a new passage must hold to count as
#: containing it. Below 1.0 because overlap re-packing moves a sentence or two
#: across a boundary without changing which passage the content belongs to.
CONTAINED = 0.9

#: Share of the old passage's tokens a new passage must hold to be a candidate
#: piece of it at all.
PIECE = 0.2

#: Share of the answer's tokens a piece must hold to count as carrying it.
ANSWER = 0.8

_TOKEN_RE = re.compile(r"[\wऀ-ॿ]+")

STATUSES = ("unchanged", "moved", "contained", "split-answer", "review")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _cover(part: set[str], whole: set[str]) -> float:
    """Share of `part` found in `whole`."""
    return len(part & whole) / len(part) if part else 0.0


@dataclass
class Citation:
    """Where one cited passage went."""

    old_id: str
    status: str
    new_ids: list[str]
    detail: str = ""


@dataclass
class Reanchored:
    item_id: str
    citations: list[Citation]

    @property
    def status(self) -> str:
        """The item's status is its least certain citation's."""
        return max((c.status for c in self.citations), key=STATUSES.index)

    @property
    def new_ids(self) -> list[str]:
        out: list[str] = []
        for c in self.citations:
            for pid in c.new_ids:
                if pid not in out:
                    out.append(pid)
        return out


@dataclass
class ReanchorResult:
    items: list[Reanchored] = field(default_factory=list)
    #: IDs that exist in both versions but name different text. The count that
    #: makes re-segmenting dangerous, independent of whether any item cites one.
    repointed_ids: int = 0

    def counts(self) -> Counter:
        return Counter(r.status for r in self.items)

    def silently_wrong(self) -> list[Reanchored]:
        """Items whose old citation still resolves under the new version but to
        different text -- what adopting v2 without this pass would have done."""
        return [r for r in self.items if r.status != "unchanged"]


def _locate(
    old: Passage,
    new_by_id: dict[str, Passage],
    new_in_doc: Sequence[Passage],
    answers: Iterable[str],
) -> Citation:
    same = new_by_id.get(old.passage_id)
    if same is not None and same.text == old.text:
        return Citation(old.passage_id, "unchanged", [old.passage_id])

    for p in new_in_doc:
        if p.text == old.text:
            return Citation(old.passage_id, "moved", [p.passage_id], f"identical text at {p.passage_id}")

    old_toks = _tokens(old.text)
    scored = sorted(
        ((p, _cover(old_toks, _tokens(p.text))) for p in new_in_doc),
        key=lambda x: -x[1],
    )
    if scored and scored[0][1] >= CONTAINED:
        p, c = scored[0]
        return Citation(old.passage_id, "contained", [p.passage_id], f"{c:.0%} of its tokens in {p.passage_id}")

    pieces = [(p, c) for p, c in scored if c >= PIECE]
    answer_toks = [t for t in (_tokens(a) for a in answers) if t]
    carriers = [
        p for p, _c in pieces
        if any(_cover(a, _tokens(p.text)) >= ANSWER for a in answer_toks)
    ]
    shown = ", ".join(f"{p.passage_id} {c:.0%}" for p, c in pieces[:4]) or "no piece above threshold"
    if len(carriers) == 1:
        return Citation(old.passage_id, "split-answer", [carriers[0].passage_id], f"pieces: {shown}")
    why = "no piece carries the answer" if not carriers else f"{len(carriers)} pieces carry the answer"
    return Citation(old.passage_id, "review", [old.passage_id], f"{why}; pieces: {shown}")


def reanchor(
    items: Sequence[QAItem],
    old_passages: Sequence[Passage],
    new_passages: Sequence[Passage],
) -> ReanchorResult:
    """Map every item's `gold_passage_ids` from `old_passages` to `new_passages`."""
    old_by_id = {p.passage_id: p for p in old_passages}
    new_by_id = {p.passage_id: p for p in new_passages}
    new_by_doc: dict[str, list[Passage]] = defaultdict(list)
    for p in new_passages:
        new_by_doc[p.doc_id].append(p)

    result = ReanchorResult(
        repointed_ids=sum(
            1 for pid, p in old_by_id.items()
            if pid in new_by_id and new_by_id[pid].text != p.text
        )
    )
    for item in items:
        if not item.gold_passage_ids:
            continue
        cites = []
        for pid in item.gold_passage_ids:
            old = old_by_id.get(pid)
            if old is None:
                cites.append(Citation(pid, "review", [pid], "not in the old corpus"))
                continue
            cites.append(
                _locate(old, new_by_id, new_by_doc[old.doc_id], (item.answer_gold, item.answer_gold_hi))
            )
        result.items.append(Reanchored(item.id, cites))
    return result


def apply(items: Sequence[QAItem], result: ReanchorResult, *, version: int) -> list[QAItem]:
    """The gold set with every confidently located citation rewritten.

    `review` citations are left as they were, and the item is marked unverified
    either way: a citation that moved is a claim about the new corpus that no
    annotator has checked.
    """
    by_item = {r.item_id: r for r in result.items}
    out: list[QAItem] = []
    for item in items:
        r = by_item.get(item.id)
        if r is None or r.status == "unchanged":
            out.append(item)
            continue
        note = f"reanchored to segmentation v{version}: {r.status}"
        out.append(
            replace(
                item,
                gold_passage_ids=r.new_ids,
                verified=False,
                notes=f"{item.notes}; {note}" if item.notes else note,
            )
        )
    return out


def format_reanchor(result: ReanchorResult, *, old_version: int, new_version: int) -> list[str]:
    counts = result.counts()
    total = len(result.items)
    lines = [
        f"RE-ANCHORING GOLD CITATIONS -- segmentation v{old_version} -> v{new_version}",
        "-" * 78,
        f"  {result.repointed_ids} passage IDs exist in both versions but name different text.",
        f"  {total} items cite at least one passage.",
        "",
    ]
    for status in STATUSES:
        lines.append(f"  {status:<14}{counts.get(status, 0):>5}")
    wrong = result.silently_wrong()
    lines += [
        "",
        f"  {len(wrong)} items would cite changed or moved text if v{new_version} were adopted",
        "  without this pass.",
    ]
    review = [r for r in result.items if r.status == "review"]
    if review:
        lines += ["", "NEEDS REVIEW -- citation kept as it was", "-" * 78]
        for r in review:
            for c in r.citations:
                if c.status == "review":
                    lines.append(f"  {r.item_id:<22}{c.old_id:<30}{c.detail}")
    return lines
