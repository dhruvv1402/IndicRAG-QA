"""The human verification loop.

`docs/PRD.md` §6.5 step 3 requires a person to confirm every item before it
counts, and §10.2 makes that a gate: items with `verified: false` are excluded
from every reported metric. This module is what makes that affordable. Four
hundred items reviewed carelessly is worse than a hundred reviewed properly,
because the resulting numbers look equally authoritative either way.

Three design choices, each aimed at the failure mode of a long annotation session:

**Saves after every decision.** A session is hours long and will be interrupted.
Losing an hour of judgements to a closed terminal is how annotation projects die,
so the file is rewritten on each keystroke rather than at the end.

**Resumable, and resumes where it stopped.** Already-verified items are skipped,
so the loop can be run repeatedly in short sittings instead of demanding one long
one. Fatigue is a real source of label noise.

**Shows the evidence, not just the question.** The gold passage is printed with
the proposed answer highlighted inside it. The commonest annotation error on this
task is accepting an answer that is *plausible* rather than one the passage
actually states, and that error is invisible unless the passage is on screen.

The second-pass mode (`--second-pass`) re-shows a sample with the previous
`answerable` label hidden, so that Cohen's kappa in §6.5 step 5 measures
agreement rather than anchoring.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..models import Passage, QAItem, read_jsonl, write_jsonl

ACTIONS = """
  [a] accept           [e] edit answer      [q] edit question
  [u] mark UNANSWERABLE                     [r] reject (drop)
  [s] skip for now     [p] show full passage
  [t] take the pre-review's suggested question/answer (then decide again)
  [w] save and quit
"""


@dataclass
class Progress:
    total: int
    verified: int
    rejected: int
    remaining: int

    def __str__(self) -> str:
        return (
            f"{self.verified} verified, {self.rejected} rejected, "
            f"{self.remaining} remaining of {self.total}"
        )


def progress_of(items: Sequence[QAItem]) -> Progress:
    verified = sum(1 for i in items if i.verified)
    rejected = sum(1 for i in items if i.notes.startswith("REJECTED"))
    return Progress(
        total=len(items),
        verified=verified,
        rejected=rejected,
        remaining=len(items) - verified - rejected,
    )


def _highlight(passage_text: str, answer: str, width: int = 100) -> list[str]:
    """Print the passage with the proposed answer marked, wrapped.

    If the answer is not found verbatim, that is said explicitly -- it usually
    means the model paraphrased, which silently depresses Exact Match later and
    is exactly what the annotator is here to catch.
    """
    import textwrap

    lines: list[str] = []
    idx = passage_text.find(answer) if answer else -1
    if answer and idx < 0:
        lines.append("  !! proposed answer does not appear verbatim in the passage")
    shown = passage_text
    if idx >= 0:
        shown = passage_text[:idx] + ">>>" + answer + "<<<" + passage_text[idx + len(answer) :]
    for line in textwrap.wrap(shown, width=width)[:14]:
        lines.append("    " + line)
    return lines


def render_assist(record: dict | None) -> list[str]:
    """The pre-review's notes for one item, from `dataset review`.

    Shown under the evidence, never above it: the annotator should read the
    passage first and the advice second, or the advice becomes the judgement.
    """
    if not record:
        return []
    out: list[str] = []
    for f in record.get("rules", []):
        out.append(f"  rule   {f['code']}: {f['detail']}")
    note = record.get("review")
    if note:
        who = record.get("reviewed_by", "reviewer")
        issues = ", ".join(note.get("issues", []))
        out.append(f"  review [{who}] {note['verdict'].upper()}" + (f"  ({issues})" if issues else ""))
        for key, label in (
            ("comment", "why"),
            ("suggested_question", "Q?"),
            ("suggested_answer", "A?"),
            ("suggested_answer_hi", "A(hi)?"),
            ("suggested_class", "class?"),
            ("evidence_quote", "quote"),
        ):
            if note.get(key):
                out.append(f"         {label:<7}{note[key]}")
    return ["", "  -- pre-review (advice only) --", *out] if out else []


def take_suggestion(item: QAItem, record: dict | None) -> list[str]:
    """Apply a pre-review's suggested question and answers to `item` in place.

    Returns the fields changed. `verified` is never touched here.
    """
    note = (record or {}).get("review") or {}
    changed = []
    for key, field_name in (
        ("suggested_question", "question"),
        ("suggested_answer", "answer_gold"),
        ("suggested_answer_hi", "answer_gold_hi"),
    ):
        value = (note.get(key) or "").strip()
        if value and value != getattr(item, field_name):
            setattr(item, field_name, value)
            changed.append(field_name)
    return changed


def render_item(
    item: QAItem,
    passages: dict[str, Passage],
    *,
    index: int,
    total: int,
    hide_label: bool = False,
    assist: dict | None = None,
) -> list[str]:
    out = [
        "",
        "=" * 100,
        f"[{index}/{total}]  {item.id}   {item.query_lang} -> {item.passage_lang}"
        + ("" if hide_label else f"   ({'ANSWERABLE' if item.answerable else 'UNANSWERABLE'})"),
        "-" * 100,
        f"  Q: {item.question}",
    ]
    if not hide_label:
        out.append(f"  A: {item.answer_gold or '(none proposed)'}")
    out.append("")

    for pid in item.gold_passage_ids:
        p = passages.get(pid)
        if p is None:
            out.append(f"  !! gold passage {pid} not found in corpus")
            continue
        out.append(f"  evidence [{pid}]  {p.scheme} / {p.lang} / {p.section_path or 'lead'}")
        out += _highlight(p.text, item.answer_gold if not hide_label else "")
    if not hide_label:
        # Hidden in the second pass for the same reason the label is: kappa
        # must measure the annotators' agreement, not the pre-review's.
        out += render_assist(assist)
    return out


def verify_loop(
    path: Path,
    passages: Sequence[Passage],
    *,
    annotator: str,
    ask: Callable[[str], str],
    say: Callable[[str], None],
    limit: int | None = None,
    assist: dict[str, dict] | None = None,
    recheck: bool = False,
    split: str | None = None,
) -> Progress:
    """Interactive review. Writes after every decision so nothing is ever lost.

    `recheck` queues items a model verified, so a person can review them: each
    item they accept is recorded under their name, and each one they reject
    stops counting as verified. Items they skip keep the model's annotator, so
    the report banners always show the true mix of who checked what.
    """
    items = list(read_jsonl(path, QAItem))
    by_id = {p.passage_id: p for p in passages}
    if recheck:
        pending = [
            i for i in items
            if i.verified and i.annotator.startswith("model:") and not i.notes.startswith("REJECTED")
        ]
    else:
        pending = [i for i in items if not i.verified and not i.notes.startswith("REJECTED")]
    if split:
        pending = [i for i in pending if i.split == split]
    if limit:
        pending = pending[:limit]

    say(str(progress_of(items)))
    say(ACTIONS)

    for n, item in enumerate(pending, start=1):
        while True:
            for line in render_item(
                item, by_id, index=n, total=len(pending), assist=(assist or {}).get(item.id)
            ):
                say(line)
            choice = (ask("  action> ") or "").strip().lower()

            if choice == "a":
                item.verified = True
                item.annotator = annotator
                # Appended, not replaced: an item's notes are its history, and a
                # re-review must not erase the passes that came before it.
                stamp = f"verified {date.today().isoformat()} by {annotator}"
                item.notes = f"{item.notes}; {stamp}" if item.notes else stamp
                break
            if choice == "t":
                # Copies the suggestion in and shows the item again. It never
                # accepts: the annotator still reads the edited item against
                # the evidence and presses [a] or not.
                taken = take_suggestion(item, (assist or {}).get(item.id))
                say(f"  took: {', '.join(taken)}" if taken else "  no suggestion to take")
                continue
            if choice == "e":
                item.answer_gold = ask("  corrected answer> ").strip()
                continue
            if choice == "q":
                item.question = ask("  corrected question> ").strip()
                continue
            if choice == "u":
                item.answerable = False
                item.unanswerable_class = ask(
                    "  class [out-of-scope|near-miss|false-premise|under-specified]> "
                ).strip()
                item.answer_gold = ""
                item.gold_passage_ids = []
                item.verified = True
                item.annotator = annotator
                break
            if choice == "r":
                item.notes = "REJECTED " + (ask("  reason> ").strip() or "unusable")
                item.verified = False
                break
            if choice == "p":
                for pid in item.gold_passage_ids:
                    p = by_id.get(pid)
                    if p:
                        say(p.text)
                continue
            if choice == "s":
                break
            if choice == "w":
                write_jsonl(path, items)
                return progress_of(items)
            say("  unrecognised; choose one of a/e/q/u/r/s/p/w")

        write_jsonl(path, items)  # after every decision, not at the end

    return progress_of(items)


# The re-labelling sample for Cohen's kappa is drawn by
# `dataset.second_pass.draw_sample`, which stratifies over the four unanswerable
# classes. A flat 15% draw used to live here and was superseded: with
# false-premise at 15 items of 400, an unstratified sample routinely contains
# two or three of them, and kappa computed without the hard classes measures the
# easy boundary. Nothing in the CLI ever called the flat version -- only its own
# test did -- so it is gone rather than left as a second way to do this wrongly.


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary labellings of the same items.

    PRD §6.5 sets kappa >= 0.70 as the gate: below that, the answerable boundary
    is not well enough defined for any Module 5 number to mean anything, and the
    guidelines need tightening before the main results are trusted.
    """
    if not a or len(a) != len(b):
        return 0.0
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa_true, pb_true = sum(a) / n, sum(b) / n
    expected = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if abs(1 - expected) < 1e-12:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
