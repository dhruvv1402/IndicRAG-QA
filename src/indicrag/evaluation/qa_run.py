"""Run the Module 4 sweep and score it.

Ties together the four arms, the QA scorer and the grounding metrics, so the
whole of Module 4 is one call. Arms that cannot run in the current environment
are skipped with a note rather than failing the sweep -- on a CPU-only machine a
harness that refuses to report anything until every arm is available is a harness
nobody runs.

`D − C` is retrieval error and `1 − D` is generation error, and `format_module4`
prints that decomposition explicitly rather than leaving the reader to subtract.
It is the thing the brief actually asks for, and it is easy to omit by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..models import Passage, QAItem
from ..query.langid import classify
from ..rag.arms import ARMS, Arm, Generation, GenerationCache, run_arm
from .grounding import GroundingReport, format_grounding, score_generation
from .qa import QAOutcome, QAReport

ARM_ORDER = ["A", "B", "C", "D"]


@dataclass
class Module4Result:
    qa: dict[str, QAReport]
    grounding: GroundingReport
    skipped: list[str]

    def arms(self) -> list[str]:
        return [a for a in ARM_ORDER if a in self.qa]


def _golds(item: QAItem) -> list[str]:
    return [g for g in (item.answer_gold, item.answer_gold_hi) if g]


def score_arm(
    arm: Arm,
    generations: Sequence[Generation],
    items: Sequence[QAItem],
    passages: Sequence[Passage],
) -> tuple[QAReport, list]:
    """Score one arm's generations for answer quality and grounding."""
    by_item = {i.id: i for i in items}
    by_pid = {p.passage_id: p for p in passages}

    qa_outcomes: list[QAOutcome] = []
    grounding_outcomes = []

    for gen in generations:
        item = by_item.get(gen.item_id)
        if item is None:
            continue
        abstained = not gen.answerable

        qa_outcomes.append(
            QAOutcome(
                item_id=item.id,
                slice_key=item.slice_key,
                language_group=item.language_group,
                prediction="" if abstained else gen.answer,
                golds=_golds(item),
                abstained=abstained,
            )
        )

        # Grounding is scored against the evidence the system actually cited when
        # it named one, falling back to the context it was given. For arm A there
        # is no context, so the gold passage is used -- the question there is
        # whether an unevidenced answer happens to match the truth, which is
        # exactly the hallucination baseline H3 needs.
        cited = gen.citation.strip()
        evidence_ids = [cited] if cited in by_pid else (gen.context_ids or item.gold_passage_ids)
        evidence = " ".join(by_pid[pid].text for pid in evidence_ids if pid in by_pid)

        grounding_outcomes.append(
            score_generation(
                item_id=item.id,
                arm=arm.key,
                answer=gen.answer,
                abstained=abstained,
                evidence=evidence,
                query_type=classify(item.question).query_type,
                cited_found=bool(cited and cited in by_pid),
            )
        )

    return QAReport(system=f"{arm.key} ({arm.label})", outcomes=qa_outcomes), grounding_outcomes


def run_module4(
    items: Sequence[QAItem],
    passages: Sequence[Passage],
    complete: Callable[[str], str],
    *,
    retrievers: dict[str, Callable[[QAItem], list[str]]] | None = None,
    cache: GenerationCache | None = None,
    model: str = "",
    k: int = 5,
    arms: Sequence[str] = tuple(ARM_ORDER),
    progress: Callable[[str], None] | None = None,
) -> Module4Result:
    """Run every requested arm and score it.

    `retrievers` maps an arm key to a retrieval function. An arm needing one that
    is not supplied is skipped with a note, not an exception.
    """
    say = progress or (lambda _m: None)
    retrievers = retrievers or {}
    answerable = [i for i in items if i.answerable]

    qa: dict[str, QAReport] = {}
    grounding = GroundingReport(system=model or "system")
    skipped: list[str] = []

    for key in arms:
        arm = ARMS[key]
        if arm.uses_retrieval and key not in retrievers:
            skipped.append(f"arm {key} ({arm.label}): no retriever supplied")
            continue
        say(f"  running arm {key} ({arm.label}) over {len(answerable)} items")
        gens = run_arm(
            arm,
            answerable,
            passages,
            complete,
            retrieve=retrievers.get(key),
            cache=cache,
            model=model,
            k=k,
            progress=say,
        )
        report, ground = score_arm(arm, gens, answerable, passages)
        qa[key] = report
        grounding.outcomes.extend(ground)

    return Module4Result(qa=qa, grounding=grounding, skipped=skipped)


def format_module4(result: Module4Result) -> list[str]:
    rule = "-" * 78
    out = [
        "MODULE 4 -- direct LLM vs retrieval-augmented question answering",
        rule,
        f"  {'arm':<18}{'n':>6}{'EM':>9}{'token-F1':>11}{'abstain':>10}{'CSR':>9}",
        rule,
    ]
    for key in result.arms():
        report = result.qa[key]
        out.append(
            f"  {ARMS[key].label:<18}{report.n():>6}{report.exact_match():>9.3f}"
            f"{report.token_f1():>11.3f}{report.abstention_rate():>10.3f}"
            f"{result.grounding.citation_support_rate(key):>9.3f}"
        )

    if "C" in result.qa and "D" in result.qa:
        c, d = result.qa["C"].token_f1(), result.qa["D"].token_f1()
        out += [
            "",
            "ERROR DECOMPOSITION",
            rule,
            f"  oracle (D) token-F1        {d:.3f}",
            f"  full system (C) token-F1   {c:.3f}",
            f"  retrieval error  (D - C)   {d - c:.3f}",
            f"  generation error (1 - D)   {1 - d:.3f}",
            "",
            "  This is what the brief asks for: a wrong answer attributable to the",
            "  component that caused it. D is generation given perfect evidence, so",
            "  what D still gets wrong is the generator, and the gap to C is what",
            "  retrieval lost.",
        ]

    if "A" in result.qa:
        out += [
            "",
            "H3 -- does retrieval reduce unsupported answers?",
            rule,
        ]
        base = result.grounding.citation_support_rate("A")
        for key in result.arms():
            if key == "A":
                continue
            rate = result.grounding.citation_support_rate(key)
            out.append(f"  {ARMS[key].label:<18} CSR {rate:.3f}   vs closed-book {base:.3f}   {rate - base:+.3f}")

    out += ["", *format_grounding(result.grounding, result.arms())]

    if result.skipped:
        out += ["", "NOT RUN", rule] + [f"  {s}" for s in result.skipped]
    return out
