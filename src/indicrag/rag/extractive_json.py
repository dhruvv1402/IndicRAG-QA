"""Adapter making the extractive provider answer through the same interface as a model.

The arms in `rag/arms.py` take a single `complete(prompt) -> json_string`
callable so that any backend can be dropped in. The extractive provider has a
richer signature -- it wants the question and the passages as objects -- so this
bridges the two by reading them back out of the prompt.

Parsing a prompt we constructed ourselves is a little circular, but it is honest
here: `rag/prompts.py` owns the format and this module owns the inverse, so the
two are changed together and a mismatch fails loudly in tests rather than
producing a subtly wrong baseline.

Having the extractive path answer through the real arm machinery is what makes
`--no-model` a genuine evaluation rather than a stub. It produces a full Module 4
table with no inference, which is what keeps the retrieval and answerability
halves of the project testable on a machine with no spare CPU, and it gives the
generative arms a floor to be read against: every extractive answer is a verbatim
span of its evidence, so its Citation Support Rate is 1.0 by construction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from ..models import Passage
from ..query.langid import classify
from .prompts import REFUSAL

_QUESTION_RE = re.compile(r"^Question:\s*(.+)$", re.M)
_PASSAGE_ID_RE = re.compile(r"^\[\d+\]\s+id=(\S+)", re.M)


def parse_prompt(prompt: str) -> tuple[str, list[str]]:
    """Recover (question, passage_ids) from a prompt built by `rag/prompts.py`."""
    match = _QUESTION_RE.search(prompt)
    question = match.group(1).strip() if match else ""
    return question, _PASSAGE_ID_RE.findall(prompt)


def extractive_complete(passages: Sequence[Passage]) -> Callable[[str], str]:
    """Build a `complete`-compatible callable backed by sentence extraction."""
    from .providers import ExtractiveProvider

    by_id = {p.passage_id: p for p in passages}
    provider = ExtractiveProvider()

    def complete(prompt: str) -> str:
        question, ids = parse_prompt(prompt)
        context = [by_id[pid] for pid in ids if pid in by_id]

        # No context means the closed-book arm. An extractive system has no
        # parametric knowledge to draw on, so it abstains -- which is the correct
        # and informative answer here rather than a limitation: it establishes
        # that any closed-book score belongs to the model, not the scaffolding.
        if not context:
            return json.dumps(
                {"answerable": False, "answer": REFUSAL, "citation": "", "confidence": 0.0},
                ensure_ascii=False,
            )

        generated = provider.answer(question, context, lang=classify(question))
        citation = (generated.citations or [""])[0]
        return json.dumps(
            {
                "answerable": bool(generated.text.strip()),
                "answer": generated.text,
                "citation": citation,
                "confidence": round(float(generated.confidence), 3),
            },
            ensure_ascii=False,
        )

    return complete
