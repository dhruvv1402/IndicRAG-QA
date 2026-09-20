"""Grounded answering prompts.

Distinct from `dataset/prompts.py`, which builds the dataset. These are the
prompts the system uses to answer a user's question from retrieved evidence, and
each instruction in them answers a failure observed on small models during the
dataset work.

The instruction set is deliberately short. A 3B model given eight rules follows
about four of them; the ones kept here are the ones whose absence breaks a
measurement rather than merely degrading style.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import Passage

REFUSAL = "Insufficient information available in the provided documents."

#: The answer schema. `answerable` is the generator's self-report, which is one
#: of the four answerability signals in ARCHITECTURE §12 and the only one that
#: sees the passage text rather than a retrieval score.
ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string", "maxLength": 400},
        "citation": {"type": "string", "maxLength": 60},
        "confidence": {"type": "number"},
    },
    "required": ["answerable", "answer", "citation", "confidence"],
}

_RULES_EN = """Rules:
- Use ONLY the passages above. Never use outside knowledge.
- Answer in the SAME language and script as the question.
- Quote numbers, dates and names exactly as the passage writes them.
- Put the id of the passage you used in "citation".
- If the passages do not contain the answer, set "answerable" to false and leave
  "answer" empty. Do not guess."""

PROMPT_RAG = """You answer questions about Indian government schemes using only the passages provided.

{passages}

Question: {question}

{rules}

Reply with JSON only: {{"answerable": true, "answer": "...", "citation": "...", "confidence": 0.0}}"""

#: Arm A. No passages at all -- this is the closed-book control, and its whole
#: purpose is to measure what the model will assert with no evidence in front of
#: it. It is therefore NOT told to abstain when unsure, because an instruction to
#: abstain would suppress the very behaviour being measured.
PROMPT_CLOSED_BOOK = """Answer this question about Indian government schemes from your own knowledge.

Question: {question}

Answer in the same language and script as the question. Be specific: give numbers, dates and names where you can.

Reply with JSON only: {{"answerable": true, "answer": "...", "citation": "", "confidence": 0.0}}"""


def format_passages(passages: Sequence[Passage], *, max_chars: int = 900) -> str:
    """Number the passages and label each with its id and section.

    Best evidence first, matching retrieval order. Small models attend unevenly
    across a long context, so the ordering is not cosmetic -- burying the best
    passage at position five measurably costs answer quality.
    """
    lines: list[str] = []
    for i, p in enumerate(passages, start=1):
        where = p.section_path or "lead"
        body = p.text[:max_chars]
        lines.append(f"[{i}] id={p.passage_id} ({p.scheme}, {where})\n{body}")
    return "\n\n".join(lines)


def build_answer_prompt(question: str, passages: Sequence[Passage]) -> str:
    return PROMPT_RAG.format(
        passages=format_passages(passages), question=question, rules=_RULES_EN
    )


def build_closed_book_prompt(question: str) -> str:
    return PROMPT_CLOSED_BOOK.format(question=question)
