"""Bootstrap question-answer candidates against the PRD's language-pair matrix.

This produces **candidates**, never gold. Every item comes out `verified: false`,
and `docs/PRD.md` §6.5 step 3 requires a human to confirm each one, correct the
answer to the document's exact wording, and record the gold passage before it
counts. `indicrag dataset verify` is that loop. Nothing here can be reported.

How each cell of the §6.2 matrix is built:

| cell        | passage | question generated in | then                    |
|-------------|---------|-----------------------|-------------------------|
| EN->EN      | English | English               | --                      |
| HI->HI      | Hindi   | Hindi                 | --                      |
| EN->HI      | Hindi   | Hindi                 | translated to English   |
| HI->EN      | English | English               | translated to Hindi     |
| Hing->EN    | English | English               | -> Hindi -> Romanized   |
| Hing->HI    | Hindi   | Hindi                 | -> Romanized            |

The passage never moves. Only the question is transformed, which is what makes a
cross-lingual item genuinely cross-lingual: the evidence exists only in the other
language, so a retriever has to cross the boundary rather than find a
same-language paraphrase. Translating the passage as well would quietly turn a
cross-lingual item into a monolingual one and inflate every cross-lingual number.

Passage selection is stratified over schemes so that no single long article
dominates a cell, and prefers passages that contain a digit, because a passage
with a number in it yields a checkable answer and a passage of narrative history
usually does not.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..models import Passage, QAItem
from .prompts import (
    QA_GRAMMAR,
    build_hinglish_prompt,
    build_prompt,
    build_translate_prompt,
)

#: PRD §6.2. (query_lang, passage_lang) -> count. Sums to 320.
MATRIX: dict[tuple[str, str], int] = {
    ("en", "en"): 70,
    ("hi", "hi"): 60,
    ("en", "hi"): 45,
    ("hi", "en"): 45,
    ("hinglish", "en"): 55,
    ("hinglish", "hi"): 45,
}

MIN_TOKENS = 70
_DIGIT_RE = re.compile(r"\d")


@dataclass
class Candidate:
    question: str
    answer: str
    kind: str


def parse_reply(raw: str) -> Candidate | None:
    """Parse the model's JSON. Returns None for the deliberate empty reply.

    The grammar makes malformed JSON unreachable, but the model is explicitly
    allowed to answer with empty strings when a passage states no fact worth
    asking about, and that is a legitimate outcome rather than a failure.
    """
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    if not question:
        return None
    return Candidate(question=question, answer=answer, kind=data.get("kind") or "fact")


def select_passages(
    passages: Sequence[Passage], lang: str, n: int, *, rng: random.Random
) -> list[Passage]:
    """Pick `n` passages of `lang`, spread across schemes, preferring factual ones.

    Round-robin over schemes rather than a flat sample: MGNREGA contributes many
    more passages than Sansad Adarsh Gram Yojana, and a flat sample would fill a
    whole matrix cell from two or three long articles.
    """
    pool = [p for p in passages if p.lang == lang and p.token_count >= MIN_TOKENS]
    by_scheme: dict[str, list[Passage]] = defaultdict(list)
    for p in pool:
        by_scheme[p.scheme].append(p)

    for group in by_scheme.values():
        # Passages carrying a digit first: they yield checkable answers.
        rng.shuffle(group)
        group.sort(key=lambda p: not _DIGIT_RE.search(p.text))

    schemes = sorted(by_scheme)
    rng.shuffle(schemes)

    picked: list[Passage] = []
    cursor = 0
    while len(picked) < n and any(by_scheme[s] for s in schemes):
        scheme = schemes[cursor % len(schemes)]
        if by_scheme[scheme]:
            picked.append(by_scheme[scheme].pop(0))
        cursor += 1
    return picked[:n]


def generate_candidates(
    passages: Sequence[Passage],
    complete: Callable[[str, str | None], str],
    *,
    seed: int = 20260922,
    matrix: dict[tuple[str, str], int] | None = None,
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[list[QAItem]], None] | None = None,
) -> list[QAItem]:
    """Generate one candidate per matrix slot.

    `complete(prompt, grammar) -> str` is the only model dependency, so this is
    testable with a scripted stub and works against any provider.

    `checkpoint` is called with the items so far after each one. A full run takes
    hours of CPU inference, and losing it to a crash on item 290 would mean
    redoing every earlier call -- the results are deterministic but not free.
    """
    rng = random.Random(seed)
    matrix = matrix or MATRIX
    say = progress or (lambda _m: None)
    items: list[QAItem] = []
    used: set[str] = set()

    for (query_lang, passage_lang), count in matrix.items():
        # Over-select: some passages yield an empty reply, and a cell short of
        # its target skews the matrix the evaluation is grouped by.
        pool = select_passages(passages, passage_lang, count * 3, rng=rng)
        pool = [p for p in pool if p.passage_id not in used]
        made = 0

        for passage in pool:
            if made >= count:
                break

            base = parse_reply(
                complete(
                    build_prompt(
                        lang=passage.lang,
                        scheme=passage.scheme.replace("-", " "),
                        section=passage.section_path,
                        passage=passage.text[:1100],
                    ),
                    QA_GRAMMAR,
                )
            )
            if base is None or len(base.question) < 8:
                continue

            question = base.question
            # Transform the question only. The passage stays put.
            if query_lang != passage_lang and query_lang != "hinglish":
                translated = parse_reply(
                    complete(build_translate_prompt(question, to=query_lang), QA_GRAMMAR)
                )
                if translated is None:
                    continue
                question = translated.question
            elif query_lang == "hinglish":
                hindi = question
                if passage_lang == "en":
                    step = parse_reply(
                        complete(build_translate_prompt(question, to="hi"), QA_GRAMMAR)
                    )
                    if step is None:
                        continue
                    hindi = step.question
                romanised = parse_reply(complete(build_hinglish_prompt(hindi), QA_GRAMMAR))
                if romanised is None:
                    continue
                question = romanised.question

            if len(question) < 8:
                continue

            items.append(
                QAItem(
                    id=f"qa-{query_lang}-{passage_lang}-{made:03d}",
                    question=question,
                    query_lang=query_lang,
                    passage_lang=passage_lang,
                    answerable=True,
                    answer_gold=base.answer,
                    gold_passage_ids=[passage.passage_id],
                    scheme=passage.scheme,
                    difficulty_tags=[base.kind],
                    annotator="",
                    verified=False,
                    notes=f"bootstrapped; section={passage.section_path}",
                )
            )
            used.add(passage.passage_id)
            made += 1
            if checkpoint is not None:
                checkpoint(items)
            if made % 5 == 0:
                say(f"  {query_lang}->{passage_lang}: {made}/{count}")

        say(f"  {query_lang}->{passage_lang}: {made}/{count} generated")

    return items
