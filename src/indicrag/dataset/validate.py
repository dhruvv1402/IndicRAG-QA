"""Semantic checks on generated candidates.

**A validity check is not a correctness check, and the gap between them is where
a dataset quietly rots.** Qwen2.5-0.5B, asked to translate
`इस योजना के लिए आय सीमा क्या है?` into English, returned
`{"question": "What is the salary for this job?"}`. Asked to romanize
`मध्याह्न भोजन योजना कब शुरू हुई?` it returned
`{"question": "What was the date when the college started?"}`. Both are
well-formed JSON with a non-empty `question`, so `parse_failure_rate` scored them
100% successful while the meaning was entirely invented.

That failure mode is worse than a crash. A malformed reply is discarded; a
plausible mistranslation is annotated, split, and reported.

The checks here are deliberately cheap and structural rather than semantic --
they cannot tell a good translation from a mediocre one. What they catch is the
model not doing the task at all: a "translation into Hindi" containing no
Devanagari, a "romanization" that is still in Devanagari or has lost every Hindi
function word, or a transform that returned its input unchanged. That is exactly
the class of silent failure above, and it is the class a small model produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..query.langid import UNAMBIGUOUS_MARKERS, devanagari_ratio

if TYPE_CHECKING:  # avoids a cycle: generate imports the checks below
    from .generate import Candidate


@dataclass
class Check:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


OK = Check(True)


def _tokens(text: str) -> set[str]:
    import re

    return {t.lower() for t in re.findall(r"[a-zA-Zऀ-ॿ]+", text)}


def check_translation(source: str, result: str, *, to: str) -> Check:
    """The output must actually be in the requested language, and must differ.

    Returning the input unchanged is the commonest small-model failure on a
    translation prompt, and it produces a "cross-lingual" item whose question is
    in the same language as its evidence -- silently converting the cell into a
    monolingual one and inflating cross-lingual recall.
    """
    if not result or len(result) < 6:
        return Check(False, "empty or too short")

    ratio = devanagari_ratio(result)
    if to == "hi" and ratio < 0.5:
        return Check(False, f"asked for Hindi, got devanagari ratio {ratio:.2f}")
    if to == "en" and ratio > 0.1:
        return Check(False, f"asked for English, got devanagari ratio {ratio:.2f}")

    if result.strip() == source.strip():
        return Check(False, "returned the input unchanged")
    return OK


def check_romanization(source_hi: str, result: str) -> Check:
    """A Hinglish question must be Latin script and still recognisably Hindi.

    Two failures to catch. Returning Devanagari means the romanization did not
    happen. Returning fluent English with no Hindi function words means the model
    translated instead of transliterating, which produces an item labelled
    code-mixed that contains no code-mixing -- and the code-mixed slice is one of
    the three the project exists to compare.
    """
    if not result or len(result) < 6:
        return Check(False, "empty or too short")

    ratio = devanagari_ratio(result)
    if ratio > 0.2:
        return Check(False, f"still in Devanagari (ratio {ratio:.2f})")

    markers = len(_tokens(result) & UNAMBIGUOUS_MARKERS)
    if markers == 0:
        return Check(False, "no Hindi function words: translated rather than romanized")
    return OK


def check_base(candidate: Candidate, passage_text: str, passage_lang: str) -> Check:
    """The generated question must be in the passage's language and be a question.

    The `नियम` failure is what this catches: Qwen2.5-0.5B echoed the literal word
    "Rules" from the prompt's own section header back as the question. Any check
    that only asked "is the field non-empty" would have accepted it.
    """
    q = candidate.question.strip()
    if len(q) < 10:
        return Check(False, "question too short")

    ratio = devanagari_ratio(q)
    if passage_lang == "hi" and ratio < 0.5:
        return Check(False, f"Hindi passage but question devanagari ratio {ratio:.2f}")
    if passage_lang == "en" and ratio > 0.1:
        return Check(False, f"English passage but question devanagari ratio {ratio:.2f}")

    # Prompt scaffolding echoed back rather than a question written.
    for leak in ("नियम", "Rules:", "अनुच्छेद के शब्द", "passage states", "{scheme}"):
        if leak in q:
            return Check(False, f"echoed prompt scaffolding ({leak!r})")

    if not candidate.answer.strip():
        return Check(False, "empty answer")

    # The answer should come from the passage. Require some lexical footing;
    # a wholly invented answer shares nothing with its own evidence.
    a_tokens = _tokens(candidate.answer)
    if a_tokens and not (a_tokens & _tokens(passage_text)):
        return Check(False, "answer shares no token with the passage")
    return OK
