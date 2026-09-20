"""Semantic checks on generated candidates.

Every case here is a real failure observed from Qwen2.5-0.5B, not a hypothetical.
They share one property that makes them dangerous: the JSON is well formed and
the `question` field is non-empty, so a parse-rate metric scores them as
successes while the content is invented. A validity check is not a correctness
check, and these tests pin the difference.
"""

from __future__ import annotations

from indicrag.dataset.generate import Candidate
from indicrag.dataset.validate import check_base, check_romanization, check_translation

# --- translation -----------------------------------------------------------------


def test_translation_that_returns_the_input_unchanged_is_rejected():
    """Observed: asked for Hindi, the model echoed the English back.

    Left in, this makes a "cross-lingual" item whose question is in the same
    language as its evidence, silently turning the cell monolingual and inflating
    cross-lingual recall.
    """
    q = "What is the income limit for this scheme?"
    assert not check_translation(q, q, to="hi")


def test_translation_to_hindi_must_actually_contain_devanagari():
    assert not check_translation("What is the limit?", "What is the boundary?", to="hi")
    assert check_translation("What is the limit?", "सीमा क्या है?", to="hi")


def test_translation_to_english_must_not_be_devanagari():
    assert not check_translation("सीमा क्या है?", "सीमा कितनी है?", to="en")
    assert check_translation("सीमा क्या है?", "What is the limit?", to="en")


def test_empty_translation_is_rejected():
    assert not check_translation("What is the limit?", "", to="hi")


# --- romanization ----------------------------------------------------------------


def test_romanization_that_translated_instead_is_rejected():
    """Observed: 'मध्याह्न भोजन योजना कब शुरू हुई?' came back as
    'What was the date when the college started?' -- fluent English, no code-mix,
    and a different question. Labelling that code-mixed would corrupt one of the
    three slices the project compares."""
    verdict = check_romanization(
        "मध्याह्न भोजन योजना कब शुरू हुई?", "What was the date when the college started?"
    )
    assert not verdict
    assert "translated rather than romanized" in verdict.reason


def test_romanization_still_in_devanagari_is_rejected():
    assert not check_romanization("आय सीमा क्या है?", "आय सीमा कितनी है?")


def test_genuine_hinglish_passes():
    assert check_romanization("आय सीमा क्या है?", "Income limit kya hai?")
    assert check_romanization("कब शुरू हुई?", "Scheme kab shuru hui thi?")


# --- base generation -------------------------------------------------------------


def _c(q, a="1995"):
    return Candidate(question=q, answer=a, kind="fact")


def test_echoed_prompt_scaffolding_is_rejected():
    """Observed: the 0.5B returned the literal word नियम ("Rules") from the
    prompt's own section header as the question."""
    verdict = check_base(_c("नियम: अनुच्छेद के शब्द दोहराएँ", "कुछ"), "कुछ पाठ", "hi")
    assert not verdict
    assert "scaffolding" in verdict.reason


def test_question_must_match_the_passage_language():
    assert not check_base(_c("When did it start?"), "योजना 1995 में शुरू हुई", "hi")
    assert not check_base(_c("योजना कब शुरू हुई?"), "The scheme started in 1995", "en")


def test_an_answer_sharing_no_token_with_the_passage_is_rejected():
    """A wholly invented answer has no lexical footing in its own evidence."""
    verdict = check_base(_c("When did the scheme start?", "zzzz qqqq"), "started in 1995", "en")
    assert not verdict
    assert "shares no token" in verdict.reason


def test_empty_answer_is_rejected():
    assert not check_base(_c("When did the scheme start?", ""), "started in 1995", "en")


def test_very_short_question_is_rejected():
    assert not check_base(_c("When?"), "started in 1995", "en")


def test_a_good_candidate_passes_in_both_languages():
    assert check_base(_c("When did the scheme start?", "1995"), "The scheme started in 1995", "en")
    assert check_base(
        _c("योजना कब शुरू हुई?", "1995 में"), "मध्याह्न भोजन योजना 1995 में शुरू हुई", "hi"
    )
