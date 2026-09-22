"""Query language identification: English / Indic / Code-Mixed.

A misclassification here is not cosmetic. The detected type selects the language
the answer is generated in, and it is the grouping key for every per-language
table in the evaluation -- so a systematic error would both answer users in the
wrong language and silently move rows between slices in the results.

The regression guarded by `test_common_english_words_are_not_hindi_evidence` is
real and was caught in development: "the", "is" and "to" are valid Romanizations
of थे / इस / तो, and counting them as Hindi evidence classified plain English
questions as Code-Mixed.
"""

from __future__ import annotations

import pytest

from indicrag.query.langid import AMBIGUOUS_MARKERS, UNAMBIGUOUS_MARKERS, classify, slice_key

ENGLISH = [
    "What is the minimum eligibility for the scholarship?",
    "List the required documents.",
    "Who can apply and how do I submit the form to us?",
    "Is the income limit the same for all states?",
    "What is the last date to apply?",
    "How much money is given per year under this scheme?",
]

CODE_MIXED = [
    "Scholarship ke liye minimum eligibility kya hai?",
    "NMMS scholarship ki income limit kitni hai?",
    "PM-YASASVI के लिए eligibility criteria kya hai?",
    "Renewal ke liye kaun se documents chahiye?",
    "Mujhe kitna paisa milega?",
    "Application ka last date kab hai?",
]

INDIC = [
    "छात्रवृत्ति के लिए न्यूनतम पात्रता क्या है?",
    "आय सीमा कितनी है?",
    "राष्ट्रीय छात्रवृत्ति पोर्टल पर आवेदन कैसे करें?",
    "इस योजना के अंतर्गत कितनी राशि दी जाती है?",
]


@pytest.mark.parametrize("q", ENGLISH)
def test_plain_english_questions_classify_as_english(q):
    assert classify(q).query_type == "English"


@pytest.mark.parametrize("q", CODE_MIXED)
def test_romanized_and_mixed_script_questions_classify_as_code_mixed(q):
    assert classify(q).query_type == "Code-Mixed"


@pytest.mark.parametrize("q", INDIC)
def test_devanagari_questions_classify_as_indic(q):
    assert classify(q).query_type == "Indic"


def test_the_briefs_own_example_is_code_mixed():
    """PJ_09.pdf §7 uses this query verbatim; it is the canonical case."""
    result = classify("Scholarship ke liye minimum eligibility kya hai?")
    assert result.query_type == "Code-Mixed"
    assert result.lang == "hi-en"
    assert result.script == "latin"


def test_common_english_words_are_not_hindi_evidence():
    """"the", "is" and "to" are Romanized Hindi *and* the commonest English words.

    They must never, on their own, push a query into Code-Mixed.
    """
    result = classify("Is the applicant required to submit the income certificate to us?")
    assert result.query_type == "English"
    assert result.hindi_marker_hits == 0
    assert result.ambiguous_marker_hits > 0, "the ambiguous markers should still be counted"


def test_ambiguous_and_unambiguous_marker_sets_are_disjoint():
    assert not (UNAMBIGUOUS_MARKERS & AMBIGUOUS_MARKERS)


def test_a_single_devanagari_word_among_english_is_code_mixed():
    """8% Devanagari by character count would slip under a ratio floor."""
    result = classify("PM-YASASVI के लिए eligibility criteria kya hai?")
    assert result.query_type == "Code-Mixed"
    assert result.script == "mixed"


def test_empty_query_does_not_raise():
    assert classify("").query_type == "English"
    assert classify("   ").query_type == "English"


def test_slice_key_matches_the_labels_used_in_the_results_tables():
    """docs/PLAN.md §3.2 names these columns; the spelling must agree."""
    assert slice_key(classify("What is the income limit?"), "en") == "EN->EN"
    assert slice_key(classify("आय सीमा कितनी है?"), "en") == "HI->EN"
    assert slice_key(classify("Income limit kitni hai?"), "hi") == "Hing->HI"
    assert slice_key(classify("What is the income limit?")) == "EN"


def test_the_indic_floor_and_latin_ceiling_match_the_paper():
    """§IV-C quotes 0.60 as the Devanagari ratio above which a query is Indic.
    A threshold stated in prose and set in code is two places to disagree."""
    from indicrag.query.langid import DEVANAGARI_INDIC_FLOOR, DEVANAGARI_LATIN_CEILING

    assert DEVANAGARI_INDIC_FLOOR == 0.60
    assert DEVANAGARI_LATIN_CEILING == 0.10
