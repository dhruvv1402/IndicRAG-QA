"""Guards the highest-risk failure in the pipeline: silently corrupted Hindi text.

A Devanagari extraction bug does not raise. It produces a string that renders as
plausible Hindi and is character-level nonsense, and it poisons the index and the
gold answers together. These tests pin the detector against the specific
corruption modes described in docs/ARCHITECTURE.md §3.2, using a real passage of
formal Hindi as the clean baseline.
"""

from __future__ import annotations

from indicrag.corpus import devanagari as dv

CLEAN_HINDI = (
    "छात्रवृत्ति के लिए आवेदन करने वाले छात्रों की पारिवारिक आय तीन लाख पचास हजार "
    "रुपये से अधिक नहीं होनी चाहिए। यह योजना केंद्र सरकार द्वारा संचालित की जाती है "
    "और इस के लिए आवेदन ऑनलाइन किया जाता है। पात्र छात्रों को प्रति वर्ष राशि दी जाती है।"
)

CLEAN_ENGLISH = (
    "Students whose parental income from all sources does not exceed Rs. 3,50,000 "
    "per annum are eligible to apply for the scholarship under this scheme."
)


def test_clean_hindi_passes_every_check():
    report = dv.validate(CLEAN_HINDI)
    assert report.ok, report.summary()
    assert report.devanagari_ratio > 0.9
    assert report.misplaced_matra_rate == 0.0
    assert report.stopword_rate > 0.1


def test_the_validator_rejects_reordered_matras():
    """The classic legacy-font bug: the vowel sign extracts before its consonant.

    This is the one that must never get through. It leaves the Devanagari ratio
    and the character count untouched, so ratio-based checks alone would pass it.
    """
    corrupted = CLEAN_HINDI.replace("के", "ेक").replace("की", "ीक")
    report = dv.validate(corrupted)
    assert not report.ok
    assert "misplaced_matras" in report.failures
    assert report.samples["misplaced_matras"], "a failure should carry example context"


def test_a_latin_string_fails_the_hindi_ratio_check():
    report = dv.validate(CLEAN_ENGLISH)
    assert not report.ok
    assert "low_devanagari_ratio" in report.failures


def test_english_passes_when_hindi_is_not_expected():
    """The same function must be usable on the English half of the corpus."""
    report = dv.validate(CLEAN_ENGLISH, expect_hindi=False)
    assert report.ok, report.summary()


def test_replacement_characters_are_always_a_failure():
    report = dv.validate(CLEAN_HINDI + "��", expect_hindi=True)
    assert not report.ok
    assert "replacement_chars" in report.failures


def test_dangling_viramas_are_detected():
    """A word-final bare virama means a conjunct lost its second half."""
    broken = " ".join(w + "्" for w in ["छात्र", "आय", "योजना", "राशि", "वर्ष"] * 8)
    report = dv.validate(broken)
    assert "dangling_viramas" in report.failures


def test_mojibake_inside_the_devanagari_block_fails_the_stopword_check():
    """Text can be 100% Devanagari codepoints and still not be Hindi."""
    junk = " ".join(["कखगघङ", "चछजझञ", "टठडढण", "तथदधन", "पफबभम"] * 12)
    report = dv.validate(junk)
    assert not report.ok
    assert "low_stopword_rate" in report.failures


def test_short_strings_are_rejected_rather_than_scored():
    assert "too_short" in dv.validate("छात्र").failures


def test_script_of_labels_all_three_cases():
    assert dv.script_of(CLEAN_HINDI) == "deva"
    assert dv.script_of(CLEAN_ENGLISH) == "latin"
    assert dv.script_of("scholarship के लिए eligibility क्या है") == "mixed"
