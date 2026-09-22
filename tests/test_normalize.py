"""Normalization must be idempotent and identical on both sides of retrieval.

If passages are normalized one way at index time and queries another at search
time, they stop matching on exactly the cases that matter -- amounts, scheme
names, dates -- and the failure presents as poor model quality rather than as a
preprocessing bug. These tests pin that invariant.
"""

from __future__ import annotations

import pytest

from indicrag.corpus.normalize import (
    canonical_amounts,
    normalize_digits,
    normalize_text,
    normalize_unicode,
    split_sentences,
)


def test_normalization_is_idempotent():
    """Running it twice must equal running it once, for every script."""
    for raw in [
        "Rs.  3,50,000   per annum",
        "छात्रवृत्ति  के  लिए‍  आवेदन",
        "PM-YASASVI – eligibility ‘criteria’",
    ]:
        once = normalize_text(raw)
        assert normalize_text(once) == once


def test_devanagari_digits_become_ascii():
    assert normalize_digits("३,५०,०००") == "3,50,000"
    assert normalize_digits("कक्षा ९") == "कक्षा 9"


def test_zero_width_characters_are_stripped():
    """Their inconsistent presence in extracted PDF text fragments tokens."""
    assert normalize_unicode("के‍लिए") == "केलिए"
    assert "﻿" not in normalize_unicode("﻿छात्रवृत्ति")


def test_nfc_composition_makes_equivalent_forms_compare_equal():
    decomposed = "क़"  # ka + combining nukta
    composed = "क़"  # qa
    assert normalize_unicode(decomposed) == normalize_unicode(composed)


@pytest.mark.parametrize(
    "surface",
    [
        "Rs. 3,50,000 per annum",
        "Rs 3,50,000",
        "₹3,50,000/-",
        "INR 3,50,000",
        "3.5 lakh",
        "आय ३,५०,००० रुपये",
    ],
)
def test_every_surface_form_of_the_same_amount_canonicalises_alike(surface):
    """Amounts are the commonest answer type in this corpus and the most varied."""
    assert "350000" in canonical_amounts(surface)


def test_sentences_split_on_danda_not_only_on_full_stop():
    """A Latin-only splitter returns an entire Hindi paragraph as one sentence."""
    text = "यह योजना केंद्र सरकार द्वारा चलाई जाती है। आवेदन ऑनलाइन किया जाता है। राशि प्रति वर्ष दी जाती है॥"
    assert len(split_sentences(text)) == 3


def test_normalization_preserves_the_terminators_the_splitter_needs():
    """Segmentation normalizes a section *before* splitting it into sentences.

    So the splitter above only works if normalization leaves danda and double
    danda alone -- punctuation folding is the stage with the opportunity to
    rewrite them, and a Hindi paragraph whose terminators were folded away comes
    back as one 400-token sentence that no packer can place. Verified at corpus
    scale too: 1095 dandas across 40 Hindi documents, none lost.
    """
    text = "यह योजना केंद्र सरकार द्वारा चलाई जाती है। आवेदन ऑनलाइन किया जाता है॥"
    normalized = normalize_text(text)
    assert normalized.count("।") == text.count("।") == 1
    assert normalized.count("॥") == text.count("॥") == 1
    assert len(split_sentences(normalized)) == len(split_sentences(text)) == 2


def test_english_and_hindi_sentences_split_in_one_pass():
    mixed = "The scheme is central. यह योजना केंद्रीय है। Apply online."
    assert len(split_sentences(mixed)) == 3


def test_pdf_soft_wrap_inside_a_sentence_is_repaired():
    """A line break mid-sentence is a layout artefact, not a paragraph break."""
    wrapped = "students whose parental\nincome does not exceed"
    assert "parental income" in normalize_text(wrapped)


def test_paragraph_breaks_survive_normalization():
    assert "\n\n" in normalize_text("First clause.\n\n\n\nSecond clause.")


def test_empty_input_is_safe():
    assert normalize_text("") == ""
    assert canonical_amounts("") == []
    assert split_sentences("") == []


def test_an_abbreviation_stop_does_not_end_a_sentence():
    """Regression. "Rs." precedes an amount in most answers in this corpus, so
    splitting after it truncated "Rs. 12,000 per annum" to "12,000 per annum",
    which then scores as a miss on Exact Match against the document's wording.
    Found by reading an extracted answer, not from a failing test."""
    assert split_sentences("The scheme provides Rs. 12,000 per annum. Students are eligible.") == [
        "The scheme provides Rs. 12,000 per annum.",
        "Students are eligible.",
    ]


def test_other_common_abbreviations_are_protected():
    assert len(split_sentences("See Sec. 4 and Art. 21. Both apply here.")) == 2
    assert len(split_sentences("Submit Form No. 5 to the office.")) == 1
    assert len(split_sentences("Income under Rs. 3,50,000 per annum qualifies.")) == 1


def test_abbreviation_protection_leaves_real_boundaries_alone():
    assert len(split_sentences("Apply online. Then submit the form.")) == 2
    assert len(split_sentences("यह योजना 1995 में शुरू हुई। राशि दी जाती है।")) == 2


def test_the_guard_character_never_survives_into_output():
    """The splitter protects abbreviation stops with a NUL placeholder; it must
    never leak into a passage or an answer."""
    from indicrag.corpus.normalize import _ABBREV_GUARD

    for text in ["Rs. 12,000 per annum.", "See Sec. 4. Done.", "Plain text."]:
        assert all(_ABBREV_GUARD not in part for part in split_sentences(text))
