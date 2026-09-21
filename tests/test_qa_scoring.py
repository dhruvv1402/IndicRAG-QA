"""Answer scoring, and the normalizer underneath it.

A quiet bug here shifts every Exact Match and token-F1 number in the paper
without anything looking wrong, which is why these are pinned against
hand-computed expectations rather than against current behaviour.

`test_a_plain_digit_run_canonicalises_whole` guards a real bug: the amount regex
was written `\\d{1,3}(?:,\\d{2,3})*`, so a plain "350000" matched only its first
three digits and canonicalised to 350. Every comparison between a grouped and an
un-grouped amount was silently wrong, in both scoring and retrieval matching.
"""

from __future__ import annotations

import pytest

from indicrag.corpus.normalize import canonical_amounts
from indicrag.evaluation.qa import (
    QAOutcome,
    QAReport,
    exact_match,
    normalize_answer,
    token_f1,
)

# --- normalization --------------------------------------------------------------


def test_english_articles_and_case_are_stripped():
    assert normalize_answer("The Student") == normalize_answer("a student")


def test_danda_is_removed_where_ascii_punctuation_rules_would_miss_it():
    """The SQuAD normalizer drops ASCII punctuation and leaves danda behind."""
    assert normalize_answer("यह योजना है।") == "यह योजना है"
    assert normalize_answer("राशि दी जाती है॥") == "राशि दी जाती है"


def test_hindi_postposition_spacing_is_unified():
    """Extracted text writes these joined and separated interchangeably."""
    assert normalize_answer("के लिए आवेदन") == normalize_answer("केलिए आवेदन")


def test_devanagari_digits_normalise_to_ascii():
    assert normalize_answer("३,५०,०००").replace(" ", "") == "350000"


def test_normalization_is_idempotent():
    for s in ["The Rs. 3,50,000 per annum", "३,५०,००० रुपये", "यह योजना है।"]:
        assert normalize_answer(normalize_answer(s)) == normalize_answer(s)


# --- amounts ---------------------------------------------------------------------


def test_a_plain_digit_run_canonicalises_whole():
    """Regression: "350000" used to canonicalise to 350."""
    assert canonical_amounts("350000") == ["350000"]
    assert canonical_amounts("2500") == ["2500"]


@pytest.mark.parametrize(
    "a,b",
    [
        ("३,५०,०००", "350000"),
        ("350000", "3,50,000"),
        ("Rs. 3,50,000", "₹3.5 lakh"),
        ("2500", "Rs 2,500/-"),
        ("३,५०,०००", "₹3.5 lakh"),
        ("1,00,000", "1 lakh"),
    ],
)
def test_every_surface_form_of_an_amount_exact_matches(a, b):
    """Amounts are the commonest answer type, and their form differs between the
    English and Hindi versions of the same document."""
    assert exact_match(a, b)


def test_a_qualified_answer_does_not_match_the_bare_amount():
    """Collapsing these would score an incomplete answer as perfect."""
    assert not exact_match("income up to Rs 3,50,000 for rural applicants", "3.5 lakh")


def test_two_different_amounts_do_not_match():
    assert not exact_match("Rs 3,50,000", "Rs 2,50,000")


# --- token F1 --------------------------------------------------------------------


def test_f1_endpoints():
    assert token_f1("free hostel", "free hostel") == 1.0
    assert token_f1("abc", "xyz") == 0.0


def test_f1_uses_multisets_so_repetition_cannot_inflate_recall():
    """Set-based overlap would score this 1.0."""
    assert token_f1("hostel hostel hostel", "hostel") == 0.5


def test_f1_is_symmetric():
    a, b = "free hostel for students", "hostel for students"
    assert token_f1(a, b) == pytest.approx(token_f1(b, a))


def test_two_empty_answers_agree_but_one_empty_does_not():
    assert token_f1("", "") == 1.0
    assert token_f1("", "something") == 0.0


def test_cross_script_answers_score_against_the_matching_gold():
    """Items carry both answer_gold and answer_gold_hi; either may be correct."""
    o = QAOutcome(
        item_id="q1",
        slice_key="HI->EN",
        language_group="cross-lingual",
        prediction="3,50,000 रुपये प्रति वर्ष",
        golds=["Rs. 3,50,000 per annum", "3,50,000 रुपये प्रति वर्ष"],
    )
    assert o.em == 1.0


# --- report ----------------------------------------------------------------------


def _o(item_id, pred, gold, *, group="monolingual", abstained=False):
    return QAOutcome(
        item_id=item_id,
        slice_key="EN->EN",
        language_group=group,
        prediction=pred,
        golds=[gold],
        abstained=abstained,
    )


def test_report_slices_by_language_group():
    r = QAReport(
        "sys",
        [
            _o("a", "x", "x", group="monolingual"),
            _o("b", "y", "z", group="cross-lingual"),
        ],
    )
    assert r.exact_match("monolingual") == 1.0
    assert r.exact_match("cross-lingual") == 0.0
    assert r.exact_match() == 0.5


def test_citation_support_rate_excludes_abstentions():
    """Refusing to answer is not a grounding failure; counting it as one would
    reward a system that answers nothing.

    Asserted on GroundingReport, which owns this metric. QAReport carried a
    second copy reading a field nothing ever populated, so it returned 0.0
    always -- and this test passed regardless, because it set that field by
    hand. A test that constructs the state production never reaches proves the
    formula and nothing about the pipeline.
    """
    from indicrag.evaluation.grounding import GroundingOutcome, GroundingReport

    report = GroundingReport(
        system="sys",
        outcomes=[
            GroundingOutcome(item_id="a", arm="C", lexical_support=1.0),
            GroundingOutcome(item_id="b", arm="C", lexical_support=0.0),
            GroundingOutcome(item_id="c", arm="C", abstained=True),
        ],
    )
    assert report.citation_support_rate("C") == 0.5
    assert report.abstention_rate("C") == pytest.approx(1 / 3)


# --- fabricated citations --------------------------------------------------------


def _ground(**kw):
    from indicrag.evaluation.grounding import score_generation

    base = dict(item_id="i", arm="A", answer="Rs. 12,000", abstained=False, evidence="Rs. 12,000")
    return score_generation(**{**base, **kw})


def test_declining_to_cite_is_not_a_fabricated_citation():
    """Collapsing these two scores an honest abstention from citing as an
    invented passage id."""
    assert _ground(cited=False, cited_found=False).fabricated_citation is False


def test_citing_a_passage_that_does_not_exist_is_fabrication():
    assert _ground(cited=True, cited_found=False).fabricated_citation is True


def test_citing_a_real_passage_is_not_fabrication():
    assert _ground(cited=True, cited_found=True).fabricated_citation is False


def test_the_rate_is_none_rather_than_zero_when_nothing_cited():
    """0.0 would read as a system that cites faithfully."""
    from indicrag.evaluation.grounding import GroundingReport

    report = GroundingReport(system="s", outcomes=[_ground(cited=False)])
    assert report.fabricated_citation_rate("A") is None
    assert report.citation_rate("A") == 0.0


def test_the_rate_is_over_citing_answers_not_all_answers():
    from indicrag.evaluation.grounding import GroundingReport

    report = GroundingReport(
        system="s",
        outcomes=[
            _ground(cited=True, cited_found=False),
            _ground(cited=True, cited_found=True),
            _ground(cited=False),
            _ground(cited=False),
        ],
    )
    assert report.fabricated_citation_rate("A") == 0.5
    assert report.citation_rate("A") == 0.5


def test_abstentions_are_excluded_from_both_rates():
    from indicrag.evaluation.grounding import GroundingReport

    report = GroundingReport(
        system="s",
        outcomes=[
            _ground(cited=True, cited_found=False),
            _ground(abstained=True, cited=False),
        ],
    )
    assert report.citation_rate("A") == 1.0
    assert report.fabricated_citation_rate("A") == 1.0
