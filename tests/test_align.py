"""Tests for locating normalized passage text back in raw source.

The interesting cases are all ones where the passage is *not* a substring of its
document: normalization rewrote it, or segmentation duplicated part of it. A
naive `source.find(passage)` passes none of these, which is roughly why the
field this replaces was wrong for three passages in four.
"""

from __future__ import annotations

from indicrag.corpus.align import (
    MIN_COVERAGE,
    SourceIndex,
    locate,
    token_after,
)
from indicrag.corpus.normalize import normalize_text

SOURCE = (
    "== Eligibility ==\n"
    "The scheme is open to students whose annual family income is below "
    "₹२,००,००० per year. Applicants must be "
    "enrolled in a recognised institution. Renewal requires seventy-five per "
    "cent attendance.\n"
    "== Monitoring ==\n"
    "District officers submit quarterly utilisation certificates to the state "
    "nodal agency, which forwards them to the ministry.\n"
)

FIRST = "The scheme is open to students whose annual family income is below"
RENEWAL = "Renewal requires seventy-five per cent attendance."
MONITORING = "District officers submit quarterly utilisation certificates"


def index() -> SourceIndex:
    return SourceIndex(SOURCE)


def test_a_passage_is_located_where_the_source_actually_says_it():
    span = locate(index(), normalize_text(FIRST))
    assert span is not None
    assert SOURCE[span.start : span.end].startswith("The scheme is open")
    assert span.coverage == 1.0


def test_the_located_slice_normalizes_back_to_the_passage():
    # The property the repair depends on, and the one the audit checks.
    passage = normalize_text(
        "Applicants must be enrolled in a recognised institution."
    )
    span = locate(index(), passage)
    assert span is not None
    assert normalize_text(SOURCE[span.start : span.end]) == passage


def test_a_passage_whose_digits_were_normalized_is_still_found():
    # The source writes the amount in Devanagari digits; the passage carries the
    # ASCII form. Nothing matches on a raw substring search.
    passage = normalize_text(FIRST + " ₹२,००,००० per year.")
    assert "2,00,000" in passage
    assert "2,00,000" not in SOURCE
    span = locate(index(), passage)
    assert span is not None
    assert span.coverage >= MIN_COVERAGE


def test_a_passage_that_repeats_its_overlap_is_still_located():
    # segment.py's short-chunk merge concatenates a chunk that already carries
    # its own overlap prefix, so the same sentence appears twice and the second
    # copy sits behind where the first ended. A forward-only walk fails here.
    passage = normalize_text(RENEWAL + " " + RENEWAL + " " + MONITORING)
    span = locate(index(), passage)
    assert span is not None


def test_the_span_of_a_welded_passage_is_the_hull_of_its_parts():
    passage = normalize_text(RENEWAL + " " + MONITORING)
    span = locate(index(), passage)
    assert span is not None
    covered = SOURCE[span.start : span.end]
    assert "Renewal requires" in covered
    assert "District officers" in covered
    # Honest about what it swallowed: the heading between them is inside.
    assert "== Monitoring ==" in covered


def test_text_absent_from_the_document_is_declined():
    assert locate(index(), "The applicant must hold a valid fishing licence") is None


def test_an_empty_passage_is_declined():
    assert locate(index(), "") is None
    assert locate(index(), "   ") is None


def test_an_empty_document_declines_everything():
    assert locate(SourceIndex(""), normalize_text(FIRST)) is None


def test_a_cursor_does_not_prevent_finding_earlier_text():
    # Overlap re-emits earlier sentences, so a passage can legitimately start
    # behind the cursor.
    idx = index()
    late = locate(idx, normalize_text(MONITORING))
    assert late is not None
    early = locate(idx, normalize_text(FIRST), cursor=token_after(idx, late.end))
    assert early is not None
    assert early.start < late.start


def test_a_weak_anchor_ahead_does_not_beat_the_real_match_behind():
    # The failure this guards: `_anchor` accepts a half-matching window, so a
    # forward search from the cursor can return a non-null but wrong result --
    # and a retry that only fires on None never runs. 20 of 23 passages were
    # declined this way, every sentence of which aligns perfectly from zero.
    doc = SourceIndex(
        "Alpha beta gamma delta epsilon zeta. "
        "Filler sentence in between here. "
        "Alpha beta gamma different words entirely."
    )
    wanted = "Alpha beta gamma delta epsilon zeta."
    past_it = token_after(doc, doc.text.index("Filler"))
    span = locate(doc, wanted, cursor=past_it)
    assert span is not None
    assert doc.text[span.start : span.end].startswith("Alpha beta gamma delta")


def test_a_cursor_disambiguates_a_repeated_sentence():
    doc = SourceIndex("Alpha beta gamma delta. Filler text here. Alpha beta gamma delta.")
    first = locate(doc, "Alpha beta gamma delta.")
    assert first is not None
    second = locate(doc, "Alpha beta gamma delta.", cursor=token_after(doc, first.end))
    assert second is not None
    assert second.start > first.start


def test_token_after_walks_past_a_character_offset():
    idx = index()
    assert token_after(idx, 0) == 0
    assert token_after(idx, len(SOURCE)) == len(idx)


def test_coverage_is_reported_so_a_caller_can_refuse():
    span = locate(index(), normalize_text(FIRST))
    assert span is not None
    assert 0.0 <= span.coverage <= 1.0
    assert MIN_COVERAGE < 1.0
