"""Tests for passage-level corpus invariants.

Each test builds the *defect* the check exists to catch, because a check that
only ever sees clean input is indistinguishable from `return True` -- which is
effectively what the codebase had before this module: four documented invariants
and nothing that read them.
"""

from __future__ import annotations

from indicrag.corpus.integrity import (
    RAW_FIXEDPOINT_CEILING,
    RAW_MIN_PASSAGES,
    SPAN_JACCARD_FLOOR,
    audit,
    format_audit,
)
from indicrag.models import Passage

SOURCE = (
    "== Eligibility ==\n"
    "The scheme is open to students whose family income is below two lakh rupees "
    "per year. Applicants must be enrolled in a recognised institution.\n"
    "== Monitoring ==\n"
    "District officers submit quarterly utilisation certificates to the state "
    "nodal agency, which forwards them to the ministry.\n"
)


def passage(text: str, **kw) -> Passage:
    body = dict(
        passage_id="doc#p0001",
        doc_id="doc",
        scheme="test",
        lang="en",
        text=text,
        text_raw=text,
        token_count=len(text.split()),
    )
    body.update(kw)
    return Passage(**body)


def span_of(text: str) -> tuple[int, int]:
    start = SOURCE.index(text)
    return (start, start + len(text))


ELIGIBILITY = (
    "The scheme is open to students whose family income is below two lakh rupees "
    "per year. Applicants must be enrolled in a recognised institution."
)
MONITORING = (
    "District officers submit quarterly utilisation certificates to the state "
    "nodal agency, which forwards them to the ministry."
)


def test_a_faithful_passage_satisfies_every_check():
    p = passage(ELIGIBILITY, char_span=span_of(ELIGIBILITY))
    report = audit([p], {"doc": SOURCE})
    assert report.ok, report.failures
    assert report.rates["span_recovers_text"] == 1.0
    assert report.rates["single_section"] == 1.0


def test_a_char_span_pointing_at_the_wrong_text_is_caught():
    p = passage(ELIGIBILITY, char_span=span_of(MONITORING))
    report = audit([p], {"doc": SOURCE})
    assert "span_recovers_text" in report.failures
    assert report.rates["span_recovers_text"] == 0.0


def test_a_char_span_running_past_the_end_of_the_document_is_caught():
    # The drift in the committed corpus is monotonic, so late passages in a
    # document name a slice that is empty rather than merely wrong.
    p = passage(ELIGIBILITY, char_span=(len(SOURCE) + 500, len(SOURCE) + 900))
    report = audit([p], {"doc": SOURCE})
    assert "span_recovers_text" in report.failures


def test_a_span_that_merely_includes_extra_whitespace_still_passes():
    start, end = span_of(ELIGIBILITY)
    p = passage(ELIGIBILITY, char_span=(start, end + 1))
    report = audit([p], {"doc": SOURCE})
    assert report.rates["span_recovers_text"] == 1.0
    assert SPAN_JACCARD_FLOOR < 1.0


def test_a_passage_welded_across_two_sections_is_caught():
    welded = ELIGIBILITY + " " + MONITORING
    p = passage(welded, char_span=(0, len(SOURCE)))
    report = audit([p], {"doc": SOURCE})
    assert "single_section" in report.failures


def test_overlap_carried_from_the_same_section_is_not_mistaken_for_welding():
    # Overlap duplicates the tail of the previous chunk. It stays inside one
    # section by construction, so it must not trip the welding check.
    overlapped = "Applicants must be enrolled in a recognised institution. " + ELIGIBILITY
    p = passage(overlapped, char_span=span_of(ELIGIBILITY))
    report = audit([p], {"doc": SOURCE})
    assert "single_section" not in report.failures


def test_a_duplicate_passage_id_is_blocking():
    a = passage(ELIGIBILITY, char_span=span_of(ELIGIBILITY))
    b = passage(MONITORING, char_span=span_of(MONITORING))
    report = audit([a, b], {"doc": SOURCE})
    assert "unique_ids" in report.blocking_failures


def test_a_stale_token_count_is_blocking():
    p = passage(ELIGIBILITY, char_span=span_of(ELIGIBILITY), token_count=3)
    report = audit([p], {"doc": SOURCE})
    assert "token_count_accurate" in report.blocking_failures


def test_a_passage_over_the_stated_token_bound_is_caught_but_not_blocking():
    long = " ".join(["word"] * 400)
    p = passage(long, char_span=(0, 0))
    report = audit([p], {"doc": SOURCE})
    assert "within_max_tokens" in report.failures
    assert "within_max_tokens" not in report.blocking_failures


def test_a_passage_whose_document_was_never_extracted_is_blocking():
    p = passage(ELIGIBILITY, doc_id="missing", char_span=(0, 100))
    report = audit([p], {"doc": SOURCE})
    assert "source_present" in report.blocking_failures


def test_text_raw_filled_from_the_normalized_stream_is_caught():
    # Every text_raw here is already normalized, which is exactly the state the
    # committed corpus is in: 691 of 694 are fixed points of normalize_text.
    ps = [
        passage(ELIGIBILITY, passage_id=f"doc#p{i:04d}", char_span=span_of(ELIGIBILITY))
        for i in range(RAW_MIN_PASSAGES)
    ]
    report = audit(ps, {"doc": SOURCE})
    assert "raw_preserves_source" in report.failures
    assert report.counts["text_raw_normalization_fixed_points"] == RAW_MIN_PASSAGES


#: A correctly segmented passage: `text` normalized for matching, `text_raw` as
#: the document spelled it, and a span that names the raw slice. This is the
#: state the committed corpus was supposed to be in.
RAW_SENTENCE = "The grant is ₹१२०० per year for every enrolled student."
RAW_SOURCE = "== Grant ==\n" + RAW_SENTENCE + "\n"
CLEAN_TEXT = RAW_SENTENCE.replace("१२००", "1200")


def clean_corpus(n: int = RAW_MIN_PASSAGES) -> list[Passage]:
    start = RAW_SOURCE.index(RAW_SENTENCE)
    return [
        passage(
            CLEAN_TEXT,
            passage_id=f"raw#p{i:04d}",
            doc_id="raw",
            text_raw=RAW_SENTENCE,
            char_span=(start, start + len(RAW_SENTENCE)),
        )
        for i in range(n)
    ]


def test_raw_text_carrying_source_spelling_passes():
    report = audit(clean_corpus(), {"raw": RAW_SOURCE})
    assert "raw_preserves_source" not in report.failures
    assert RAW_FIXEDPOINT_CEILING < 1.0


def test_auditing_nothing_does_not_raise():
    report = audit([], {})
    assert report.ok
    assert report.n_passages == 0


def test_the_formatter_names_every_failing_check():
    p = passage(ELIGIBILITY, char_span=span_of(MONITORING), token_count=3)
    text = "\n".join(format_audit(audit([p], {"doc": SOURCE})))
    assert "span_recovers_text" in text
    assert "BLOCKING" in text
    assert "every passage-level invariant holds" not in text


def test_the_formatter_says_so_when_everything_holds():
    text = "\n".join(format_audit(audit(clean_corpus(), {"raw": RAW_SOURCE})))
    assert "every passage-level invariant holds" in text
    assert "BLOCKING" not in text
