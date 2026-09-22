"""Passage-level invariants for a segmented corpus.

`corpus validate` checks the *extracted text* -- is this really Devanagari, did
the PDF font mapping survive. Nothing checked the *passages*, and that is where
the promises accumulate: `models.Passage` documents `text_raw` as "what the
document actually said", `docs/ARCHITECTURE.md` lists `char_span` twice as
carried provenance metadata, and `segment.py` states a 240-token bound and a rule
that overlap never crosses a section boundary.

Several of those promises were broken in the committed corpus, and none of them
raised anything. That is the pattern this module exists to break. A field that
nothing reads cannot be wrong loudly, so the audit reads all of them once and
says what it finds.

The checks are independent and each reports a rate rather than a bare pass/fail,
because "3 of 694" and "656 of 694" are different problems with different fixes
and a boolean cannot tell them apart.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..models import Document, Passage
from .normalize import normalize_text
from .segment import MAX_TOKENS, count_tokens, split_sections

#: A span is accepted when the source slice it names recovers the passage's own
#: words. Exact equality is too strict -- the slice legitimately includes the
#: whitespace the sentence splitter folded -- so this compares token sets.
SPAN_JACCARD_FLOOR = 0.95

#: A passage belongs to one section when a single section's vocabulary covers
#: this much of it. Not 1.0: the sentence splitter and the normalizer both move
#: a few tokens around, and a passage that is 99% one section is not the failure
#: mode being looked for, which is a chunk welded to unrelated material.
SECTION_COVERAGE_FLOOR = 0.98

#: Above this, a "raw" text that survives normalization unchanged is evidence the
#: field was filled from the normalized stream rather than the source. One
#: passage can legitimately be a fixed point (it contained nothing to normalize);
#: nearly all of them cannot.
RAW_FIXEDPOINT_CEILING = 0.90

#: ...but only once there are enough passages for the rate to mean anything. An
#: all-ASCII fixture with a single passage is a 100% fixed-point corpus and no
#: evidence of anything, so below this the check abstains rather than reporting a
#: failure it cannot support.
RAW_MIN_PASSAGES = 20


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


@dataclass
class Finding:
    """One failed invariant, with enough context to act on it."""

    check: str
    passage_id: str
    detail: str


@dataclass
class CorpusAudit:
    """Outcome of auditing a passage set against its source documents.

    `rates` is the reportable surface: one fraction per check, in [0, 1], where
    1.0 means every passage satisfied it. `findings` is capped per check so a
    systematically broken corpus prints a diagnosis rather than 694 lines.
    """

    #: Checks that must hold for the corpus to be usable at all, as distinct from
    #: those that degrade a feature. A duplicate ID silently drops a passage from
    #: every downstream map; a drifted char_span misleads a reader. Both are
    #: defects, only the first invalidates the experiments.
    BLOCKING = ("unique_ids", "nonempty_text", "token_count_accurate", "source_present")

    n_passages: int = 0
    n_documents: int = 0
    rates: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> list[str]:
        return [k for k, v in sorted(self.rates.items()) if v < 1.0]

    @property
    def blocking_failures(self) -> list[str]:
        return [k for k in self.failures if k in self.BLOCKING]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_passages": self.n_passages,
            "n_documents": self.n_documents,
            "rates": {k: round(v, 4) for k, v in sorted(self.rates.items())},
            "counts": dict(sorted(self.counts.items())),
            "failures": self.failures,
            "blocking_failures": self.blocking_failures,
        }


def audit(passages: list[Passage], sources: dict[str, str]) -> CorpusAudit:
    """Check every passage-level invariant the codebase claims.

    `sources` maps doc_id to the extracted text the passages were cut from.
    Passages whose document is absent fail `source_present` and are skipped by
    the checks that need it, rather than silently passing.
    """
    report = CorpusAudit(n_passages=len(passages), n_documents=len(sources))
    if not passages:
        return report

    per_check: dict[str, list[bool]] = {}
    seen: Counter[str] = Counter(p.passage_id for p in passages)

    def record(check: str, passage: Passage, passed: bool, detail: str = "") -> None:
        per_check.setdefault(check, []).append(passed)
        if not passed and sum(1 for f in report.findings if f.check == check) < 5:
            report.findings.append(Finding(check, passage.passage_id, detail))

    # Sentence-to-section vocabularies, built once per document. This is how
    # section spanning is detected without trusting char_span: a passage belongs
    # to one section iff that section's words cover it.
    sections: dict[str, list[set[str]]] = {
        doc_id: [set(normalize_text(s.text).split()) for s in split_sections(text)]
        for doc_id, text in sources.items()
    }

    for p in passages:
        record("unique_ids", p, seen[p.passage_id] == 1, f"appears {seen[p.passage_id]}x")
        record("nonempty_text", p, bool(p.text.strip()) and bool(p.text_raw.strip()))

        actual = count_tokens(p.text)
        record("token_count_accurate", p, p.token_count == actual,
               f"stored {p.token_count}, actual {actual}")
        record("within_max_tokens", p, actual <= MAX_TOKENS,
               f"{actual} tokens against a stated bound of {MAX_TOKENS}")

        src = sources.get(p.doc_id)
        record("source_present", p, src is not None, f"no extracted text for {p.doc_id}")
        if src is None:
            continue

        # Two separate questions, because "no span" and "wrong span" are
        # different states with different costs. An unset span is honest: the
        # aligner could not place this passage and said so. A span that names
        # the wrong text is the failure mode that misleads a reader, and
        # collapsing the two would let a repair hide its misses among its
        # refusals.
        a, b = p.char_span
        record("span_present", p, (a, b) != (0, 0), "char_span is unset")
        if (a, b) != (0, 0):
            # Compared against `text`, not `text_raw`, and through the
            # normalizer. Checking the slice against `text_raw` would be
            # circular the moment anything sets one from the other -- which is
            # exactly what the repair does -- and would then pass by
            # construction while proving nothing.
            j = _jaccard(normalize_text(src[a:b]), p.text)
            record("span_recovers_text", p, j >= SPAN_JACCARD_FLOOR,
                   f"char_span ({a},{b}) recovers {j:.0%} of the passage's tokens")

        # Overlap sentences are drawn from the previous chunk of the *same*
        # section, so carrying them does not create a false positive here.
        words = set(p.text.split())
        covered = max(
            (len(words & vocab) / len(words) if words else 1.0
             for vocab in sections.get(p.doc_id, [])),
            default=1.0,
        )
        record("single_section", p, covered >= SECTION_COVERAGE_FLOOR,
               f"best-matching section covers {covered:.0%} of its words")

    fixed_points = sum(
        1 for p in passages
        if p.text_raw and normalize_text(p.text_raw) == p.text_raw
    )
    report.counts["text_raw_normalization_fixed_points"] = fixed_points
    # Reported as a corpus-level rate, not per passage: a single fixed point says
    # nothing, and the claim being tested -- that text_raw is the source text --
    # is only falsifiable in aggregate.
    if len(passages) >= RAW_MIN_PASSAGES:
        per_check["raw_preserves_source"] = [
            fixed_points / len(passages) <= RAW_FIXEDPOINT_CEILING
        ]

    for check, results in per_check.items():
        report.rates[check] = sum(results) / len(results) if results else 1.0
        report.counts.setdefault(check, sum(1 for r in results if not r))
    return report


def format_audit(report: CorpusAudit) -> list[str]:
    """Render an audit for the console and for `--report`, one code path."""
    out = [
        f"{report.n_passages} passages from {report.n_documents} documents",
        "",
        f"  {'check':<24}{'pass':>8}{'failing':>10}",
        "  " + "-" * 44,
    ]
    for check in sorted(report.rates):
        rate = report.rates[check]
        bad = report.counts.get(check, 0)
        mark = "" if rate == 1.0 else ("  BLOCKING" if check in report.BLOCKING else "  *")
        out.append(f"  {check:<24}{rate:>7.1%}{bad:>10}{mark}")
    if report.findings:
        out += ["", "  examples"]
        for f in report.findings:
            out.append(f"    {f.check:<22}{f.passage_id:<28}{f.detail}")
    out += [""]
    if report.ok:
        out.append("  every passage-level invariant holds")
    else:
        out.append(f"  failing: {', '.join(report.failures)}")
        if report.blocking_failures:
            out.append(f"  BLOCKING: {', '.join(report.blocking_failures)}")
    return out


def load_sources(text_dir, docs: list[Document]) -> dict[str, str]:
    """Read the extracted text for each manifest document that has one."""
    out: dict[str, str] = {}
    for doc in docs:
        path = text_dir / f"{doc.doc_id}.txt"
        if path.exists():
            out[doc.doc_id] = path.read_text(encoding="utf-8")
    return out
