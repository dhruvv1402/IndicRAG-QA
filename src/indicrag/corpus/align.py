"""Locate normalized passage text back in its raw source document.

Segmentation normalizes a section, splits it into sentences, packs those into
chunks, and then throws the correspondence away. `char_span` was an attempt to
keep it that did not survive contact with a normalizer that changes length:
`segment.py` advanced a cursor by the normalized chunk length while searching
raw text, so the offsets drifted monotonically and 3 passages in 4 named a slice
that does not contain them.

Recovering the mapping properly means aligning two strings that differ by local
rewrites -- digit transliteration, punctuation folding, soft-wrap joining,
whitespace collapse. Three things make that tractable here:

* Normalization is *token-local* almost everywhere. Normalizing a token on its
  own gives the same result as normalizing the document and reading that token
  back, except where soft-wrap joining merges a hyphenated pair.
* Passages are emitted in document order, so a forward cursor per document turns
  a global search into a local one and removes almost every false anchor.
* Passage text is long. Matching on a window of tokens rather than one makes an
  accidental anchor vanishingly unlikely even for a passage that opens with
  "The scheme".

So: tokenize the raw source once, keeping each token's offsets; normalize each
token; then walk the passage's tokens against that stream, allowing either side
to skip where a merge or a split happened. The result is the raw span, from
which the true `text_raw` is simply a slice.

This is deliberately not a general diff. `difflib` over a 30 KB document is both
slower and less accurate here, because it has no notion that a run of tokens
must stay contiguous.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from .normalize import normalize_text, split_sentences

#: Tokens compared when anchoring the start of a passage. One token anchors on
#: "The" and lands anywhere; this many in sequence is effectively unique within
#: the forward window a single document offers.
ANCHOR_WINDOW = 6

#: How far a search may run ahead of the cursor before giving up, in tokens.
#: Generous enough to skip a dropped section (`_SKIP_SECTIONS` removes reference
#: lists mid-document, so the cursor legitimately jumps), bounded so a passage
#: that genuinely is not present fails rather than matching something distant.
MAX_LOOKAHEAD = 20000

#: Fraction of a passage's tokens that must align for the span to be returned.
#: Below this the alignment is not trusted and the caller gets nothing, which is
#: the honest outcome: a wrong span is worse than an absent one.
MIN_COVERAGE = 0.90

_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Span:
    """A located passage: raw character offsets plus the alignment quality."""

    start: int
    end: int
    coverage: float

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, self.end)


class SourceIndex:
    """A raw document tokenized once, with per-token offsets and normal forms.

    Built once per document and reused across its passages; rebuilding it per
    passage turns the repair from seconds into minutes.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.norms: list[str] = []
        for m in _TOKEN_RE.finditer(text):
            norm = normalize_text(m.group(0))
            self.starts.append(m.start())
            self.ends.append(m.end())
            # A token can normalize to empty (a stray control character) or to
            # several tokens (punctuation folding splits "12,000/-"). Store the
            # first form for matching and keep the slot so offsets stay aligned.
            self.norms.append(norm.split()[0] if norm.split() else "")

    def __len__(self) -> int:
        return len(self.norms)


def _anchor(index: SourceIndex, wanted: list[str], cursor: int) -> int:
    """First position at or after `cursor` where `wanted` starts to match.

    Scores a window rather than requiring exact equality: normalization is
    token-local *almost* everywhere, and a single merged hyphenation inside the
    window should not reject an otherwise correct anchor.
    """
    window = wanted[:ANCHOR_WINDOW]
    if not window:
        return -1
    best, best_score = -1, 0.0
    limit = min(len(index), cursor + MAX_LOOKAHEAD)
    for i in range(cursor, limit):
        if index.norms[i] != window[0]:
            continue
        hit = sum(
            1 for k, tok in enumerate(window)
            if i + k < len(index) and index.norms[i + k] == tok
        )
        score = hit / len(window)
        if score > best_score:
            best, best_score = i, score
            if score == 1.0:
                break
    return best if best_score >= 0.5 else -1


def _locate_run(
    index: SourceIndex, wanted: list[str], cursor: int
) -> tuple[int, int, int] | None:
    """Align one contiguous token run. Returns (start_tok, end_tok, matched)."""
    if not wanted:
        return None
    start_tok = _anchor(index, wanted, cursor)
    if start_tok < 0:
        return None

    # Walk both sequences forward. The source may carry tokens the run does not
    # (normalization split one into two) and vice versa (it merged two), so each
    # side may skip -- but only by a little, and never backwards.
    i, j, matched, misses = start_tok, 0, 0, 0
    last_match = start_tok
    while j < len(wanted) and i < len(index):
        if index.norms[i] == wanted[j]:
            matched += 1
            last_match = i
            i += 1
            j += 1
            continue
        if i + 1 < len(index) and index.norms[i + 1] == wanted[j]:
            i += 1
        elif j + 1 < len(wanted) and index.norms[i] == wanted[j + 1]:
            j += 1
        else:
            i += 1
            j += 1
            misses += 1
        if misses > max(2, len(wanted) // 4):
            break
    return (start_tok, last_match, matched)


def locate(index: SourceIndex, passage_text: str, cursor: int = 0) -> Span | None:
    """Find the raw span of `passage_text` in the indexed document.

    Aligned sentence by sentence rather than as one run, because a passage is
    not always contiguous in its source. Two constructions in `segment.py` see
    to that: overlap prepends the tail of the previous chunk, and the short-chunk
    merge concatenates a chunk that already carries its own overlap -- so the
    same sentences appear twice, and the second copy sits *behind* where the
    first one ended. A single forward walk cannot survive that backward jump; it
    scored 0.03 coverage on 39 passages that are in fact entirely present in
    their source.

    Each sentence is located independently and the span is their hull. Coverage
    is the share of the passage's tokens that aligned, so a passage welded across
    a section boundary -- whose hull would swallow the heading and everything
    between -- still reports honestly rather than returning a span that contains
    material the passage does not.

    Returns None when coverage is too poor to trust. `cursor` is a token
    position, not a character offset.
    """
    if not passage_text.strip() or not len(index):
        return None
    total = len(passage_text.split())
    if not total:
        return None

    runs = [s.split() for s in split_sentences(passage_text)] or [passage_text.split()]
    bounds: list[tuple[int, int]] = []
    matched = 0
    search = cursor
    for run in runs:
        got = _locate_run(index, run, search)
        if got is None and search > 0:
            # Overlap re-emits earlier text, so a sentence may legitimately sit
            # behind the cursor. Retry from the top before giving up on it.
            got = _locate_run(index, run, 0)
        if got is None:
            continue
        start_tok, end_tok, hits = got
        bounds.append((start_tok, end_tok))
        matched += hits
        search = end_tok + 1

    if not bounds:
        return None
    coverage = matched / total
    if coverage < MIN_COVERAGE:
        return None
    return Span(index.starts[min(b[0] for b in bounds)],
                index.ends[max(b[1] for b in bounds)],
                coverage)


def token_after(index: SourceIndex, char_end: int) -> int:
    """Token position at or after a character offset, for advancing a cursor."""
    for i, start in enumerate(index.starts):
        if start >= char_end:
            return i
    return len(index)


@dataclass
class RepairResult:
    """What a repair pass did, for reporting before anything is written."""

    located: int = 0
    declined: int = 0
    coverages: list[float] = field(default_factory=list)
    unaligned: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.located + self.declined

    @property
    def rate(self) -> float:
        return self.located / self.total if self.total else 1.0


def repair_spans(passages, sources: dict[str, str]) -> RepairResult:
    """Re-derive `char_span` and `text_raw` for every passage, in place.

    Deliberately touches nothing else. `passage_id` and `text` are what the gold
    set, the embedding caches and the error analysis key off, so leaving them
    untouched is what lets this run against a frozen corpus: no re-indexing, no
    re-running evaluations, no re-anchoring.

    A passage the aligner declines has its span *cleared* rather than left as it
    was. The existing value is the drifted one; keeping it would let a passage
    that could not be placed go on claiming a span that points at other text.
    """
    result = RepairResult()
    indexes: dict[str, SourceIndex] = {}
    cursors: dict[str, int] = {}

    for p in passages:
        src = sources.get(p.doc_id)
        if src is None:
            p.char_span = (0, 0)
            result.declined += 1
            result.unaligned.append(f"{p.passage_id} (no source document)")
            continue
        if p.doc_id not in indexes:
            indexes[p.doc_id] = SourceIndex(src)
            cursors[p.doc_id] = 0

        span = locate(indexes[p.doc_id], p.text, cursors[p.doc_id])
        if span is None:
            p.char_span = (0, 0)
            result.declined += 1
            result.unaligned.append(f"{p.passage_id} ({p.token_count} tokens)")
            continue

        p.char_span = span.as_tuple()
        p.text_raw = src[span.start : span.end]
        cursors[p.doc_id] = token_after(indexes[p.doc_id], span.end)
        result.coverages.append(span.coverage)
        result.located += 1
    return result


def format_repair(result: RepairResult) -> list[str]:
    """Render a repair pass for the console and for `--report`."""
    out = [
        f"{result.total} passages",
        f"  located     {result.located:4d}  {result.rate:6.1%}",
        f"  declined    {result.declined:4d}",
    ]
    if result.coverages:
        out.append(
            f"  coverage    median {statistics.median(result.coverages):.3f}  "
            f"min {min(result.coverages):.3f}"
        )
    if result.unaligned:
        out += ["", "  could not align (char_span cleared):"]
        out += [f"    {line}" for line in result.unaligned[:10]]
        if len(result.unaligned) > 10:
            out.append(f"    ... and {len(result.unaligned) - 10} more")
    return out
