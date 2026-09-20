"""Text normalization applied identically to passages and queries.

The single most important property of this module is that **both sides run the
same code path**. If a passage is normalized one way at index time and a query
another way at search time, the two stop matching on exactly the cases that
matter -- amounts, scheme names, dates -- and the failure looks like poor model
quality rather than a preprocessing bug. There is a test asserting the functions
here are idempotent and side-agnostic.

The digit mapping deserves particular note. Amounts are the most common answer
type in this corpus, and the same figure appears as `Rs. 3,50,000`, `₹3,50,000`,
`3.5 lakh`, `3,50,000/-` and `३,५०,०००` across documents. Retrieval and exact-match
scoring both need these to collapse to one key, while the *displayed* text must
keep whatever the document actually said -- hence `text` and `text_raw` travelling
together everywhere downstream.
"""

from __future__ import annotations

import re
import unicodedata

# --- Character-level tables ----------------------------------------------------

DEVANAGARI_DIGITS = "०१२३४५६७८९"
ASCII_DIGITS = "0123456789"
_DIGIT_MAP = str.maketrans(DEVANAGARI_DIGITS, ASCII_DIGITS)

# Zero-width joiner / non-joiner. Their presence in extracted PDF text is
# inconsistent -- the same word appears with and without them in one document --
# which silently fragments tokens. Devanagari needs ZWNJ in a few genuine places
# but none of them occur in this register of formal prose.
ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)

DANDA = "।"
DOUBLE_DANDA = "॥"

# PDF extraction artefacts: soft hyphen, various dashes and quotes that differ
# between the English and Hindi versions of the same circular.
_PUNCT_MAP = {
    "­": "",  # soft hyphen
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    " ": " ",
}
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_MAP)))

_WS_RE = re.compile(r"[ \t ]+")
_NL_RE = re.compile(r"\n{3,}")
# A line break mid-sentence is a PDF layout artefact, not a paragraph break.
# Join it when the next line starts lowercase or with Devanagari.
_SOFTWRAP_RE = re.compile(r"(?<=[^\n.;:।])\n(?=[a-zऀ-ॿ])")


# --- Public API ----------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """NFC composition plus zero-width removal.

    NFC matters because Devanagari can be represented decomposed (consonant +
    combining nukta) or composed (precomposed letter), and the two forms compare
    unequal while rendering identically.
    """
    text = unicodedata.normalize("NFC", text)
    return text.translate(ZERO_WIDTH)


def normalize_digits(text: str) -> str:
    """Map Devanagari digits to ASCII.

    A query typed with ASCII digits must match a passage written with Devanagari
    ones. Applied to both sides.
    """
    return text.translate(_DIGIT_MAP)


def normalize_punctuation(text: str) -> str:
    """Collapse the dash/quote/ellipsis variants that differ between EN and HI files."""
    return _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group(0)], text)


def normalize_whitespace(text: str, *, join_softwrap: bool = True) -> str:
    """Collapse runs of spaces, repair PDF soft wraps, cap blank-line runs."""
    text = _WS_RE.sub(" ", text)
    if join_softwrap:
        text = _SOFTWRAP_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# --- Amount canonicalization ---------------------------------------------------

# The number alternation must try the comma-grouped form FIRST, and the plain
# digit run must be a separate branch rather than a `*` quantifier on the groups.
# Written as `\d{1,3}(?:,\d{2,3})*`, a plain "350000" matches only its first
# three digits and canonicalises to 350 -- so "350000" and "३,५०,०००" compared
# unequal, and every Exact Match on an un-grouped amount was silently wrong.
_CURRENCY_RE = re.compile(
    r"(?:(?:Rs\.?|INR|₹|रु\.?|रुपये|रुपए)\s*)?"
    r"(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*/-)?"
    r"(?:\s*(lakh|lakhs|lac|crore|crores|लाख|करोड़))?",
    re.IGNORECASE,
)

_MULTIPLIER = {
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "लाख": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "करोड़": 10_000_000,
}


def canonical_amounts(text: str) -> list[str]:
    """Extract every monetary figure as a canonical integer string.

    `Rs. 3,50,000`, `₹3.5 lakh`, `350000/-` and `३,५०,०००` all yield `"350000"`.
    Used as an additional matching signal and by the QA scorer, so that a correct
    answer written in a different surface form is not scored wrong.
    """
    text = normalize_digits(normalize_unicode(text))
    out: list[str] = []
    for m in _CURRENCY_RE.finditer(text):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        if not raw:
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if unit in _MULTIPLIER:
            value *= _MULTIPLIER[unit]
        if value >= 100:  # ignore bare small integers: clause numbers, years handled below
            out.append(str(int(value)))
    return out


# --- Composite -----------------------------------------------------------------


def normalize_text(text: str, *, join_softwrap: bool = True) -> str:
    """The full passage/query normalizer. Idempotent.

    Order matters: unicode first (so later regexes see composed forms), digits
    before punctuation (so `३,५०,०००/-` is reachable), whitespace last.
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    text = normalize_digits(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text, join_softwrap=join_softwrap)
    return text


#: Abbreviations whose trailing full stop is not a sentence boundary. `Rs.` is
#: the one that matters: it precedes an amount in most answers in this corpus, so
#: splitting after it truncates "Rs. 12,000 per annum" to "12,000 per annum" --
#: which then scores as a miss on Exact Match against the document's own wording.
#: Found by reading an extracted answer, not from a failing test.
_ABBREVIATIONS = (
    "rs", "no", "sr", "jr", "dr", "mr", "mrs", "ms", "smt", "shri", "st",
    "vs", "etc", "viz", "approx", "govt", "deptt", "dept", "ltd", "pvt",
    "i.e", "e.g", "fig", "vol", "pp", "ch", "sec", "art", "cl",
)
_ABBREV_RE = re.compile(
    r"(?<![\w.])(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.\s",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(rf"(?<=[.!?{DANDA}{DOUBLE_DANDA}])\s+")
_ABBREV_GUARD = "\x00"


def split_sentences(text: str) -> list[str]:
    """Split on both Latin and Devanagari sentence terminators.

    Danda (।) and double danda (॥) are Hindi's full stops; a Latin-only splitter
    returns an entire Hindi paragraph as one sentence, which breaks segmentation
    and every sentence-level scorer downstream.

    Abbreviation stops are protected first -- see `_ABBREVIATIONS`. The guard is a
    NUL placeholder rather than a lookbehind because Python's `re` requires
    fixed-width lookbehind and the abbreviations differ in length.
    """
    if not text:
        return []
    guarded = _ABBREV_RE.sub(lambda m: m.group(1) + _ABBREV_GUARD, text)
    parts = _SENT_SPLIT_RE.split(guarded)
    return [p.replace(_ABBREV_GUARD, ". ").strip() for p in parts if p.strip()]
