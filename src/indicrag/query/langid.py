"""Query language and script identification.

Three classes, exactly as the brief requires: `English`, `Indic`, `Code-Mixed`.

The approach is rule-based, and deliberately so. On a three-class problem where
one class is defined by *script* and the boundary between the other two is
defined by a small closed set of function words, a statistical classifier buys
nothing and costs explainability -- and when a query is misclassified, the answer
comes back in the wrong language, so being able to say exactly why a decision was
made is worth more than a point of accuracy.

The function-word lexicon is the load-bearing part. A query like
"eligibility criteria kya hai" is Latin-script and lexically English apart from
the two Hindi words that determine its type. Content words code-switch freely;
grammatical words do not, which is what makes them a reliable signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_TOKEN_RE = re.compile(r"[a-zA-Zऀ-ॿ]+")

# Closed-class Hindi function words in Latin script, with their common spelling
# variants. These survive code-mixing because they are grammatical rather than
# lexical -- a speaker switching to English for "scholarship" and "eligibility"
# still says "ke liye" and "kya hai".
#
# The split below is not cosmetic, it fixes a real misclassification. Several
# valid Romanizations of Hindi function words are also among the commonest
# English words: "the" (थे, "were"), "is" (इस, "this"), "to" (तो), "us" (उस),
# "me" (में), "tab" (तब), "par" (पर). Counting those as Hindi evidence made
# "What is the minimum eligibility for the scholarship?" score three Hindi
# markers and classify as Code-Mixed -- which would then have the system answer
# an English question in Hinglish, and would corrupt every per-language table in
# the evaluation. Ambiguous markers therefore never trigger Code-Mixed on their
# own; they only corroborate an unambiguous one.
UNAMBIGUOUS_MARKERS = frozenset(
    """
    kya kyaa kyu kyun kyon kaise kaisa kaisi kab kahan kahaan kaun kaunsa kitna
    kitni kitne kitnaa hai hain haii hota hoti hote hona hoga hogi honge tha thi
    ka ki ke ko se mein mai liye pe tak bhi aur nahi nahin chahiye sakta sakte
    sakti milega milta milti wala wali wale apna apni apne mera meri mere tera
    teri aapka aapki uska uski iska iski jis jab agar yadi lekin magar kuch koi
    sab sabhi yeh woh bataye batao karna karne kare karta karti diya dena deta
    milne kitnaa jaruri zaruri jarurat zarurat milti hoti kaunse kisi kisne
    """.split()
)

# Also Hindi, but homographs of common English words. Corroborating evidence only.
AMBIGUOUS_MARKERS = frozenset("the is to us me hi na ya par lie tab wo ye jo so".split())

HINGLISH_MARKERS = UNAMBIGUOUS_MARKERS | AMBIGUOUS_MARKERS

# English function words, used as the counterweight. A query with many of these
# and no Hindi markers is English even if it contains an Indic proper noun.
ENGLISH_MARKERS = frozenset(
    """
    the a an is are was were what which who whom whose when where why how
    of for to in on at by with from and or not do does did can could should
    would will shall have has had be been being i you he she it we they this
    that these those my your his her its our their there here if then than
    """.split()
)

QueryType = str  # "English" | "Indic" | "Code-Mixed"

# Thresholds. The gap between 0.10 and 0.60 is the mixed-script band: a query
# with some Devanagari but not mostly Devanagari is code-mixed by definition.
DEVANAGARI_INDIC_FLOOR = 0.60
DEVANAGARI_LATIN_CEILING = 0.10


@dataclass
class LangIdResult:
    """A classification with the evidence that produced it.

    `reason` exists so that a disagreement between this and the n-gram
    cross-check can be logged with enough context to become a Module 6 error
    case, rather than just a counter going up.
    """

    query_type: QueryType
    script: str  # "deva" | "latin" | "mixed"
    lang: str  # "en" | "hi" | "hi-en"
    devanagari_ratio: float
    hindi_marker_hits: int  # unambiguous only -- the count the decision uses
    english_marker_hits: int
    reason: str
    ambiguous_marker_hits: int = 0

    def as_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "script": self.script,
            "lang": self.lang,
            "devanagari_ratio": round(self.devanagari_ratio, 4),
            "hindi_marker_hits": self.hindi_marker_hits,
            "ambiguous_marker_hits": self.ambiguous_marker_hits,
            "english_marker_hits": self.english_marker_hits,
            "reason": self.reason,
        }


def devanagari_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace() and c.isalpha()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if DEVANAGARI_RE.match(c)) / len(chars)


def _marker_hits(text: str) -> tuple[int, int, int]:
    """Return (unambiguous Hindi, ambiguous Hindi, English) marker counts."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    hi = sum(1 for t in tokens if t in UNAMBIGUOUS_MARKERS)
    amb = sum(1 for t in tokens if t in AMBIGUOUS_MARKERS)
    en = sum(1 for t in tokens if t in ENGLISH_MARKERS)
    return hi, amb, en


def classify(text: str) -> LangIdResult:
    """Classify a query as English / Indic / Code-Mixed.

    Decision order:
      1. Mostly Devanagari                         -> Indic
      2. Devanagari and Latin both present         -> Code-Mixed (mixed script)
      3. Latin only, >=1 unambiguous Hindi marker  -> Code-Mixed (Romanized)
      4. Otherwise                                 -> English

    Step 2 tests for *presence* of both scripts rather than a ratio band. A query
    like "PM-YASASVI के लिए eligibility criteria kya hai?" is only 8% Devanagari
    by character count, which would slip under a 10% floor, yet it is plainly
    code-mixed. Any Devanagari at all alongside Latin letters settles it.
    """
    text = (text or "").strip()
    ratio = devanagari_ratio(text)
    hi, amb, en = _marker_hits(text)

    if not text:
        return LangIdResult("English", "unknown", "en", 0.0, 0, 0, "empty query", 0)

    has_deva = bool(DEVANAGARI_RE.search(text))
    has_latin = any(c.isascii() and c.isalpha() for c in text)

    if ratio >= DEVANAGARI_INDIC_FLOOR:
        return LangIdResult(
            "Indic", "deva", "hi", ratio, hi, en,
            f"devanagari ratio {ratio:.2f} >= {DEVANAGARI_INDIC_FLOOR}", amb,
        )

    if has_deva and has_latin:
        return LangIdResult(
            "Code-Mixed", "mixed", "hi-en", ratio, hi, en,
            f"both scripts present (devanagari ratio {ratio:.2f})", amb,
        )

    if has_deva:
        return LangIdResult(
            "Indic", "deva", "hi", ratio, hi, en,
            "devanagari only, below the indic floor but no latin present", amb,
        )

    if hi >= 1:
        return LangIdResult(
            "Code-Mixed", "latin", "hi-en", ratio, hi, en,
            f"latin script with {hi} unambiguous hindi marker(s), {amb} ambiguous", amb,
        )

    return LangIdResult(
        "English", "latin", "en", ratio, hi, en,
        f"latin script, no unambiguous hindi markers ({amb} ambiguous ignored)", amb,
    )


def is_code_mixed(text: str) -> bool:
    return classify(text).query_type == "Code-Mixed"


def slice_key(result: LangIdResult, passage_lang: str | None = None) -> str:
    """Evaluation slice key: query type crossed with evidence language.

    This is what every per-language table in docs/PLAN.md §3 is grouped by, so it
    lives next to the classifier rather than in the reporting code -- the two must
    agree on the label spelling or the tables silently split into extra rows.
    """
    q = {"English": "EN", "Indic": "HI", "Code-Mixed": "Hing"}[result.query_type]
    if passage_lang is None:
        return q
    return f"{q}->{passage_lang.upper()}"
