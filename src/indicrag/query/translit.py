"""Devanagari to Romanized Hindi, deterministically.

**Why this is not a model call.** Asking a 3B model to "rewrite this Hindi
question the way an Indian user would type it" failed about 83% of the time in
practice: it returned the Devanagari unchanged, or translated the question into
English instead of transliterating it. Each failure cost a full Hindi generation
(~56 s) before being discarded, which is what turned the `hinglish` cells into a
seven-minute-per-item crawl. Script conversion is a deterministic mapping, and
handing it to a stochastic model was the wrong tool.

`indic-transliteration` does the mapping exactly, instantly, and identically
every run. What it does not do is produce *natural* Hinglish: it emits scholarly
transliteration with capitals marking vowel length and the inherent schwa
retained, so मध्याह्न भोजन योजना कब शुरू हुई? becomes

    madhyAhna bhojana yojanA kaba shurU huI?

where a person would type `madhyahn bhojan yojana kab shuru hui?`. Two mechanical
rules close most of that gap.

**Schwa deletion.** Hindi drops the inherent word-final schwa: भोजन is *bhojan*,
not *bhojana*. The transliterator writes it because the character is there. A
final lowercase `a` is always an inherent schwa in this scheme, whereas `A` marks
a genuine long vowel (योजना → *yojanA* → *yojana*, which keeps its final vowel).
Deleting final `a` before casefolding preserves that distinction; doing it after
would destroy it.

**Anusvara.** `M` renders the nasal, which people type as `n`.

What survives is medial schwa deletion — *kitani* where a person writes *kitni* —
and the fact that real Hinglish keeps English loanwords in English rather than
transliterating them back. Both are left to the annotator, which
`docs/PRD.md` §6.5 step 2 already requires: Hinglish variants are hand-written or
hand-corrected, because machine-generated code-mix is not how people type. This
produces a correct, consistent starting point rather than a finished item.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z]+")

#: Scheme names and common English loanwords that should stay in English rather
#: than being transliterated back out of Devanagari. Real code-mixed text keeps
#: these in Latin script, and round-tripping them produces forms no one writes.
_KEEP_ENGLISH = {
    "yojana": "yojana",
    "skIma": "scheme",
    "akAuMTa": "account",
    "bERka": "bank",
    "kArDa": "card",
    "onalAina": "online",
    "phorma": "form",
    "vebasAiTa": "website",
    "porTala": "portal",
    "sabsiDI": "subsidy",
    "bImA": "bima",
}


def _deschwa(word: str) -> str:
    """Drop the inherent word-final schwa, keeping genuine long vowels.

    Must run before casefolding: `a` is the schwa, `A` is a long vowel, and
    lowercasing first would conflate them.
    """
    if len(word) > 2 and word.endswith("a"):
        return word[:-1]
    return word


def to_roman(text: str, *, naturalize: bool = True) -> str:
    """Transliterate Devanagari to Latin script.

    With `naturalize`, applies the schwa and anusvara rules above to approximate
    how Hindi is actually typed in Roman script. Non-Devanagari characters pass
    through untouched, so a mixed-script input keeps its English words as they
    are -- which is what makes this usable on code-mixed text directly.
    """
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    out = transliterate(text, sanscript.DEVANAGARI, sanscript.OPTITRANS)
    if not naturalize:
        return out

    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        if word in _KEEP_ENGLISH:
            return _KEEP_ENGLISH[word]
        return _deschwa(word).replace("M", "n")

    out = _WORD_RE.sub(fix, out)
    return out.lower()


def has_devanagari(text: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in text)


def romanize_question(question: str) -> str | None:
    """Romanize a Hindi question for the code-mixed slice.

    Returns None when the input has no Devanagari to convert, which means the
    caller handed us something already Latin and the item should be rejected
    rather than silently duplicated into the code-mixed cell.
    """
    if not has_devanagari(question):
        return None
    roman = to_roman(question)
    return roman if roman.strip() else None
