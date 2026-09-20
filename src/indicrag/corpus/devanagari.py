"""Devanagari integrity validation for extracted text.

This is the highest-risk stage in the pipeline and the one most likely to fail
silently. A Hindi PDF using a legacy or subset-embedded font can extract into a
string that *renders* as plausible Devanagari but is character-level nonsense:
dependent vowel signs reordered ahead of their consonant, conjuncts broken at the
virama, or glyphs mapped into the Latin private-use area. None of this raises an
exception. It produces text that anyone who does not read Devanagari will accept,
and once it is in the index, retrieval quality collapses in a way that looks like
a model problem for days before anyone suspects the corpus.

The five checks below are deliberately cheap and independent. They catch
*systematic* corruption. They do not catch subtle loss, which is why
docs/PLAN.md task 1.5 also requires a human to read 30 sampled passages before
the corpus is frozen. No amount of validation code substitutes for reading the
text once.

Reference for the syllable model: Unicode Standard, ch. 12.1 (Devanagari).
A well-formed orthographic syllable is

    consonant [nukta] (virama consonant [nukta])* [matra] [anusvara|visarga]

so a dependent vowel sign must be preceded by a consonant or a nukta. A matra in
any other position is the signature of the reordering bug.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --- Devanagari codepoint classes (U+0900-U+097F) ------------------------------

DEVANAGARI_RANGE = (0x0900, 0x097F)

CANDRABINDU = 0x0901
ANUSVARA = 0x0902
VISARGA = 0x0903
NUKTA = 0x093C
VIRAMA = 0x094D
DANDA = 0x0964
DOUBLE_DANDA = 0x0965

# Independent vowels: अ-औ
VOWELS = set(range(0x0905, 0x0915))
# Consonants: क-ह, plus the nukta-composed forms क़-य़
CONSONANTS = set(range(0x0915, 0x093A)) | set(range(0x0958, 0x0960))
# Dependent vowel signs (matras): ऺ-ौ. This is the class whose misplacement
# is the classic extraction bug.
MATRAS = set(range(0x093A, 0x094D))
DIGITS = set(range(0x0966, 0x0970))

# Closed-class Hindi function words. These survive almost any real Hindi text and
# are absent from mojibake that happens to sit in the Devanagari block.
HINDI_STOPWORDS = frozenset(
    ["के", "का", "की", "को", "में", "से", "है", "हैं", "था", "थे", "थी", "और", "या", "पर", "यह", "वह", "ये", "वे", "एक", "लिए", "किया", "कर", "करने", "हो", "होता", "होती", "होने", "गया", "गई", "इस", "उस", "जो", "कि", "तो", "ही", "नहीं", "भी", "तक", "साथ", "बाद", "पहले", "अपने", "सकता", "सकते", "सभी", "कोई", "कुछ", "जब", "तब", "यदि", "अगर", "द्वारा", "रूप", "प्रकार", "आदि", "अथवा", "एवं", "तथा", "हुए", "हुई"]
)

_WORD_RE = re.compile(r"[ऀ-ॿ]+")

#: Zero-width joiner / non-joiner / BOM. Stripped before validation, not just
#: before indexing -- see `validate` for why that distinction matters.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def _is_devanagari(cp: int) -> bool:
    return DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]


# --- Report --------------------------------------------------------------------


@dataclass
class IntegrityReport:
    """Outcome of validating one extracted document or passage.

    `ok` is the gate. `failures` names which checks tripped, so a caller can
    route (for example) a ratio failure to OCR while treating a stopword
    failure as "this is Devanagari but not Hindi", which is a different problem.
    """

    n_chars: int
    devanagari_ratio: float
    misplaced_matra_rate: float
    final_virama_rate: float
    stopword_rate: float
    replacement_chars: int
    failures: list[str] = field(default_factory=list)
    samples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_chars": self.n_chars,
            "devanagari_ratio": round(self.devanagari_ratio, 4),
            "misplaced_matra_rate": round(self.misplaced_matra_rate, 5),
            "final_virama_rate": round(self.final_virama_rate, 5),
            "stopword_rate": round(self.stopword_rate, 4),
            "replacement_chars": self.replacement_chars,
            "failures": list(self.failures),
            "samples": {k: v[:5] for k, v in self.samples.items()},
        }

    def summary(self) -> str:
        verdict = "OK" if self.ok else "FAIL(" + ",".join(self.failures) + ")"
        return (
            f"{verdict} chars={self.n_chars} deva={self.devanagari_ratio:.2f} "
            f"matra_err={self.misplaced_matra_rate:.4f} "
            f"final_virama={self.final_virama_rate:.4f} stop={self.stopword_rate:.3f}"
        )


# --- Individual checks ---------------------------------------------------------


def devanagari_ratio(text: str) -> float:
    """Share of non-space characters that sit in the Devanagari block.

    A Hindi document that extracts with a font mapped into the Latin
    private-use area scores near zero here while still *looking* like Hindi in a
    PDF viewer, which is why this is checked on the extracted string rather than
    trusted from the source.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    deva = sum(1 for c in chars if _is_devanagari(ord(c)))
    return deva / len(chars)


def find_misplaced_matras(text: str) -> list[tuple[int, str]]:
    """Locate dependent vowel signs not preceded by a consonant or nukta.

    This is the reordering bug. In correctly extracted text the count is zero;
    a handful can legitimately appear in malformed source, so callers compare a
    *rate* against a tolerance rather than demanding zero.
    """
    hits: list[tuple[int, str]] = []
    for i, ch in enumerate(text):
        if ord(ch) not in MATRAS:
            continue
        prev = ord(text[i - 1]) if i > 0 else None
        if prev is None or not (prev in CONSONANTS or prev == NUKTA or prev in MATRAS):
            start = max(0, i - 6)
            hits.append((i, text[start : i + 6]))
    return hits


def find_final_viramas(text: str) -> list[str]:
    """Words ending in a bare virama.

    A virama joins two consonants into a conjunct. Word-final it means the second
    half of the conjunct was dropped during extraction -- the halant is left
    dangling. Hindi does write a few genuine final viramas, so again this is a
    rate, not a hard zero.
    """
    return [w for w in _WORD_RE.findall(text) if w and ord(w[-1]) == VIRAMA]


def stopword_rate(text: str) -> float:
    """Share of Devanagari word tokens that are common Hindi function words.

    Mojibake that lands inside the Devanagari block still produces "words", but
    they are not Hindi words. Real Hindi prose runs roughly 0.15-0.35 here.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(1 for w in words if w in HINDI_STOPWORDS) / len(words)


# --- Composite validator -------------------------------------------------------

# Tolerances. These are floors and ceilings on *rates*, chosen to pass clean
# government Hindi and fail the corruption modes above. They are deliberately
# loose: a false alarm costs an OCR pass, a miss costs the whole corpus.
MIN_DEVANAGARI_RATIO = 0.45
MAX_MISPLACED_MATRA_RATE = 0.005
MAX_FINAL_VIRAMA_RATE = 0.02
MIN_STOPWORD_RATE = 0.05
MIN_CHARS = 40


def validate(text: str, *, expect_hindi: bool = True) -> IntegrityReport:
    """Run every check and return a report.

    `expect_hindi=False` skips the script-ratio and stopword checks, so the same
    function can sanity-check an English document (where only the replacement-
    character check is meaningful) without reporting spurious failures.

    Zero-width characters are stripped first, and that is not cosmetic. Hindi
    source text -- government prose especially -- writes ZWNJ after a virama to
    force an explicit halant rather than a conjunct ligature, as in उद्देश्‍य
    (उद्देश + virama + ZWNJ + य). ZWNJ sits outside the Devanagari block, so it
    splits the word token and leaves the first half apparently ending in a bare
    virama. Three genuinely clean Hindi documents failed the dangling-virama
    check for exactly this reason before the strip was added.

    Stripping here also keeps the validator honest in a second way: it now checks
    the same string the indexer will see, since `normalize.normalize_text` strips
    these too. Validating text that differs from what gets indexed would mean
    passing documents that then behave differently in retrieval.
    """
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    chars = [c for c in text if not c.isspace()]
    n_chars = len(chars)

    ratio = devanagari_ratio(text)
    misplaced = find_misplaced_matras(text)
    finals = find_final_viramas(text)
    words = _WORD_RE.findall(text)
    stop = stopword_rate(text)
    replacements = text.count("�")

    n_deva = max(1, sum(1 for c in chars if _is_devanagari(ord(c))))
    matra_rate = len(misplaced) / n_deva
    virama_rate = len(finals) / max(1, len(words))

    failures: list[str] = []
    samples: dict[str, list[str]] = {}

    if n_chars < MIN_CHARS:
        failures.append("too_short")
    if replacements:
        failures.append("replacement_chars")
    if expect_hindi:
        if ratio < MIN_DEVANAGARI_RATIO:
            failures.append("low_devanagari_ratio")
        if matra_rate > MAX_MISPLACED_MATRA_RATE:
            failures.append("misplaced_matras")
            samples["misplaced_matras"] = [ctx for _, ctx in misplaced[:10]]
        if virama_rate > MAX_FINAL_VIRAMA_RATE:
            failures.append("dangling_viramas")
            samples["dangling_viramas"] = finals[:10]
        if stop < MIN_STOPWORD_RATE and len(words) >= 20:
            failures.append("low_stopword_rate")

    return IntegrityReport(
        n_chars=n_chars,
        devanagari_ratio=ratio,
        misplaced_matra_rate=matra_rate,
        final_virama_rate=virama_rate,
        stopword_rate=stop,
        replacement_chars=replacements,
        failures=failures,
        samples=samples,
    )


#: A script has to be more than incidental before it makes a string "mixed".
#: One borrowed English word inside a Hindi paragraph is still Hindi; half a
#: sentence in each is not. 0.15 is the line between those two.
MINOR_SCRIPT_CEILING = 0.15


def script_of(text: str) -> str:
    """Coarse script label for a string: 'deva', 'latin' or 'mixed'.

    A dominance test alone is not enough. "scholarship के लिए eligibility क्या है"
    is 65% Latin by character count, which a simple `> 0.6` rule labels Latin even
    though a third of it is Devanagari. A string only counts as single-script when
    the *other* script is incidental, which keeps this consistent with
    `query.langid.classify`, where any real presence of both means code-mixed.
    """
    chars = [c for c in text if c.isalnum()]
    if not chars:
        return "unknown"
    total = len(chars)
    deva = sum(1 for c in chars if _is_devanagari(ord(c))) / total
    latin = sum(1 for c in chars if c.isascii() and c.isalpha()) / total

    if deva >= 0.6 and latin < MINOR_SCRIPT_CEILING:
        return "deva"
    if latin >= 0.6 and deva < MINOR_SCRIPT_CEILING:
        return "latin"
    if deva == 0.0 and latin > 0.0:
        return "latin"
    if latin == 0.0 and deva > 0.0:
        return "deva"
    return "mixed"
