"""Answer scoring: Exact Match, token-level F1, and semantic similarity.

**A normalizer written for English is wrong for Hindi, and silently so.** The
SQuAD normalizer everyone reaches for strips `a`, `an` and `the`, lowercases, and
drops ASCII punctuation. Applied to Devanagari it does nothing useful: Hindi has
no articles to strip, casefolding is a no-op, and the sentence terminator it
needs to remove is `।`, which is not ASCII punctuation. The result is not a
crash. It is an Exact Match score for Hindi that is a few points too low for
reasons no one notices, on every table in the paper.

So normalization dispatches on script, and three corpus-specific equivalences are
folded in because they are the commonest answer types here:

- **Digits.** `३,५०,०००` and `3,50,000` are the same answer.
- **Amounts.** `Rs. 3,50,000`, `₹3.5 lakh` and `350000/-` are the same answer.
  This is the one that matters most: an amount is the answer to a large share of
  the questions in this dataset, and the surface form varies between the English
  and Hindi versions of the same document.
- **Postposition spacing.** Hindi writes `के लिए` and `केलिए` interchangeably in
  extracted text, and a token-level F1 that treats them as different tokens
  punishes a correct answer.

`token_f1` is computed over multisets, not sets, following SQuAD: a repeated
token in the prediction should only match as many times as it appears in the
gold, or padding an answer with a repeated word would inflate recall.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..corpus.normalize import canonical_amounts, normalize_digits, normalize_unicode

_ENGLISH_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\sऀ-ॿ]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

#: Hindi postpositions that extracted text writes both joined and separated.
_POSTPOSITIONS = ("के लिए", "के साथ", "के बाद", "के अनुसार", "के अंतर्गत", "के द्वारा")


def normalize_answer(text: str) -> str:
    """Script-aware normalization for Exact Match and token F1."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = normalize_digits(text)

    # Collapse the joined/separated postposition variants before tokenizing.
    for phrase in _POSTPOSITIONS:
        text = text.replace(phrase.replace(" ", ""), phrase)

    text = text.replace("।", " ").replace("॥", " ")  # danda, double danda
    text = _PUNCT_RE.sub(" ", text)
    text = text.lower()
    tokens = [t for t in _WS_RE.split(text) if t and t not in _ENGLISH_ARTICLES]
    return " ".join(tokens)


def _amount_key(text: str) -> str | None:
    """Canonical integer for an answer that is purely an amount, else None.

    Only applied when the answer is *essentially* the amount. `Rs. 3,50,000` and
    `3.5 lakh` should match; "income up to Rs 3,50,000 for rural applicants" and
    "3.5 lakh" should not, because the first carries a qualification the second
    does not, and collapsing them would score an incomplete answer as perfect.
    """
    amounts = canonical_amounts(text)
    if len(amounts) != 1:
        return None
    stripped = normalize_answer(text)
    # Allow the amount plus a few unit/currency words, nothing more.
    if len(stripped.split()) <= 4:
        return amounts[0]
    return None


def exact_match(prediction: str, gold: str) -> bool:
    """Normalized string equality, with amount-form equivalence."""
    if normalize_answer(prediction) == normalize_answer(gold):
        return True
    p_amount, g_amount = _amount_key(prediction), _amount_key(gold)
    return p_amount is not None and p_amount == g_amount


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style token F1 over multisets."""
    p_tokens = normalize_answer(prediction).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens or not g_tokens:
        # Two empty answers agree; one empty and one not does not.
        return float(p_tokens == g_tokens)

    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(fn, prediction: str, golds: Sequence[str]) -> float:
    """Score against the best of several acceptable gold answers.

    Items carry both `answer_gold` and `answer_gold_hi`, and a correct answer in
    either language is correct. Scoring only against one would penalise the
    system for answering in the language it was asked in.
    """
    candidates = [g for g in golds if g]
    if not candidates:
        return 0.0
    return max(float(fn(prediction, g)) for g in candidates)


# --- report --------------------------------------------------------------------


@dataclass
class QAOutcome:
    item_id: str
    slice_key: str
    language_group: str
    prediction: str
    golds: list[str]
    abstained: bool = False

    @property
    def em(self) -> float:
        return best_over_golds(exact_match, self.prediction, self.golds)

    @property
    def f1(self) -> float:
        return best_over_golds(token_f1, self.prediction, self.golds)


@dataclass
class QAReport:
    system: str
    outcomes: list[QAOutcome] = field(default_factory=list)

    def _sel(self, slice_key: str | None) -> list[QAOutcome]:
        if slice_key in (None, "all"):
            return self.outcomes
        return [
            o for o in self.outcomes if slice_key in (o.slice_key, o.language_group)
        ]

    def n(self, slice_key: str | None = None) -> int:
        return len(self._sel(slice_key))

    def _mean(self, fn, slice_key: str | None) -> float:
        sel = self._sel(slice_key)
        return sum(fn(o) for o in sel) / len(sel) if sel else 0.0

    def exact_match(self, slice_key: str | None = None) -> float:
        return self._mean(lambda o: o.em, slice_key)

    def token_f1(self, slice_key: str | None = None) -> float:
        return self._mean(lambda o: o.f1, slice_key)

    def abstention_rate(self, slice_key: str | None = None) -> float:
        return self._mean(lambda o: float(o.abstained), slice_key)

    # Citation Support Rate deliberately lives on GroundingReport, not here.
    # A second copy used to sit on this class reading `QAOutcome.supported`, a
    # field `score_arm` never set -- so it returned 0.0 for every slice, every
    # run, while looking like the metric it shares a name with. The live one
    # reports both the lexical and the entailment variant and is what
    # `format_module4` prints.
