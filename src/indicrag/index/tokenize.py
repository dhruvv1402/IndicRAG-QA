"""Script-aware tokenization for lexical retrieval.

One tokenizer cannot serve three scripts. English wants casefolding and
suffix stripping; Hindi wants neither of those rules but does want its own
postposition handling; a code-mixed query wants both, chosen per token.

The expected consequence is stated up front so that the result is read as a
measurement rather than as a broken baseline: **a Romanized Hinglish query shares
almost no tokens with a Devanagari passage.** BM25 across that pair scores near
zero except where English loanwords (`scholarship`, `income`, `NMMS`) survive in
both. That is not a bug to be fixed here. It is the finding that motivates dense
retrieval, and hypothesis H4 predicts it.
"""

from __future__ import annotations

import re

from ..corpus.normalize import normalize_text

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[ऀ-ॿ]+")

ENGLISH_STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "than", "of", "for", "to", "in", "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "this", "that", "these", "those", "it", "its", "he", "she", "they", "them", "their", "there", "here", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "not", "no", "nor", "so", "such", "own", "same", "too", "very", "s", "t", "just", "don", "now", "under", "over", "again", "further", "once"]
)

HINDI_STOPWORDS = frozenset(
    ["के", "का", "की", "को", "में", "से", "है", "हैं", "था", "थे", "थी", "और", "या", "पर", "यह", "वह", "ये", "वे", "एक", "लिए", "किया", "कर", "करने", "हो", "होता", "होती", "होने", "गया", "गई", "इस", "उस", "जो", "कि", "तो", "ही", "नहीं", "भी", "तक", "साथ", "बाद", "पहले", "अपने", "सकता", "सकते", "सभी", "कोई", "कुछ", "जब", "तब", "यदि", "अगर", "द्वारा", "रूप", "प्रकार", "आदि", "अथवा", "एवं", "तथा", "हुए", "हुई", "जाता", "जाती", "जाते", "किसी", "वाले", "वाली", "वाला", "रहा", "रहे", "रही", "बहुत", "अधिक", "कम", "लेकिन", "अपना", "अपनी"]
)

#: Suffixes stripped from Hindi tokens. This is a *light* stripper covering the
#: common case markers and plural forms, not a morphological analyser -- a proper
#: one is out of scope (docs/ARCHITECTURE.md §21), and its absence costs some
#: lexical recall on Hindi queries, which should be remembered when reading the
#: BM25-on-Hindi number. Ordered longest-first so the greedy match is correct.
_HINDI_SUFFIXES = (
    "ाओं", "ियों", "ाएँ", "ाएं", "ओं", "ुओं", "ियाँ", "ियां",
    "ाई", "ाओ", "ाएii", "ेंगे", "ेगा", "ेगी",
    "ों", "ें", "ाँ", "ां", "ीय", "ता", "ते", "ती", "ना", "ने", "नी",
    "ा", "ि", "ी", "ो", "े", "ू", "ु",
)
_MIN_STEM_LEN = 3


def _strip_hindi_suffix(token: str) -> str:
    for suf in _HINDI_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= _MIN_STEM_LEN:
            return token[: -len(suf)]
    return token


def _stem_english(token: str) -> str:
    """Very light English suffix stripping.

    Deliberately not Porter/Snowball: this runs on both sides of retrieval and on
    code-mixed input where a token may be a Romanized Hindi word that an English
    stemmer would mangle ("liye" -> "li"). Conservative rules only.
    """
    for suf in ("ations", "ation", "ments", "ment", "ness", "ing", "ies", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            if suf == "ies":
                return token[:-3] + "y"
            return token[: -len(suf)]
    return token


def is_devanagari_token(token: str) -> bool:
    return bool(token) and "ऀ" <= token[0] <= "ॿ"


def tokenize(text: str, *, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    """Tokenize mixed-script text, routing each token to its own script's rules.

    Per-token dispatch rather than per-document is what makes this work on
    code-mixed input, where a single query contains English content words and
    Devanagari or Romanized Hindi function words side by side.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalize_text(text, join_softwrap=False)):
        if is_devanagari_token(raw):
            if drop_stopwords and raw in HINDI_STOPWORDS:
                continue
            tokens.append(_strip_hindi_suffix(raw) if stem else raw)
        else:
            low = raw.lower()
            if drop_stopwords and low in ENGLISH_STOPWORDS:
                continue
            if len(low) == 1 and not low.isdigit():
                continue
            tokens.append(_stem_english(low) if stem else low)
    return tokens
