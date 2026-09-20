"""Synthetic retrieval probes: ground truth by construction.

**These are not the gold set.** `docs/PRD.md` §6 specifies 400 human-verified
question-answer instances, and building those is P3 in the plan. This module
builds something narrower and cheaper that is available immediately: queries whose
correct passage is known *by construction*, requiring no annotation at all.

That buys an early, repeatable signal on retrieval while the real set is being
written, and it catches gross pipeline errors -- wrong E5 prefixes, misaligned
embeddings, a broken tokenizer -- while they are still cheap to fix rather than
after 400 items have been annotated against a broken index.

What it cannot do is stand in for the gold set in the paper. Three probe shapes
are generated, and they differ sharply in how much they flatter lexical retrieval:

- `fragment`  -- content words lifted from the passage. Maximum lexical overlap,
                 so BM25 is heavily favoured. Useful as a sanity ceiling, close to
                 worthless as evidence for H1.
- `entity`    -- scheme name plus section heading, phrased as a question. Moderate
                 overlap; the passage body is not quoted.
- `crosslang` -- the query is built from the *Hindi* side of a scheme and the gold
                 passages are the *English* side, or vice versa. Lexical overlap is
                 near zero by construction, which makes this the only probe shape
                 that says anything honest about cross-lingual retrieval.

Every probe records its own question-to-passage token overlap, so the leakage
control in `docs/PRD.md` §6.6 can be applied to probe results exactly as it will
be applied to the gold set: report the headline comparison on the full set *and*
on the low-overlap subset, and if the gap collapses, say so.

Probes are emitted with `verified: false` and are excluded from any reported
metric that claims to be the gold-set result.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from ..index.tokenize import tokenize
from ..models import Passage, QAItem

#: Section headings that make good question stems, in both languages.
_QUESTION_TEMPLATES_EN = [
    "What does {scheme} say about {section}?",
    "{scheme} {section} details",
    "Explain the {section} of {scheme}.",
]
_QUESTION_TEMPLATES_HI = [
    "{scheme} में {section} क्या है?",
    "{scheme} {section} की जानकारी",
]
_QUESTION_TEMPLATES_HING = [
    "{scheme} ka {section} kya hai?",
    "{scheme} ke liye {section} kaise hai?",
    "{scheme} {section} ke baare mein bataye",
]

FRAGMENT_WORDS = 10
MIN_PASSAGE_TOKENS = 60


def token_overlap(question: str, passage_text: str) -> float:
    """Jaccard over content tokens. The leakage measure from PRD §6.6."""
    q = set(tokenize(question))
    p = set(tokenize(passage_text))
    if not q or not p:
        return 0.0
    return len(q & p) / len(q | p)


def _scheme_display(passage: Passage, titles: dict[tuple[str, str], str] | None = None) -> str:
    """The scheme's name as written in the passage's own language.

    This must not fall back to the slug for Hindi. The slug is Latin
    ("suraksha-bima"), so interpolating it into a Hindi template produced queries
    like "suraksha bima आलोचना की जानकारी" -- mostly Latin, and therefore
    classified Code-Mixed rather than Indic. That collapsed the cross-lingual
    slice from an intended ~30 probes to 8 and quietly moved the rest into the
    code-mixed group, which is a different measurement entirely.

    `titles` maps (scheme, lang) to the article title actually fetched, so a Hindi
    probe says "प्रधानमंत्री सुरक्षा बीमा योजना".
    """
    if titles:
        title = titles.get((passage.scheme, passage.lang))
        if title:
            return title
    return passage.scheme.replace("-", " ")


def _section_leaf(passage: Passage) -> str:
    leaf = (passage.section_path or "").split(" > ")[-1].strip()
    return leaf if leaf and leaf.lower() != "introduction" else ""


def build_probes(
    passages: Sequence[Passage],
    *,
    seed: int = 20260922,
    per_shape: int = 60,
    titles: dict[tuple[str, str], str] | None = None,
) -> list[QAItem]:
    """Generate probes of all three shapes over the corpus.

    `titles` maps (scheme, lang) to the source article title, so that a Hindi
    probe names its scheme in Devanagari rather than by its Latin slug. Without
    it the cross-lingual slice is mislabelled -- see `_scheme_display`.
    """
    rng = random.Random(seed)
    usable = [p for p in passages if p.token_count >= MIN_PASSAGE_TOKENS]
    by_scheme_lang: dict[tuple[str, str], list[Passage]] = defaultdict(list)
    for p in usable:
        by_scheme_lang[(p.scheme, p.lang)].append(p)

    items: list[QAItem] = []

    # --- fragment -------------------------------------------------------------
    for p in rng.sample(usable, min(per_shape, len(usable))):
        words = p.text.split()
        if len(words) < FRAGMENT_WORDS + 4:
            continue
        start = rng.randrange(0, max(1, len(words) - FRAGMENT_WORDS))
        question = " ".join(words[start : start + FRAGMENT_WORDS])
        items.append(
            _item(
                f"probe-frag-{len(items):04d}",
                question,
                p,
                gold=[p.passage_id],
                shape="fragment",
            )
        )

    # --- entity ---------------------------------------------------------------
    candidates = [p for p in usable if _section_leaf(p)]
    for p in rng.sample(candidates, min(per_shape, len(candidates))):
        section = _section_leaf(p)
        scheme = _scheme_display(p, titles)
        templates = _QUESTION_TEMPLATES_HI if p.lang == "hi" else _QUESTION_TEMPLATES_EN
        question = rng.choice(templates).format(scheme=scheme, section=section)
        # Gold is every passage in the same scheme+section: the question names a
        # section, not a sentence, so any passage from it is correct evidence.
        gold = [
            q.passage_id
            for q in by_scheme_lang[(p.scheme, p.lang)]
            if _section_leaf(q) == section
        ]
        items.append(
            _item(f"probe-ent-{len(items):04d}", question, p, gold=gold, shape="entity")
        )

    # --- crosslang ------------------------------------------------------------
    # Query built from one language's side of a scheme; gold is the other side.
    # Lexical overlap is near zero by construction, so this is the shape that
    # actually tests cross-lingual retrieval.
    schemes = sorted({p.scheme for p in usable})
    made = 0
    for scheme in rng.sample(schemes, len(schemes)):
        for q_lang, g_lang in (("hi", "en"), ("en", "hi")):
            src = by_scheme_lang.get((scheme, q_lang)) or []
            tgt = by_scheme_lang.get((scheme, g_lang)) or []
            if not src or not tgt:
                continue
            p = rng.choice(src)
            scheme_name = _scheme_display(p, titles)
            section = _section_leaf(p) or scheme_name
            # A Hindi probe whose scheme name is still Latin would classify as
            # Code-Mixed, not Indic, so skip rather than mislabel the slice.
            if q_lang == "hi" and not any("ऀ" <= c <= "ॿ" for c in scheme_name):
                continue
            if q_lang == "hi":
                question = rng.choice(_QUESTION_TEMPLATES_HI).format(
                    scheme=scheme_name, section=section
                )
            else:
                question = rng.choice(_QUESTION_TEMPLATES_HING).format(
                    scheme=scheme_name, section=section
                )
            items.append(
                _item(
                    f"probe-xl-{len(items):04d}",
                    question,
                    p,
                    gold=[q.passage_id for q in tgt],
                    shape="crosslang",
                    passage_lang=g_lang,
                )
            )
            made += 1
        if made >= per_shape:
            break

    return items


def _item(
    item_id: str,
    question: str,
    passage: Passage,
    *,
    gold: list[str],
    shape: str,
    passage_lang: str | None = None,
) -> QAItem:
    from ..query.langid import classify

    lang = classify(question)
    query_lang = {"English": "en", "Indic": "hi", "Code-Mixed": "hinglish"}[lang.query_type]
    return QAItem(
        id=item_id,
        question=question,
        query_lang=query_lang,
        script=lang.script,
        passage_lang=passage_lang or passage.lang,
        answerable=True,
        gold_passage_ids=gold,
        scheme=passage.scheme,
        difficulty_tags=[shape],
        split="probe",
        annotator="synthetic",
        verified=False,
        notes=f"shape={shape}; overlap={token_overlap(question, passage.text):.3f}",
    )
