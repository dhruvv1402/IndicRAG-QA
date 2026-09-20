"""Scaffold the 80 UNANSWERABLE items.

These are **not** model-generated, and that is deliberate rather than a
consequence of the model being unavailable. `docs/PRD.md` §6.3 requires the
near-miss class be authored while looking at the passage it is designed to nearly
match, and a language model asked for "a question this passage cannot answer"
reliably produces the trivial out-of-scope kind instead -- which proves nothing,
because any threshold rejects a question about train tickets over a corpus of
scholarship schemes. The interesting failure is the near-miss: topic present,
scheme present, specific fact absent. Retrieval returns a confident, highly
similar passage, the generator has plausible-looking context, and only a properly
calibrated answerability check abstains.

What this module produces is scaffolding, not finished items: each one is a real
scheme name dropped into a class-appropriate stem, paired (for near-miss and
false-premise) with the **distractor passage** it is designed to nearly match.
The annotator's job in `dataset verify` is to confirm the fact really is absent
from that passage and sharpen the wording. Everything comes out `verified: false`.

`distractor_passage_ids` is the field that makes near-miss checkable. Without it
the annotator would have to search the corpus to confirm absence, which is slow
enough that it would not get done, and an unverified "unanswerable" item that is
actually answerable poisons the answerability metric in the direction that looks
like success.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..models import Passage, QAItem
from .split import UNANSWERABLE_TARGET

#: Stems per class, as (stem, query_lang). `{scheme}` is filled with a real
#: scheme name written in the stem's own language.
#:
#: The lists are **interleaved en / hi / hinglish** rather than grouped, and that
#: is load-bearing. Stems are consumed cyclically to fill a class quota, so the
#: language mix of the output is exactly the language mix of the list. A first
#: version grouped English stems first and produced 55% English items against the
#: roughly 36/33/31 split PRD §6.3 calls for, which would have made the
#: per-query-type abstention rates in Module 5 incomparable with the answerable
#: set. Interleaving in threes makes any quota come out near-balanced.
STEMS: dict[str, list[tuple[str, str]]] = {
    "out-of-scope": [
        ("What is the current home loan interest rate in India?", "en"),
        ("आज सोने का भाव क्या है?", "hi"),
        ("Train ka ticket online kaise book karte hain?", "hinglish"),
        ("Who is the captain of the Indian cricket team?", "en"),
        ("दिल्ली से मुंबई की उड़ान कितने घंटे की है?", "hi"),
        ("Mobile recharge ka best plan kaun sa hai?", "hinglish"),
        ("What is the petrol price in Delhi today?", "en"),
        ("आयकर की दरें क्या हैं?", "hi"),
        ("Driving licence ke liye kahan apply karein?", "hinglish"),
    ],
    # Topic and scheme are in the corpus; this specific fact is not.
    "near-miss": [
        ("What is the exact application deadline for {scheme} in 2026?", "en"),
        ("{scheme} के लिए आवेदन की अंतिम तिथि क्या है?", "hi"),
        ("{scheme} ka helpline number kya hai?", "hinglish"),
        ("How many applications under {scheme} were rejected last year?", "en"),
        ("{scheme} में कितने आवेदन अस्वीकृत हुए?", "hi"),
        ("{scheme} ke liye application fee kitni hai?", "hinglish"),
        ("What is the toll-free helpline number for {scheme}?", "en"),
        ("{scheme} का बजट इस वर्ष कितना है?", "hi"),
        ("{scheme} ka portal kis din band rehta hai?", "hinglish"),
    ],
    # Presupposes a benefit the scheme does not provide.
    "false-premise": [
        ("How much laptop allowance does {scheme} provide?", "en"),
        ("{scheme} में विदेश यात्रा भत्ता कितना है?", "hi"),
        ("{scheme} ke tahat free laptop kab milta hai?", "hinglish"),
        ("What is the foreign travel grant under {scheme}?", "en"),
        ("{scheme} के अंतर्गत निःशुल्क छात्रावास कितने माह का है?", "hi"),
        ("{scheme} mein smartphone subsidy kitni milti hai?", "hinglish"),
    ],
    # Cannot be answered without naming a scheme; schemes differ.
    "under-specified": [
        ("What is the income limit?", "en"),
        ("आवेदन की अंतिम तिथि क्या है?", "hi"),
        ("Eligibility kya hai aur kitna paisa milega?", "hinglish"),
    ],
}

#: Sections whose passages make good distractors: they discuss eligibility,
#: benefits and process, so a near-miss question about a *missing* detail of one
#: of those is maximally confusable.
_GOOD_DISTRACTOR_HINTS = (
    "eligib", "benefit", "feature", "criteria", "implement", "objective",
    "पात्रता", "लाभ", "उद्देश्य", "विशेषता",
)


def _scheme_names(
    passages: Sequence[Passage], titles: dict[tuple[str, str], str] | None
) -> dict[str, dict[str, str]]:
    """scheme -> {lang: display name}. Falls back to the slug only for English."""
    out: dict[str, dict[str, str]] = {}
    for p in passages:
        name = (titles or {}).get((p.scheme, p.lang))
        if not name and p.lang == "en":
            name = p.scheme.replace("-", " ").title()
        if name:
            out.setdefault(p.scheme, {})[p.lang] = name
    return out


def _distractor(
    passages: Sequence[Passage], scheme: str, lang: str, rng: random.Random
) -> str | None:
    pool = [p for p in passages if p.scheme == scheme and p.lang == lang]
    if not pool:
        pool = [p for p in passages if p.scheme == scheme]
    if not pool:
        return None
    preferred = [
        p
        for p in pool
        if any(h in (p.section_path or "").lower() for h in _GOOD_DISTRACTOR_HINTS)
    ]
    return rng.choice(preferred or pool).passage_id


def build_unanswerable(
    passages: Sequence[Passage],
    *,
    titles: dict[tuple[str, str], str] | None = None,
    targets: dict[str, int] | None = None,
    seed: int = 20260922,
) -> list[QAItem]:
    """Instantiate the four classes to their PRD §6.3 quotas."""
    rng = random.Random(seed)
    targets = targets or UNANSWERABLE_TARGET
    names = _scheme_names(passages, titles)
    schemes = sorted(names)
    rng.shuffle(schemes)

    items: list[QAItem] = []
    for klass, quota in targets.items():
        stems = STEMS[klass]
        made = 0
        cursor = 0
        while made < quota:
            stem, query_lang = stems[made % len(stems)]

            if "{scheme}" not in stem:
                question = stem
                scheme = ""
                distractor = None
            else:
                # Cycle through schemes so no single one carries a whole class.
                scheme = schemes[cursor % len(schemes)]
                cursor += 1
                name_lang = "hi" if query_lang == "hi" else "en"
                name = names[scheme].get(name_lang) or names[scheme].get("en")
                if not name:
                    continue
                question = stem.format(scheme=name)
                distractor = _distractor(passages, scheme, name_lang, rng)

            items.append(
                QAItem(
                    id=f"un-{klass}-{made:03d}",
                    question=question,
                    query_lang=query_lang,
                    script="deva" if query_lang == "hi" else "latin",
                    passage_lang="",
                    answerable=False,
                    unanswerable_class=klass,
                    answer_gold="",
                    gold_passage_ids=[],
                    scheme=scheme,
                    difficulty_tags=[klass],
                    annotator="",
                    verified=False,
                    notes=(
                        "scaffold; confirm the fact is absent"
                        + (f"; distractor={distractor}" if distractor else "")
                    ),
                )
            )
            made += 1

    return items
