"""Answer generation behind one interface.

Three backends. The extractive one is not a placeholder -- it is a genuine
non-neural QA baseline and the `--no-model` fast path, and having it means the
retrieval and answerability halves of the project are testable, evaluable and
demonstrable without a 2 GB model download. Every result it produces is by
construction a span of a retrieved passage, so its Citation Support Rate is 1.0
by definition, which makes it a useful floor when reading the generative arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..corpus.normalize import split_sentences
from ..index.tokenize import tokenize
from ..models import Passage
from ..query.langid import LangIdResult


@dataclass
class Generated:
    text: str
    lang: str = ""
    confidence: float = 0.0
    explanation: str = ""
    answerable: bool = True
    citations: list[str] | None = None


class Provider(Protocol):
    name: str

    def answer(
        self, query: str, passages: Sequence[Passage], *, lang: LangIdResult
    ) -> Generated: ...


class ExtractiveProvider:
    """Select the best-supporting sentence from the top passages.

    Scoring is token overlap with the query, weighted by inverse sentence length
    so a long sentence cannot win merely by containing more words. A small bonus
    is given for sentences carrying a digit when the query asks a quantity
    question ("how much", "kitna", "कितनी"), because in this corpus the answer to
    such a question is almost always the sentence with the number in it, and
    overlap alone routinely picks the topic sentence next to it instead.
    """

    name = "extractive"

    _QUANTITY_MARKERS = {
        "how", "much", "many", "amount", "rate", "limit", "age", "income",
        "kitna", "kitni", "kitne", "kab", "year", "years",
        "कितना", "कितनी", "कितने", "राशि", "आयु", "आय", "सीमा", "वर्ष",
    }

    def answer(
        self, query: str, passages: Sequence[Passage], *, lang: LangIdResult
    ) -> Generated:
        if not passages:
            return Generated(text="", confidence=0.0, answerable=False)

        q_tokens = set(tokenize(query))
        wants_number = bool(q_tokens & {t.lower() for t in self._QUANTITY_MARKERS}) or any(
            m in query for m in ("कितन", "kitn", "how much", "how many")
        )

        best: tuple[float, str, Passage] | None = None
        for rank, p in enumerate(passages):
            # Passages further down the ranking start from a lower base, so a
            # marginally better sentence in a much worse passage does not win.
            rank_decay = 1.0 / (1.0 + 0.35 * rank)
            for sent in split_sentences(p.text):
                s_tokens = tokenize(sent)
                if not s_tokens:
                    continue
                overlap = len(q_tokens & set(s_tokens))
                if not overlap:
                    continue
                score = overlap / (1.0 + 0.02 * len(s_tokens))
                if wants_number and any(c.isdigit() for c in sent):
                    score *= 1.6
                score *= rank_decay
                if best is None or score > best[0]:
                    best = (score, sent.strip(), p)

        if best is None:
            top = passages[0]
            sents = split_sentences(top.text)
            return Generated(
                text=sents[0] if sents else top.text[:300],
                lang=top.lang,
                confidence=0.15,
                explanation="No query term matched; returning the lead sentence of the top passage.",
                citations=[top.passage_id],
            )

        score, sentence, passage = best
        confidence = min(0.95, 0.35 + 0.12 * score)
        return Generated(
            text=sentence,
            lang=passage.lang,
            confidence=confidence,
            explanation=(
                f"Extracted verbatim from {passage.passage_id} "
                f"({passage.scheme}, {passage.section_path or 'lead'})."
            ),
            citations=[passage.passage_id],
        )


class LlamaCppProvider:
    """Local GGUF generation via llama-cpp-python.

    Not exercised yet -- the model is an optional extra (`uv sync --extra llm`) and
    docs/PLAN.md schedules it for P4. The class exists now so the interface it must
    satisfy is fixed by the extractive provider rather than retrofitted to it.
    """

    name = "llamacpp"

    def __init__(self, gguf_path: str, n_ctx: int = 4096, n_threads: int = 4, seed: int = 0):
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, seed=seed, verbose=False
        )

    def answer(
        self, query: str, passages: Sequence[Passage], *, lang: LangIdResult
    ) -> Generated:  # pragma: no cover -- requires a model file
        raise NotImplementedError("scheduled for P4; see docs/PLAN.md §1.5")
