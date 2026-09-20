"""Retrieval metrics, sliced by language pair.

Report objects expose named metric methods and formatters return `list[str]`, so
the console path and the `--report <path>` path share one code path and what gets
committed under `evals/` is byte-identical to what was printed.

Every metric takes a `slice` key, which is how the per-language breakdown that
the brief requires (§6, "Language Analysis") comes out of one pass rather than
one pass per table.

A note on `gold_passage_ids` being a list. When a fact appears in both the English
and the Hindi version of a scheme, either passage is correct evidence, and a hit
on either counts. Recall is therefore "did we find *any* gold passage", not "did
we find *the* gold passage" -- scoring the latter would penalise a cross-lingual
retriever for the very behaviour being measured.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models import QAItem


@dataclass
class Outcome:
    """What one query produced, with everything the metrics need."""

    item_id: str
    slice_key: str
    language_group: str
    gold: list[str]
    retrieved: list[str]
    scores: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def hit_at(self, k: int) -> bool:
        return any(p in self.gold for p in self.retrieved[:k])

    def n_hits_at(self, k: int) -> int:
        return sum(1 for p in self.retrieved[:k] if p in self.gold)

    def recall_at(self, k: int) -> float:
        if not self.gold:
            return 0.0
        return self.n_hits_at(k) / len(self.gold)

    def precision_at(self, k: int) -> float:
        if k == 0:
            return 0.0
        return self.n_hits_at(k) / k

    def reciprocal_rank(self) -> float:
        for i, p in enumerate(self.retrieved, start=1):
            if p in self.gold:
                return 1.0 / i
        return 0.0

    def ndcg_at(self, k: int) -> float:
        dcg = sum(
            1.0 / math.log2(i + 1)
            for i, p in enumerate(self.retrieved[:k], start=1)
            if p in self.gold
        )
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(self.gold), k) + 1))
        return dcg / ideal if ideal else 0.0


@dataclass
class RetrievalReport:
    system: str
    outcomes: list[Outcome]

    def _sel(self, slice_key: str | None) -> list[Outcome]:
        if slice_key in (None, "all"):
            return self.outcomes
        return [
            o
            for o in self.outcomes
            if o.slice_key == slice_key or o.language_group == slice_key
        ]

    def _mean(self, fn, slice_key: str | None) -> float:
        sel = self._sel(slice_key)
        return sum(fn(o) for o in sel) / len(sel) if sel else 0.0

    def n(self, slice_key: str | None = None) -> int:
        return len(self._sel(slice_key))

    def recall_at(self, k: int, slice_key: str | None = None) -> float:
        return self._mean(lambda o: o.recall_at(k), slice_key)

    def precision_at(self, k: int, slice_key: str | None = None) -> float:
        return self._mean(lambda o: o.precision_at(k), slice_key)

    def hit_rate_at(self, k: int, slice_key: str | None = None) -> float:
        return self._mean(lambda o: 1.0 if o.hit_at(k) else 0.0, slice_key)

    def mrr(self, slice_key: str | None = None) -> float:
        return self._mean(lambda o: o.reciprocal_rank(), slice_key)

    def ndcg_at(self, k: int, slice_key: str | None = None) -> float:
        return self._mean(lambda o: o.ndcg_at(k), slice_key)

    def seconds_per_query(self) -> float:
        return self._mean(lambda o: o.seconds, None)

    def misses(self, k: int = 5, limit: int = 20) -> list[Outcome]:
        return [o for o in self.outcomes if not o.hit_at(k)][:limit]

    def slices(self) -> list[str]:
        seen: list[str] = []
        for o in self.outcomes:
            if o.slice_key not in seen:
                seen.append(o.slice_key)
        return seen


def evaluate(
    system: str,
    items: Sequence[QAItem],
    run,  # Callable[[QAItem], tuple[list[Retrieved], float]]
) -> RetrievalReport:
    """Run one retrieval system over the answerable items and collect outcomes.

    Unanswerable items are excluded here by construction: they have no gold
    passage, so every retrieval metric on them is undefined. They are the subject
    of the answerability evaluation instead.
    """
    outcomes: list[Outcome] = []
    for item in items:
        if not item.answerable or not item.gold_passage_ids:
            continue
        hits, seconds = run(item)
        outcomes.append(
            Outcome(
                item_id=item.id,
                slice_key=item.slice_key,
                language_group=item.language_group,
                gold=list(item.gold_passage_ids),
                retrieved=[h.passage_id for h in hits],
                scores=[h.score for h in hits],
                seconds=seconds,
            )
        )
    return RetrievalReport(system=system, outcomes=outcomes)

