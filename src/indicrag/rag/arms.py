"""The four Module 4 generation arms, and their disk cache.

| arm | context given | what it measures |
|---|---|---|
| A | none | parametric knowledge; the hallucination baseline for H3 |
| B | top-k from the best dense encoder | standard RAG |
| C | top-k from script-aware fusion | the full system |
| D | the gold passages, retrieval bypassed | the generation ceiling |

**Arm D is the load-bearing one and must not be cut.** The brief requires
retrieval and generation errors to be attributable separately, and without an
oracle arm they are not: a wrong answer could be a retrieval miss or a generation
failure, and the aggregate cannot tell you which. With it, `D − C` is retrieval
error and `1 − D` is generation error.

Arm A is a control with an unusual property: it is deliberately *not* instructed
to abstain when unsure. An instruction to abstain would suppress exactly the
behaviour the arm exists to measure, which is what a fluent model asserts with no
evidence in front of it.

Every generation is cached to disk keyed by a hash of (prompt, model, params).
A full sweep is 400 questions times four arms times two model sizes, which is
several hours of CPU inference; without a cache, re-running the analysis would
mean re-running the models, and analysis would stop being something you iterate
on.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..models import Passage, QAItem
from .prompts import (
    REFUSAL,
    build_answer_prompt,
    build_closed_book_prompt,
)


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    uses_retrieval: bool
    uses_oracle: bool
    note: str


ARMS: dict[str, Arm] = {
    "A": Arm("A", "closed-book", False, False, "no context; measures parametric assertion"),
    "B": Arm("B", "RAG dense", True, False, "top-k from the best dense encoder"),
    "C": Arm("C", "RAG hybrid", True, False, "top-k from script-aware fusion"),
    "D": Arm("D", "oracle", False, True, "gold passages injected; the generation ceiling"),
}


@dataclass
class Generation:
    """One model answer, with everything needed to score and audit it."""

    item_id: str
    arm: str
    question: str
    answerable: bool
    answer: str
    citation: str
    confidence: float
    context_ids: list[str] = field(default_factory=list)
    model: str = ""
    latency_ms: int = 0
    prompt_hash: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Generation:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


def prompt_hash(prompt: str, model: str, params: str = "") -> str:
    return hashlib.sha256(f"{model}\x00{params}\x00{prompt}".encode()).hexdigest()[:16]


class GenerationCache:
    """JSONL-backed, keyed by prompt hash. Append-only within a run."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._by_hash: dict[str, Generation] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    gen = Generation.from_dict(json.loads(line))
                    if gen.prompt_hash:
                        self._by_hash[gen.prompt_hash] = gen

    def __len__(self) -> int:
        return len(self._by_hash)

    def get(self, key: str) -> Generation | None:
        return self._by_hash.get(key)

    def put(self, gen: Generation) -> None:
        self._by_hash[gen.prompt_hash] = gen
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(gen.as_dict(), ensure_ascii=False) + "\n")


def parse_answer(raw: str) -> dict | None:
    """Parse the answer JSON, tolerating the wrappers small models add."""
    from .providers import extract_json_object

    found = extract_json_object(raw or "")
    if found is None:
        return None
    try:
        data = json.loads(found)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def build_prompt_for(
    arm: Arm, item: QAItem, context: Sequence[Passage]
) -> tuple[str, list[str]]:
    if not arm.uses_retrieval and not arm.uses_oracle:
        return build_closed_book_prompt(item.question), []
    return build_answer_prompt(item.question, context), [p.passage_id for p in context]


def run_arm(
    arm: Arm,
    items: Sequence[QAItem],
    passages: Sequence[Passage],
    complete: Callable[[str], str],
    *,
    retrieve: Callable[[QAItem], list[str]] | None = None,
    cache: GenerationCache | None = None,
    model: str = "",
    k: int = 5,
    progress: Callable[[str], None] | None = None,
) -> list[Generation]:
    """Run one arm over the items, using the cache where possible."""
    by_id = {p.passage_id: p for p in passages}
    say = progress or (lambda _m: None)
    out: list[Generation] = []

    for n, item in enumerate(items, start=1):
        if arm.uses_oracle:
            context = [by_id[pid] for pid in item.gold_passage_ids if pid in by_id][:k]
        elif arm.uses_retrieval:
            if retrieve is None:
                raise ValueError(f"arm {arm.key} needs a retriever")
            context = [by_id[pid] for pid in retrieve(item)[:k] if pid in by_id]
        else:
            context = []

        # An oracle item with no resolvable gold passage cannot be run: it would
        # silently become a closed-book item and inflate the measured retrieval
        # error, which is exactly the quantity arm D exists to isolate.
        if arm.uses_oracle and not context:
            continue

        prompt, context_ids = build_prompt_for(arm, item, context)
        key = prompt_hash(prompt, model, arm.key)

        if cache is not None and (hit := cache.get(key)) is not None:
            out.append(hit)
            continue

        start = time.time()
        data = parse_answer(complete(prompt))
        latency = int((time.time() - start) * 1000)

        if data is None:
            gen = Generation(
                item_id=item.id, arm=arm.key, question=item.question,
                answerable=False, answer="", citation="", confidence=0.0,
                context_ids=context_ids, model=model, latency_ms=latency,
                prompt_hash=key,
            )
        else:
            answerable = bool(data.get("answerable", True))
            answer = (data.get("answer") or "").strip()
            gen = Generation(
                item_id=item.id,
                arm=arm.key,
                question=item.question,
                answerable=answerable and bool(answer),
                answer=answer if (answerable and answer) else REFUSAL,
                citation=(data.get("citation") or "").strip(),
                confidence=float(data.get("confidence") or 0.0),
                context_ids=context_ids,
                model=model,
                latency_ms=latency,
                prompt_hash=key,
            )

        if cache is not None:
            cache.put(gen)
        out.append(gen)
        if n % 25 == 0:
            say(f"  arm {arm.key}: {n}/{len(items)}")

    say(f"  arm {arm.key} ({arm.label}): {len(out)} generations")
    return out
