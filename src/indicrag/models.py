"""Core data models and tolerant JSONL serialization.

Plain dataclasses, no ORM, no database. The corpus is under ~1500 passages and
the gold set under 400 items: files are easier to inspect, diff, grep and
hand-edit than a database, and hand-editing is exactly what the annotation phase
does.

Deserialization is deliberately **tolerant** -- unknown keys are dropped rather
than raising, and blank lines in JSONL are skipped. This exists for one concrete
reason: the gold set is hand-edited while annotation is in progress, and a schema
addition mid-project must not invalidate the file the annotator is halfway
through. The cost is that a typo'd field name is silently ignored, which is why
`QAItem.unknown_fields` records what was dropped instead of discarding it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def _coerce(cls: type[T], raw: dict[str, Any]) -> T:
    """Build a dataclass from a dict, keeping only fields it declares.

    Anything left over is stashed under `unknown_fields` when the target
    declares it, so a mistyped key in a hand-edited record is recoverable rather
    than silently gone.
    """
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    kept = {k: v for k, v in raw.items() if k in known}
    extra = {k: v for k, v in raw.items() if k not in known}
    if extra and "unknown_fields" in known:
        kept["unknown_fields"] = extra
    return cls(**kept)  # type: ignore[call-arg]


# --- Corpus --------------------------------------------------------------------


@dataclass
class Document:
    """One source document, with the provenance needed to cite and re-fetch it.

    `sha256` and `retrieved_at` are not bookkeeping. Government circulars are
    revised without changing their URL, so the paper has to be able to state
    which snapshot produced its numbers.
    """

    doc_id: str
    scheme: str
    lang: str  # "en" | "hi"
    title: str = ""
    ministry: str = ""
    source_url: str = ""
    retrieved_at: str = ""
    sha256: str = ""
    licence: str = ""
    format: str = ""  # "pdf" | "html" | "txt"
    pages: int = 0
    ocr_used: bool = False
    integrity: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    unknown_fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Document:
        return _coerce(cls, raw)


@dataclass
class Passage:
    """A retrievable chunk of a document.

    `text` is normalized (for matching); `text_raw` is meant to be what the
    document actually said (for display and citation). They travel together
    everywhere -- showing a user a digit-normalized amount when the circular
    wrote it differently would be a small lie in a system whose whole point is
    evidence fidelity.

    That is the design, and `segment.py` does not implement it: it splits
    sentences *after* normalizing the section, so both fields are cut from the
    normalized stream, and its `char_span` cursor advances by the normalized
    length through raw source text, so the offsets drift monotonically. As
    committed, 691 of 694 passages had a `text_raw` that normalization left
    unchanged and only a quarter of the spans named a slice containing their own
    passage. Nothing reads either field, which is why neither failed loudly.

    Both are now repaired in place by `corpus repair-spans`, which aligns each
    passage back to its source rather than re-segmenting -- `passage_id` and
    `text` do not move, so the frozen corpus stays frozen. `corpus audit` checks
    both, and checks whether a span is *present* separately from whether it is
    *right*: 23 passages the aligner could not place carry `(0, 0)` rather than a
    drifted guess. Segmentation itself still needs the source-offset mapping so
    that new corpora are correct at the point of cutting; that waits on the
    re-anchoring, because it re-cuts every passage.

    `passage_id` must be stable across re-segmentation runs: the gold set, the
    embedding caches and the error analysis all reference it by name.
    """

    passage_id: str
    doc_id: str
    scheme: str
    lang: str
    text: str
    text_raw: str = ""
    section_path: str = ""
    page: int = 0
    char_span: tuple[int, int] = (0, 0)
    token_count: int = 0
    unknown_fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["char_span"] = list(self.char_span)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Passage:
        raw = dict(raw)
        if "char_span" in raw and isinstance(raw["char_span"], list):
            raw["char_span"] = tuple(raw["char_span"])
        return _coerce(cls, raw)

    @property
    def citation(self) -> str:
        parts = [self.scheme, self.section_path or f"p.{self.page}"]
        return " > ".join(p for p in parts if p)


# --- Evaluation ----------------------------------------------------------------


@dataclass
class QAItem:
    """One question-answer instance. Schema fixed in docs/PRD.md §6.4.

    `gold_passage_ids` is a list, not a scalar, and that is load-bearing. When the
    same fact appears in both the English and the Hindi version of a document,
    *both* count as correct evidence; scoring recall against only one would
    penalise a retriever for picking the other, and would materially distort every
    cross-lingual number.
    """

    id: str
    question: str
    query_lang: str  # "en" | "hi" | "hinglish"
    answerable: bool
    gold_passage_ids: list[str] = field(default_factory=list)
    script: str = ""
    passage_lang: str = ""
    unanswerable_class: str | None = None
    answer_gold: str = ""
    answer_gold_hi: str = ""
    scheme: str = ""
    difficulty_tags: list[str] = field(default_factory=list)
    split: str = ""  # "dev" | "test"
    annotator: str = ""
    second_pass: bool = False
    verified: bool = False
    notes: str = ""
    unknown_fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> QAItem:
        return _coerce(cls, raw)

    @property
    def slice_key(self) -> str:
        """Language-pair label used to group every per-language results table."""
        q = {"en": "EN", "hi": "HI", "hinglish": "Hing"}.get(self.query_lang, "?")
        if not self.answerable:
            return f"{q}->UNANS"
        return f"{q}->{(self.passage_lang or '?').upper()}"

    @property
    def language_group(self) -> str:
        """Monolingual / cross-lingual / code-mixed, the three Module 2 groups."""
        if self.query_lang == "hinglish":
            return "code-mixed"
        if not self.passage_lang:
            return "unknown"
        return "monolingual" if self.query_lang == self.passage_lang else "cross-lingual"


@dataclass
class Retrieved:
    """One retrieved passage with the scores that produced its rank."""

    passage_id: str
    score: float
    rank: int
    method: str = ""
    component_scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Retrieved:
        return _coerce(cls, raw)


@dataclass
class Answer:
    """The per-query response object. Contract fixed in docs/PRD.md §9."""

    query: str
    query_type: str  # "English" | "Indic" | "Code-Mixed"
    answerability: str  # "ANSWERABLE" | "UNANSWERABLE"
    answer: str
    confidence: float = 0.0
    query_lang: str = ""
    answer_lang: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    retrieval: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    model: str = ""
    prompt_hash: str = ""

    #: The exact string the brief requires when evidence is insufficient.
    #: Referenced rather than retyped so it cannot drift; there is a test on it.
    REFUSAL = "Insufficient information available in the provided documents."

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Answer:
        return _coerce(cls, raw)

    @classmethod
    def refusal(cls, query: str, query_type: str, **kw: Any) -> Answer:
        return cls(
            query=query,
            query_type=query_type,
            answerability="UNANSWERABLE",
            answer=cls.REFUSAL,
            **kw,
        )


# --- JSONL I/O -----------------------------------------------------------------


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    """Write dataclasses (or dicts) one per line. Returns the count written.

    `ensure_ascii=False` throughout: these files hold Devanagari, and escaping it
    to \\uXXXX makes them unreadable in exactly the situation where someone needs
    to read them, which is hand-checking an annotation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            payload = rec.as_dict() if hasattr(rec, "as_dict") else rec
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, cls: type[T] | None = None) -> Iterator[T | dict]:
    """Read JSONL, skipping blank lines so hand-edited files still load."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            yield cls.from_dict(raw) if cls is not None else raw  # type: ignore[attr-defined]


def content_hash(records: Iterable[Any]) -> str:
    """Stable hash over a record sequence, used to invalidate derived caches.

    An embedding matrix silently misaligned with the passage list it was built
    from produces plausible but meaningless numbers, and nothing about it looks
    wrong. Every cache stores this hash and refuses to load when it differs.
    """
    h = hashlib.sha256()
    for rec in records:
        payload = rec.as_dict() if hasattr(rec, "as_dict") else rec
        h.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()
