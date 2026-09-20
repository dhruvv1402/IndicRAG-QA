"""Structure-aware passage segmentation.

Segmentation sets the ceiling on retrieval quality: a fact split across a chunk
boundary can never be retrieved intact, and no amount of encoder quality recovers
it. So the split follows document structure first and length second.

The source text carries `== Section ==` / `=== Subsection ===` markers, which is
why the plaintext extract was preferred over HTML upstream. They become
`section_path` ("Eligibility > Income criteria"), and that path earns its place
three times over: it is real context for the generator, it is what makes an error
case legible to a human, and it is the only thing that distinguishes the dozen
near-identical "Eligibility" passages that different schemes contribute.

Passage IDs must be stable across re-runs. The gold set, the embedding caches and
the error analysis all reference them by name, so a re-segmentation that
renumbers passages silently invalidates all three. IDs are therefore derived from
(doc_id, running index) and a re-segmentation is treated as a corpus version bump.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

from ..models import Document, Passage
from .normalize import normalize_text, split_sentences

#: Wikipedia plaintext heading markers: "== Title ==", "=== Sub ===", ...
_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)

#: Sections that are navigation or bibliography rather than content. Keeping them
#: would put reference lists and "See also" link farms into the retrieval pool,
#: where they match many queries lexically and answer none of them.
_SKIP_SECTIONS = {
    "references", "external links", "see also", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes",
    "सन्दर्भ", "इन्हें भी देखें", "बाहरी कड़ियाँ", "बाहरी कडियां", "टिप्पणी",
    "संदर्भ", "अतिरिक्त पठन", "ग्रन्थसूची",
}

TARGET_TOKENS = 170
MIN_TOKENS = 60
MAX_TOKENS = 240
OVERLAP_RATIO = 0.25


def count_tokens(text: str) -> int:
    """Whitespace token count.

    Deliberately not a model tokenizer. Segmentation must not change when the
    encoder changes -- otherwise passage IDs shift, caches invalidate and the gold
    set breaks. A crude, stable count is the right trade here; the length bounds
    are approximate by design.
    """
    return len(text.split())


@dataclass
class Section:
    path: str
    text: str
    level: int


def split_sections(text: str) -> list[Section]:
    """Split plaintext into sections, carrying a breadcrumb path for each.

    Content before the first heading is the article lead, which is usually the
    densest summary in the document and must not be dropped.
    """
    matches = list(_HEADING_RE.finditer(text))
    sections: list[Section] = []

    lead = text[: matches[0].start()] if matches else text
    if lead.strip():
        sections.append(Section(path="Introduction", text=lead.strip(), level=1))

    stack: list[str] = []
    for i, m in enumerate(matches):
        level = len(m.group(1)) - 1  # "==" is level 1
        title = m.group(2).strip()
        body = text[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)]

        stack = stack[: level - 1]
        stack.append(title)

        if title.strip().lower() in _SKIP_SECTIONS:
            continue
        if not body.strip():
            continue
        sections.append(Section(path=" > ".join(stack), text=body.strip(), level=level))

    return sections


def _pack(sentences: list[str], *, target: int, hard_max: int) -> Iterator[list[str]]:
    """Accumulate sentences into chunks near `target`, never splitting a sentence.

    A sentence longer than `hard_max` on its own is emitted alone rather than
    truncated: over-long is recoverable by the encoder's own truncation, whereas
    a severed clause is a fact destroyed at index time.
    """
    buf: list[str] = []
    size = 0
    for s in sentences:
        n = count_tokens(s)
        if buf and size + n > hard_max:
            yield buf
            buf, size = [], 0
        buf.append(s)
        size += n
        if size >= target:
            yield buf
            buf, size = [], 0
    if buf:
        yield buf


def segment_document(doc: Document, text: str) -> list[Passage]:
    """Split one document into passages with stable IDs and section metadata."""
    passages: list[Passage] = []
    index = 0
    cursor = 0

    for section in split_sections(text):
        sentences = split_sentences(normalize_text(section.text))
        if not sentences:
            continue

        chunks = list(_pack(sentences, target=TARGET_TOKENS, hard_max=MAX_TOKENS))

        for c_i, chunk in enumerate(chunks):
            body = " ".join(chunk)

            # Overlap: prepend the tail of the previous chunk so a fact sitting on
            # a boundary appears whole in at least one passage. Only applied where
            # the split was length-driven rather than structural -- a section
            # boundary is a real semantic break and bleeding across it would put
            # one scheme's clause inside another's passage.
            if c_i > 0 and OVERLAP_RATIO > 0:
                prev = chunks[c_i - 1]
                n_overlap = max(1, int(len(prev) * OVERLAP_RATIO))
                body = " ".join(prev[-n_overlap:]) + " " + body

            # Too small to stand alone: fold into the previous passage rather than
            # emit a fragment. A 3-token passage ("Main article: Skill India")
            # cannot win a retrieval but can still absorb probability mass and
            # pollute the candidate pool. The earlier version of this guard only
            # fired when a section produced several chunks, so a whole section
            # that was itself tiny -- a stub heading, a one-line note -- sailed
            # through; the corpus came out with a 3-token minimum because of it.
            if count_tokens(body) < MIN_TOKENS:
                if passages and passages[-1].doc_id == doc.doc_id:
                    merged = passages[-1].text + " " + body
                    passages[-1].text = merged
                    passages[-1].text_raw = passages[-1].text_raw + " " + " ".join(chunk)
                    passages[-1].token_count = count_tokens(merged)
                    continue
                # Nothing to merge into (first passage of the document): keep it
                # only if it is not pure boilerplate.
                if count_tokens(body) < 15:
                    continue

            start = text.find(chunk[0][:40], cursor)
            if start < 0:
                start = cursor
            cursor = start + len(body)

            passages.append(
                Passage(
                    passage_id=f"{doc.doc_id}#p{index:04d}",
                    doc_id=doc.doc_id,
                    scheme=doc.scheme,
                    lang=doc.lang,
                    text=body,
                    text_raw=" ".join(chunk),
                    section_path=section.path,
                    page=0,
                    char_span=(start, cursor),
                    token_count=count_tokens(body),
                )
            )
            index += 1

    return passages


def segment_all(docs: Iterable[tuple[Document, str]]) -> list[Passage]:
    out: list[Passage] = []
    for doc, text in docs:
        out.extend(segment_document(doc, text))
    return out
