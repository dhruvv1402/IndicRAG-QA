"""Lexical retrieval: TF-IDF cosine and BM25-Okapi.

Both are implemented because the brief names both and because they fail
differently: TF-IDF's cosine normalisation makes it length-sensitive in a way
BM25's `b` parameter is explicitly designed to tame, and on a corpus whose
passages range from 20 to 415 tokens that difference shows.

BM25 is implemented here rather than taken from `rank_bm25` for one reason that
matters: the scores have to be comparable across queries for hybrid fusion and
for the answerability threshold. A shared implementation lets the candidate pool,
the tokenizer and the score normalisation stay in one place, and lets the
component scores be reported alongside the fused score in `Retrieved`.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..models import Passage, Retrieved
from .tokenize import tokenize

BM25_K1 = 1.2
BM25_B = 0.75


@dataclass
class LexicalIndex:
    """An inverted index serving both BM25 and TF-IDF over the same tokenization."""

    passage_ids: list[str]
    doc_tokens: list[list[str]]
    df: dict[str, int]
    doc_len: list[int]
    avg_len: float
    n_docs: int

    # --- construction ----------------------------------------------------------

    @classmethod
    def build(cls, passages: Sequence[Passage]) -> LexicalIndex:
        doc_tokens = [tokenize(p.text) for p in passages]
        df: Counter[str] = Counter()
        for toks in doc_tokens:
            df.update(set(toks))
        doc_len = [len(t) for t in doc_tokens]
        n = len(passages)
        return cls(
            passage_ids=[p.passage_id for p in passages],
            doc_tokens=doc_tokens,
            df=dict(df),
            doc_len=doc_len,
            avg_len=(sum(doc_len) / n) if n else 0.0,
            n_docs=n,
        )

    # --- scoring ---------------------------------------------------------------

    def _idf_bm25(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF with the +0.5 smoothing.

        `max(..., 1e-6)` guards the case where a term appears in more than half
        the corpus, for which the unsmoothed form goes negative and would let a
        common term *subtract* from a passage's score.
        """
        n_q = self.df.get(term, 0)
        return max(math.log((self.n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0), 1e-6)

    def _idf_tfidf(self, term: str) -> float:
        return math.log((self.n_docs + 1) / (self.df.get(term, 0) + 1)) + 1.0

    def search_bm25(self, query: str, k: int = 10) -> list[Retrieved]:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores = [0.0] * self.n_docs
        q_counts = Counter(q_terms)
        for term, qf in q_counts.items():
            if term not in self.df:
                continue
            idf = self._idf_bm25(term)
            for i, toks in enumerate(self.doc_tokens):
                tf = toks.count(term)
                if not tf:
                    continue
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.doc_len[i] / (self.avg_len or 1))
                scores[i] += idf * (tf * (BM25_K1 + 1)) / denom
        return self._top(scores, k, "bm25")

    def search_tfidf(self, query: str, k: int = 10) -> list[Retrieved]:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        q_vec = {t: (1 + math.log(c)) * self._idf_tfidf(t) for t, c in Counter(q_terms).items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scores = [0.0] * self.n_docs
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue
            counts = Counter(toks)
            dot = 0.0
            d_norm_sq = 0.0
            for t, c in counts.items():
                w = (1 + math.log(c)) * self._idf_tfidf(t)
                d_norm_sq += w * w
                if t in q_vec:
                    dot += w * q_vec[t]
            if dot:
                scores[i] = dot / (math.sqrt(d_norm_sq) * q_norm)
        return self._top(scores, k, "tfidf")

    def _top(self, scores: list[float], k: int, method: str) -> list[Retrieved]:
        ranked = sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)
        out: list[Retrieved] = []
        for rank, i in enumerate(ranked[:k], start=1):
            if scores[i] <= 0:
                break
            out.append(
                Retrieved(
                    passage_id=self.passage_ids[i],
                    score=float(scores[i]),
                    rank=rank,
                    method=method,
                    component_scores={method: float(scores[i])},
                )
            )
        return out

    # --- persistence -----------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "lexical.pkl").open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (directory / "lexical.meta.json").write_text(
            json.dumps(
                {"n_docs": self.n_docs, "avg_len": self.avg_len, "vocab": len(self.df)},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> LexicalIndex:
        with (Path(directory) / "lexical.pkl").open("rb") as fh:
            return pickle.load(fh)


def build_lexical(passages: Sequence[Passage], directory: Path) -> LexicalIndex:
    index = LexicalIndex.build(passages)
    index.save(Path(directory))
    return index
