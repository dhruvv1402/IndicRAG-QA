"""Run the retrieval comparison over every available index.

Encoders whose embeddings have not been built yet are skipped with a note rather
than failing the run. A partial comparison is still a comparison, and on a
CPU-only machine where encoding the full registry takes the better part of an
hour, a harness that refuses to report anything until every model is ready is a
harness nobody runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..config import get_settings
from ..index.dense import CacheMismatch, DenseIndex, Encoder
from ..index.encoders import DEFAULT_ORDER, REGISTRY, get
from ..index.hybrid import rrf_fusion, script_aware_rrf, weighted_fusion
from ..query.langid import classify
from ..index.lexical import LexicalIndex
from ..models import Document, Passage, QAItem, read_jsonl
from .probes import build_probes
from .retrieval import RetrievalReport, evaluate

CANDIDATES = 50
TOP_K = 10


def title_map(manifest_path) -> dict[tuple[str, str], str]:
    """(scheme, lang) -> source article title, for language-correct probe stems."""
    out: dict[tuple[str, str], str] = {}
    for doc in read_jsonl(manifest_path, Document):
        out[(doc.scheme, doc.lang)] = doc.title
    return out


def _timed(fn: Callable[[QAItem], list]) -> Callable[[QAItem], tuple[list, float]]:
    def run(item: QAItem):
        start = time.time()
        hits = fn(item)
        return hits, time.time() - start

    return run


def run_retrieval(
    passages: Sequence[Passage],
    items: Sequence[QAItem],
    *,
    alpha: float = 0.4,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[RetrievalReport], list[str]]:
    """Evaluate lexical, every available dense encoder, and the two fusions."""
    cfg = get_settings()
    say = progress or (lambda _m: None)
    skipped: list[str] = []

    lex = LexicalIndex.load(cfg.lex_dir)
    reports: list[RetrievalReport] = [
        evaluate("BM25", items, _timed(lambda i: lex.search_bm25(i.question, TOP_K))),
        evaluate("TF-IDF", items, _timed(lambda i: lex.search_tfidf(i.question, TOP_K))),
    ]
    say(f"  BM25 / TF-IDF over {len(items)} probes")

    best_dense: tuple[str, DenseIndex, dict] | None = None

    for name in DEFAULT_ORDER:
        spec = get(name)
        poolings = ["mean", "cls"] if spec.is_mlm else ["mean"]
        for pooling in poolings:
            label = f"{name.split('/')[-1]}{'/' + pooling if spec.is_mlm else ''}"
            try:
                index = DenseIndex.load(spec, cfg.emb_dir, passages, pooling=pooling)
            except FileNotFoundError:
                skipped.append(f"{label}: embeddings not built")
                continue
            except CacheMismatch as exc:
                skipped.append(f"{label}: {exc}")
                continue

            encoder = Encoder(spec, pooling=pooling)
            # Query vectors are computed once and reused across k, and reused by
            # the hybrid arms below. Encoding is by far the dominant cost per
            # query on CPU, and re-encoding per system would multiply the whole
            # evaluation by the number of arms for no gain.
            qvecs = {i.id: encoder.encode_query(i.question) for i in items}
            report = evaluate(
                label, items, _timed(lambda i, ix=index, qv=qvecs: ix.search_vector(qv[i.id], TOP_K))
            )
            reports.append(report)
            say(f"  {label}: R@5={report.recall_at(5):.3f}")

            if best_dense is None or report.recall_at(5) > best_dense[2]["r5"]:
                best_dense = (label, index, {"r5": report.recall_at(5), "qvecs": qvecs})

    if best_dense is not None:
        label, index, extra = best_dense
        qvecs = extra["qvecs"]
        reports.append(
            evaluate(
                f"Hybrid RRF (BM25 + {label})",
                items,
                _timed(
                    lambda i: rrf_fusion(
                        lex.search_bm25(i.question, CANDIDATES),
                        index.search_vector(qvecs[i.id], CANDIDATES),
                        k=TOP_K,
                    )
                ),
            )
        )
        reports.append(
            evaluate(
                f"Hybrid weighted a={alpha:g} (BM25 + {label})",
                items,
                _timed(
                    lambda i: weighted_fusion(
                        lex.search_bm25(i.question, CANDIDATES),
                        index.search_vector(qvecs[i.id], CANDIDATES),
                        alpha=alpha,
                        k=TOP_K,
                    )
                ),
            )
        )
        # Script-aware fusion is the paper's central contribution and was not
        # among the arms this harness evaluated: its numbers came from a
        # one-off script, on probes only. So `eval retrieval` never measured
        # the method the system actually ships as its default, and there was no
        # gold-set figure for it at all.
        script_of = {
            p.passage_id: ("deva" if p.lang == "hi" else "latin") for p in passages
        }
        reports.append(
            evaluate(
                f"Hybrid RRF script-aware (BM25 + {label})",
                items,
                _timed(
                    lambda i: script_aware_rrf(
                        lex.search_bm25(i.question, CANDIDATES),
                        index.search_vector(qvecs[i.id], CANDIDATES),
                        script_of=script_of,
                        query_script=classify(i.question).script,
                        k=TOP_K,
                    )
                ),
            )
        )
        say(f"  hybrid arms built on {label}")
    else:
        skipped.append("hybrid: no dense index available")

    for name, spec in REGISTRY.items():
        if not spec.available:
            skipped.append(f"{name.split('/')[-1]}: {spec.note.split('.')[0]}")

    return reports, skipped


def alpha_verdict(sweep: Sequence[tuple[float, float]], *, margin: float = 0.02) -> str:
    """Judge H2 from an alpha sweep, requiring the win to clear a noise margin.

    The naive test -- "does the best interior alpha beat the best endpoint" --
    reports H2 supported on a 0.001 difference, which on 180 queries is one query
    changing its mind. The same report states that differences smaller than the
    bootstrap interval width are not distinguishable from noise, so declaring a
    win on 0.001 would contradict the page it is printed on. `margin` is the
    minimum improvement worth calling a result.
    """
    if not sweep:
        return "no sweep data"
    interior = [(a, r) for a, r in sweep if 0.0 < a < 1.0]
    if not interior:
        return "no interior alpha evaluated"
    best_a, best_r = max(interior, key=lambda x: x[1])
    endpoint = max(sweep[0][1], sweep[-1][1])
    delta = best_r - endpoint
    if delta > margin:
        return f"best a={best_a:.1f} ({best_r:.3f}) beats best endpoint ({endpoint:.3f}) by {delta:+.3f} -> H2 SUPPORTED"
    return (
        f"best a={best_a:.1f} ({best_r:.3f}) vs best endpoint ({endpoint:.3f}), "
        f"delta {delta:+.3f} < {margin:.2f} margin -> H2 NOT SUPPORTED (within noise)"
    )


def alpha_sweep(
    passages: Sequence[Passage],
    items: Sequence[QAItem],
    dense_label: str | None = None,
    *,
    points: Sequence[float] = tuple(i / 10 for i in range(11)),
) -> list[tuple[float, float]]:
    """Recall@5 across alpha. The endpoints recover pure dense and pure lexical,
    so this curve is the evidence for or against H2: if no interior alpha beats
    both ends, hybrid fusion is not adding signal and the paper must say so."""
    cfg = get_settings()
    lex = LexicalIndex.load(cfg.lex_dir)

    index = None
    encoder = None
    for name in DEFAULT_ORDER:
        spec = get(name)
        try:
            index = DenseIndex.load(spec, cfg.emb_dir, passages)
        except (FileNotFoundError, CacheMismatch):
            continue
        encoder = Encoder(spec)
        if dense_label is None or name.endswith(dense_label):
            break
    if index is None or encoder is None:
        return []

    qvecs = {i.id: encoder.encode_query(i.question) for i in items}
    out: list[tuple[float, float]] = []
    for a in points:
        rep = evaluate(
            f"a={a}",
            items,
            _timed(
                lambda i, a=a: weighted_fusion(
                    lex.search_bm25(i.question, CANDIDATES),
                    index.search_vector(qvecs[i.id], CANDIDATES),
                    alpha=a,
                    k=TOP_K,
                )
            ),
        )
        out.append((a, rep.recall_at(5)))
    return out


def load_probes(passages: Sequence[Passage], per_shape: int = 60) -> list[QAItem]:
    cfg = get_settings()
    return build_probes(passages, per_shape=per_shape, titles=title_map(cfg.manifest_path))
