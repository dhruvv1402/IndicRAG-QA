"""Does question-passage wording overlap inflate the lexical retriever?

§III-F promises three leakage controls: a paraphrase instruction in the
generation prompt, a published overlap distribution, and the headline retrieval
comparison repeated on the low-overlap tertile. The first is in the prompt. The
second and third had never been produced.

This does both, and it does them on the cells where leakage is *possible*.
Cross-script pairs score essentially zero overlap by construction -- a
Devanagari question and a Latin passage share no tokens -- so pooling them with
the same-script cells makes the control look stronger than it is while
measuring the script boundary instead of the annotation.

    python scripts/check-leakage.py
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

#: Cells whose question and evidence share a script, so copied wording could
#: hand the lexical retriever its answer.
SAME_SCRIPT = {("en", "en"), ("hi", "hi"), ("hinglish", "en")}


def main() -> int:
    from indicrag.config import get_settings
    from indicrag.evaluation.qa import normalize_answer
    from indicrag.evaluation.retrieval import evaluate
    from indicrag.evaluation.stats import paired_randomization_test
    from indicrag.index.dense import DenseIndex, Encoder
    from indicrag.index.encoders import get
    from indicrag.index.lexical import LexicalIndex
    from indicrag.models import Passage, QAItem, read_jsonl

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    by = {p.passage_id: p for p in passages}
    items = [
        i for i in read_jsonl(ROOT / "evals" / "gold.jsonl", QAItem)
        if i.answerable and i.gold_passage_ids
    ]

    def overlap(item) -> float:
        text = " ".join(by[p].text for p in item.gold_passage_ids if p in by)
        a = set(normalize_answer(item.question).split())
        b = set(normalize_answer(text).split())
        return len(a & b) / len(a | b) if (a | b) else 0.0

    scored = [(i, overlap(i)) for i in items]

    from indicrag.evaluation.all_reports import verification_label
    _label = verification_label(items)
    print(f"{len(scored)} answerable gold items ({_label})\n")
    print("OVERLAP BY CELL -- content-token Jaccard, question against gold passage")
    print("-" * 78)
    print(f"  {'cell':<16}{'n':>5}{'median':>9}{'mean':>8}{'max':>8}   leakage possible?")
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item, value in scored:
        cells[(item.query_lang, item.passage_lang)].append(value)
    for key in sorted(cells):
        v = cells[key]
        tag = "yes" if key in SAME_SCRIPT else "no -- cross-script"
        print(
            f"  {key[0] + '->' + key[1]:<16}{len(v):>5}{statistics.median(v):>9.3f}"
            f"{statistics.fmean(v):>8.3f}{max(v):>8.3f}   {tag}"
        )

    same = [(i, v) for i, v in scored if (i.query_lang, i.passage_lang) in SAME_SCRIPT]
    vals = [v for _i, v in same]
    print()
    print(f"  same-script cells: n={len(same)} median {statistics.median(vals):.3f} "
          f"mean {statistics.fmean(vals):.3f} max {max(vals):.3f}")
    print()
    print("  Cross-script cells sit at zero because the two scripts share no")
    print("  tokens. That is the script boundary, not evidence of annotation")
    print("  hygiene, so the same-script figure is the one that bears on leakage.")

    # --- the within-cell comparison -------------------------------------------
    #
    # Splitting all same-script items into overlap tertiles does not work: the
    # low tertile comes out 54/61 hinglish->en, because that cell's questions
    # are Romanized Hindi whose function words match nothing in an English
    # passage. The split is then a proxy for the cell, and comparing across it
    # measures Hinglish against monolingual rather than low overlap against
    # high. Leakage has to be tested *within* a cell, holding the language pair
    # fixed.
    lex = LexicalIndex.load(cfg.lex_dir)
    spec = get(cfg.encoder_primary)
    index = DenseIndex.load(spec, cfg.emb_dir, passages)
    enc = Encoder(spec)

    print()
    print("RETRIEVAL BY OVERLAP, WITHIN EACH SAME-SCRIPT CELL")
    print("-" * 78)
    print("  If BM25's edge is annotation leakage, it is larger on the high-overlap")
    print("  half of a cell than on the low-overlap half of the same cell.")
    print()

    for cell in sorted(SAME_SCRIPT):
        band = sorted(
            ((i, v) for i, v in scored if (i.query_lang, i.passage_lang) == cell),
            key=lambda kv: kv[1],
        )
        if len(band) < 20:
            continue
        half = len(band) // 2
        print(f"  {cell[0]}->{cell[1]}  (n={len(band)})")
        deltas = {}
        for name, part in (("low ", band[:half]), ("high", band[half:])):
            part_items = [i for i, _v in part]
            qv = {i.id: enc.encode_query(i.question) for i in part_items}
            bm = evaluate("bm25", part_items, lambda i: (lex.search_bm25(i.question, 5), 0.0))
            dn = evaluate("dense", part_items, lambda i, qv=qv: (index.search_vector(qv[i.id], 5), 0.0))
            res = paired_randomization_test(
                bm.outcomes, dn.outcomes, lambda o: o.recall_at(5)
            )
            deltas[name.strip()] = res.delta
            span = f"{part[0][1]:.3f}-{part[-1][1]:.3f}"
            print(
                f"    {name} overlap {span:<14} n={len(part_items):<4} "
                f"BM25 {bm.recall_at(5):.3f}  dense {dn.recall_at(5):.3f}  {res}"
            )
        gap = deltas.get("high", 0.0) - deltas.get("low", 0.0)
        verdict = (
            "consistent with leakage" if gap > 0.05
            else "no leakage signal" if abs(gap) <= 0.05
            else "opposite of leakage"
        )
        print(f"    BM25 edge, high minus low: {gap:+.3f}  -> {verdict}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
