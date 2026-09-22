"""Script-aware fusion against plain RRF, on the gold set, per slice.

`eval retrieval` reports every system against BM25. The paper's central claim is
a different comparison -- script-aware fusion against *plain RRF*, which isolates
the arithmetic change from everything else -- and against dense retrieval alone,
which is what decides whether fusing is worth doing at all.

Both were previously measured only on synthetic probes, by a script that is not
in the repository. This computes them on the 320 answerable gold items.

    python scripts/check-fusion-goldset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from indicrag.config import get_settings
    from indicrag.evaluation.retrieval import evaluate
    from indicrag.evaluation.stats import bootstrap_ci, paired_randomization_test
    from indicrag.index.dense import DenseIndex, Encoder
    from indicrag.index.encoders import get
    from indicrag.index.hybrid import rrf_fusion, script_aware_rrf
    from indicrag.index.lexical import LexicalIndex
    from indicrag.models import Passage, QAItem, read_jsonl
    from indicrag.query.langid import classify

    # `--split test` scores the sealed test split only (PLAN §2.1); the default
    # is every answerable item, dev and test together.
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "all"
    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    items = [
        i for i in read_jsonl(ROOT / "evals" / "gold.jsonl", QAItem)
        if i.answerable and i.gold_passage_ids and (split == "all" or i.split == split)
    ]
    if not passages or not items:
        print("need passages and answerable gold items")
        return 1

    lex = LexicalIndex.load(cfg.lex_dir)
    spec = get(cfg.encoder_primary)
    dense = DenseIndex.load(spec, cfg.emb_dir, passages)
    enc = Encoder(spec)
    script_of = {p.passage_id: ("deva" if p.lang == "hi" else "latin") for p in passages}
    qvecs = {i.id: enc.encode_query(i.question) for i in items}

    def plain(i):
        return rrf_fusion(
            lex.search_bm25(i.question, 50), dense.search_vector(qvecs[i.id], 50), k=5
        )

    def aware(i):
        return script_aware_rrf(
            lex.search_bm25(i.question, 50),
            dense.search_vector(qvecs[i.id], 50),
            script_of=script_of,
            query_script=classify(i.question).script,
            k=5,
        )

    def dense_only(i):
        return dense.search_vector(qvecs[i.id], 5)

    runs = {
        "plain RRF": evaluate("plain", items, lambda i: (plain(i), 0.0)),
        "script-aware": evaluate("aware", items, lambda i: (aware(i), 0.0)),
        "dense alone": evaluate("dense", items, lambda i: (dense_only(i), 0.0)),
    }

    from indicrag.evaluation.all_reports import verification_label
    _label = verification_label(items)
    print(f"{len(items)} answerable gold items ({_label})\n")
    for slice_key in ("cross-lingual", "code-mixed", "monolingual", None):
        label = slice_key or "all"
        sel = {
            name: [o for o in r.outcomes if slice_key is None or o.language_group == slice_key]
            for name, r in runs.items()
        }
        n = len(sel["plain RRF"])
        if not n:
            continue
        print(f"--- {label}  (n={n})")
        for name in runs:
            print(f"    {name:<14}{bootstrap_ci(sel[name], lambda o: o.recall_at(5))}")
        for baseline in ("plain RRF", "dense alone"):
            res = paired_randomization_test(
                sel["script-aware"], sel[baseline], lambda o: o.recall_at(5)
            )
            print(f"    script-aware - {baseline:<12}{res}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
