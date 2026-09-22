"""Script-aware fusion over each dense encoder, on the gold set.

§IX lists "script-aware fusion over LaBSE" as the most obvious next experiment
and not an oversight. The gold-set numbers make the question sharper than the
probes did: there LaBSE reaches 0.422 on the cross-lingual slice against 0.367
for multilingual-e5-base, so fusing over LaBSE may beat the 0.500 that fusing
over e5 achieves.

The correction is a statement about how rankings are combined and is orthogonal
to which encoder produces them, so this is a fair test of that claim rather than
a new method.

    python scripts/check-fusion-encoders.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ENCODERS = [
    ("e5-base", "intfloat/multilingual-e5-base"),
    ("LaBSE", "sentence-transformers/LaBSE"),
    ("MiniLM-L12", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
]


def main() -> int:
    from indicrag.config import get_settings
    from indicrag.evaluation.retrieval import evaluate
    from indicrag.evaluation.stats import bootstrap_ci, paired_randomization_test
    from indicrag.index.dense import DenseIndex, Encoder
    from indicrag.index.encoders import get
    from indicrag.index.hybrid import script_aware_rrf
    from indicrag.index.lexical import LexicalIndex
    from indicrag.models import Passage, QAItem, read_jsonl
    from indicrag.query.langid import classify

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    items = [
        i for i in read_jsonl(ROOT / "evals" / "gold.jsonl", QAItem)
        if i.answerable and i.gold_passage_ids
    ]
    lex = LexicalIndex.load(cfg.lex_dir)
    script_of = {p.passage_id: ("deva" if p.lang == "hi" else "latin") for p in passages}

    runs: dict[str, object] = {}
    for label, name in ENCODERS:
        try:
            spec = get(name)
            index = DenseIndex.load(spec, cfg.emb_dir, passages)
            enc = Encoder(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: skipped ({type(exc).__name__}: {exc})")
            continue
        qv = {i.id: enc.encode_query(i.question) for i in items}

        def fuse(i, index=index, qv=qv):
            return script_aware_rrf(
                lex.search_bm25(i.question, 50),
                index.search_vector(qv[i.id], 50),
                script_of=script_of,
                query_script=classify(i.question).script,
                k=5,
            )

        runs[f"fusion/{label}"] = evaluate(label, items, lambda i, f=fuse: (f(i), 0.0))
        runs[f"dense/{label}"] = evaluate(
            label, items, lambda i, ix=index, q=qv: (ix.search_vector(q[i.id], 5), 0.0)
        )

    from indicrag.evaluation.all_reports import verification_label
    _label = verification_label(items)
    print(f"\n{len(items)} answerable gold items ({_label})\n")
    for slice_key in ("cross-lingual", "code-mixed", "monolingual", None):
        label = slice_key or "all"
        print(f"--- {label}")
        sel = {
            k: [o for o in r.outcomes if slice_key is None or o.language_group == slice_key]
            for k, r in runs.items()
        }
        for k in sorted(sel):
            if sel[k]:
                print(f"    {k:<20}{bootstrap_ci(sel[k], lambda o: o.recall_at(5))}")
        base = "fusion/e5-base"
        for k in sorted(sel):
            if k.startswith("fusion/") and k != base and sel[k] and sel[base]:
                res = paired_randomization_test(
                    sel[k], sel[base], lambda o: o.recall_at(5)
                )
                print(f"    {k} - {base}  {res}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
