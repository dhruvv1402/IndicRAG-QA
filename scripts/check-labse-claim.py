"""Is LaBSE's code-mixed advantage over script-aware fusion significant?

§VI-F of the paper states that LaBSE used alone reaches 0.332 on the code-mixed
slice against 0.173 for script-aware fusion, and treats that as a caveat
qualifying the paper's own contribution. The slice holds 30 queries. A
difference of 0.159 on 30 binary-ish outcomes may or may not survive a paired
test, and a caveat asserted without one is no better than a claim asserted
without one.

    python scripts/check-labse-claim.py

Prints the paired comparison on the code-mixed slice and on the cross-lingual
slice, where LaBSE is the weaker system and the same question applies in
reverse.
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
    from indicrag.index.hybrid import script_aware_rrf
    from indicrag.index.lexical import LexicalIndex
    from indicrag.models import Passage, QAItem, read_jsonl
    from indicrag.query.langid import classify

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    items = list(read_jsonl(ROOT / "evals" / "probes.jsonl", QAItem))
    if not passages or not items:
        print("need passages and evals/probes.jsonl")
        return 1

    lex = LexicalIndex.load(cfg.lex_dir)
    script_of = {p.passage_id: ("deva" if p.lang == "hi" else "latin") for p in passages}

    def load(name):
        spec = get(name)
        return DenseIndex.load(spec, cfg.emb_dir, passages), Encoder(spec)

    e5_index, e5_enc = load(cfg.encoder_primary)
    labse_index, labse_enc = load("sentence-transformers/LaBSE")

    e5_q = {i.id: e5_enc.encode_query(i.question) for i in items}
    labse_q = {i.id: labse_enc.encode_query(i.question) for i in items}

    def fused(item):
        return script_aware_rrf(
            lex.search_bm25(item.question, 50),
            e5_index.search_vector(e5_q[item.id], 50),
            script_of=script_of,
            query_script=classify(item.question).script,
            k=5,
        )

    def labse(item):
        return labse_index.search_vector(labse_q[item.id], 5)

    a = evaluate("script-aware fusion", items, lambda i: (fused(i), 0.0))
    b = evaluate("LaBSE alone", items, lambda i: (labse(i), 0.0))

    print(f"{len(items)} probes\n")
    for slice_key in ("code-mixed", "cross-lingual", "monolingual"):
        a_sel = [o for o in a.outcomes if o.language_group == slice_key]
        b_sel = [o for o in b.outcomes if o.language_group == slice_key]
        if not a_sel:
            continue
        a_ci = bootstrap_ci(a_sel, lambda o: o.recall_at(5))
        b_ci = bootstrap_ci(b_sel, lambda o: o.recall_at(5))
        res = paired_randomization_test(b_sel, a_sel, lambda o: o.recall_at(5))
        print(f"--- {slice_key}  (n={len(a_sel)})")
        print(f"    script-aware fusion  {a_ci}")
        print(f"    LaBSE alone          {b_ci}")
        print(f"    LaBSE - fusion       {res}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
