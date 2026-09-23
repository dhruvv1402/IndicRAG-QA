"""Does script-aware fusion double votes that were actually cast?

Found reading the test-split error cases (paper §VII). Script-aware fusion
decides eligibility by script and doubles a cross-script passage's score to make
up for the lexical vote it could not receive. But BM25 does sometimes return a
cross-script passage -- a Hindi passage quoting "Soil Health Card" in Latin
letters, or sharing a digit string -- and the doubling then applies to a vote
that was cast. In the worst case this lifts an irrelevant passage to rank 1.

`count_observed_votes=True` treats a passage the lexical retriever returned as
eligible. Because the defect was found on the test split, the decision is made
on dev and the test figures are reported beside it as a post-hoc check, not as a
replacement for the committed results.

    python scripts/check-observed-votes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from indicrag.config import get_settings
    from indicrag.evaluation.all_reports import verification_label
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
        if i.answerable and i.gold_passage_ids and i.split in {"dev", "test"}
    ]
    if not passages or not items:
        print("need passages and answerable gold items with a split")
        return 1

    lex = LexicalIndex.load(cfg.lex_dir)
    spec = get(cfg.encoder_primary)
    dense = DenseIndex.load(spec, cfg.emb_dir, passages)
    enc = Encoder(spec)
    script_of = {p.passage_id: ("deva" if p.lang == "hi" else "latin") for p in passages}

    runs_of = {}
    fired = {}
    for i in items:
        lexical = lex.search_bm25(i.question, 50)
        dn = dense.search_vector(enc.encode_query(i.question), 50)
        qs = classify(i.question).script
        old = script_aware_rrf(lexical, dn, script_of=script_of, query_script=qs, k=5)
        new = script_aware_rrf(
            lexical, dn, script_of=script_of, query_script=qs, k=5, count_observed_votes=True
        )
        runs_of[i.id] = (old, new, dn[:5])
        # The defect fires when a top-5 passage was cross-script, lexically
        # returned anyway, and so had a cast vote doubled.
        fired[i.id] = any(
            "lexical" in r.component_scores and script_of[r.passage_id] != qs
            and qs not in ("mixed", "unknown")
            for r in old
        )

    print(f"{len(items)} answerable gold items ({verification_label(items)})\n")
    for split in ("dev", "test"):
        sub = [i for i in items if i.split == split]
        n_fired = sum(fired[i.id] for i in sub)
        runs = {
            "committed": evaluate("old", sub, lambda i: (runs_of[i.id][0], 0.0)),
            "observed votes": evaluate("new", sub, lambda i: (runs_of[i.id][1], 0.0)),
            "dense alone": evaluate("dense", sub, lambda i: (runs_of[i.id][2], 0.0)),
        }
        tag = "decides" if split == "dev" else "post-hoc: the defect was found here"
        print(f"=== {split} split ({tag})")
        print(f"    a doubled cast vote reaches the top 5 on {n_fired} of {len(sub)} queries\n")
        for key in ("cross-lingual", "code-mixed", "monolingual", None):
            sel = {
                name: [o for o in r.outcomes if key is None or o.language_group == key]
                for name, r in runs.items()
            }
            n = len(sel["committed"])
            if not n:
                continue
            print(f"--- {key or 'all'}  (n={n})")
            for name in runs:
                print(f"    {name:<16}{bootstrap_ci(sel[name], lambda o: o.recall_at(5))}")
            for base in ("committed", "dense alone"):
                res = paired_randomization_test(
                    sel["observed votes"], sel[base], lambda o: o.recall_at(5)
                )
                print(f"    observed - {base:<12}{res}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
