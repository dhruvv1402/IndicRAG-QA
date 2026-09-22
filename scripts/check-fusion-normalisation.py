"""Min-max against z-score normalisation in weighted fusion, and the reason given for it.

Section IV-E states that min-max is preferred "because BM25 score distributions
are strongly right-skewed and z-score leaves the fused ranking dominated by
lexical outliers", and that "both are implemented so the choice is ablatable
rather than asserted". Both normalisers exist in `index/hybrid.py`; nothing ever
passes `normalisation="zscore"`. The ablation the sentence claims to license had
never been run, so the choice was asserted after all.

This runs it, and separately measures the two halves of the stated mechanism:

  1. the skew of the BM25 candidate distribution, which is the premise,
  2. what the fused top-5 is actually made of under each normaliser, which is
     the consequence the premise is supposed to produce.

    INDICRAG_DATA_DIR=... python scripts/check-fusion-normalisation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CANDIDATES = 50
TOP_K = 5
ALPHA = 0.4


def _skew(values: list[float]) -> float:
    """Fisher-Pearson sample skewness. Positive is a right tail."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m3 = sum((v - mean) ** 3 for v in values) / n
    if m2 < 1e-18:
        return 0.0
    g1 = m3 / m2**1.5
    # Adjusted Fisher-Pearson, to match what scipy reports by default elsewhere.
    return g1 * ((n * (n - 1)) ** 0.5 / (n - 2))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def main() -> int:
    from indicrag.config import get_settings
    from indicrag.evaluation.retrieval import evaluate
    from indicrag.evaluation.stats import bootstrap_ci, paired_randomization_test
    from indicrag.index.dense import DenseIndex, Encoder
    from indicrag.index.encoders import get
    from indicrag.index.hybrid import weighted_fusion
    from indicrag.index.lexical import LexicalIndex
    from indicrag.models import Passage, QAItem, read_jsonl

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    items = [
        i
        for i in read_jsonl(ROOT / "evals" / "gold.jsonl", QAItem)
        if i.answerable and i.gold_passage_ids
    ]
    if not passages or not items:
        print("need passages and answerable gold items")
        return 1

    lex = LexicalIndex.load(cfg.lex_dir)
    spec = get(cfg.encoder_primary)
    dense = DenseIndex.load(spec, cfg.emb_dir, passages)
    enc = Encoder(spec)

    # Both retrieval runs are computed once per item and reused across every
    # alpha and both normalisers. Encoding is the dominant cost; fusion is
    # arithmetic over 100 candidates.
    runs: dict[str, tuple[list, list]] = {}
    for i in items:
        qv = enc.encode_query(i.question)
        runs[i.id] = (
            lex.search_bm25(i.question, CANDIDATES),
            dense.search_vector(qv, CANDIDATES),
        )

    from indicrag.evaluation.all_reports import verification_label
    _label = verification_label(items)
    print(f"{len(items)} answerable gold items ({_label})")
    print(f"weighted fusion, alpha={ALPHA:g}, {CANDIDATES} candidates per component\n")

    # --- 1. The premise: is the BM25 candidate distribution right-skewed? ------
    print("SCORE DISTRIBUTION OF THE CANDIDATE POOL")
    print("-" * 78)
    print("  The stated reason for min-max. Skew is over each query's own")
    print("  candidate scores; the table reports the distribution of that")
    print("  per-query skew across the gold set.\n")
    print(f"  {'component':<12}{'mean skew':>11}{'median':>9}{'right-skewed':>15}{'top1/mean':>12}")
    skews: dict[str, list[float]] = {"BM25": [], "dense": []}
    ratios: dict[str, list[float]] = {"BM25": [], "dense": []}
    for i in items:
        for name, run in zip(("BM25", "dense"), runs[i.id], strict=True):
            scores = [r.score for r in run]
            if len(scores) < 3:
                continue
            skews[name].append(_skew(scores))
            mean = sum(scores) / len(scores)
            ratios[name].append(max(scores) / mean if abs(mean) > 1e-12 else 0.0)
    for name in ("BM25", "dense"):
        vals = skews[name]
        share = sum(1 for v in vals if v > 0) / len(vals) if vals else 0.0
        print(
            f"  {name:<12}{sum(vals) / len(vals):>11.3f}{_median(vals):>9.3f}"
            f"{share:>14.1%}{sum(ratios[name]) / len(ratios[name]):>12.2f}"
        )
    print()

    # --- 2. The ablation the paper says the implementation licenses ------------
    def fuse(norm: str, alpha: float = ALPHA):
        def run(i):
            lexical, den = runs[i.id]
            return (
                weighted_fusion(lexical, den, alpha=alpha, k=TOP_K, normalisation=norm),
                0.0,
            )

        return run

    reports = {norm: evaluate(norm, items, fuse(norm)) for norm in ("minmax", "zscore")}

    print("RECALL@5 BY SLICE")
    print("-" * 78)
    for slice_key in ("monolingual", "cross-lingual", "code-mixed", None):
        label = slice_key or "all"
        sel = {
            norm: [
                o
                for o in rep.outcomes
                if slice_key is None or o.language_group == slice_key
            ]
            for norm, rep in reports.items()
        }
        n = len(sel["minmax"])
        if not n:
            continue
        print(f"  --- {label}  (n={n})")
        for norm in ("minmax", "zscore"):
            print(f"      {norm:<10}{bootstrap_ci(sel[norm], lambda o: o.recall_at(TOP_K))}")
        res = paired_randomization_test(
            sel["zscore"], sel["minmax"], lambda o: o.recall_at(TOP_K)
        )
        print(f"      zscore - minmax   {res}\n")

    # --- 3. Does the verdict hold across alpha, or only at 0.4? ---------------
    print("RECALL@5 ACROSS ALPHA  (alpha=1 pure lexical, alpha=0 pure dense)")
    print("-" * 78)
    print(f"  {'alpha':>7}{'minmax':>10}{'zscore':>10}{'delta':>10}")
    for step in range(11):
        alpha = step / 10
        row = {
            norm: evaluate(norm, items, fuse(norm, alpha)).recall_at(TOP_K)
            for norm in ("minmax", "zscore")
        }
        print(
            f"  {alpha:>7.1f}{row['minmax']:>10.3f}{row['zscore']:>10.3f}"
            f"{row['zscore'] - row['minmax']:>+10.3f}"
        )
    print()

    # --- 4. The consequence: what is the fused top-5 made of? -----------------
    print("COMPOSITION OF THE FUSED TOP-5")
    print("-" * 78)
    print("  'lexical-only' is a passage the dense retriever never returned, so")
    print("  its dense contribution is imputed rather than measured. That is what")
    print("  'dominated by lexical outliers' would look like if it happened.\n")
    print(f"  {'norm':<10}{'lexical-only':>14}{'dense-only':>12}{'both':>8}{'lex share of top-1':>21}")
    for norm in ("minmax", "zscore"):
        lex_only = dense_only = both = 0
        lex_share: list[float] = []
        for i in items:
            lexical, den = runs[i.id]
            in_lex = {r.passage_id for r in lexical}
            in_den = {r.passage_id for r in den}
            fused = weighted_fusion(lexical, den, alpha=ALPHA, k=TOP_K, normalisation=norm)
            for rank, r in enumerate(fused, start=1):
                if r.passage_id in in_lex and r.passage_id in in_den:
                    both += 1
                elif r.passage_id in in_lex:
                    lex_only += 1
                else:
                    dense_only += 1
                if rank == 1:
                    lex_part = ALPHA * r.component_scores.get("lexical", 0.0)
                    den_part = (1 - ALPHA) * r.component_scores.get("dense", 0.0)
                    total = abs(lex_part) + abs(den_part)
                    lex_share.append(abs(lex_part) / total if total > 1e-12 else 0.0)
        total_slots = lex_only + dense_only + both
        print(
            f"  {norm:<10}{lex_only / total_slots:>13.1%}{dense_only / total_slots:>12.1%}"
            f"{both / total_slots:>8.1%}{sum(lex_share) / len(lex_share):>20.1%}"
        )
    print()

    # --- 5. What absence means under each normaliser --------------------------
    print("WHAT A MISSING COMPONENT SCORES")
    print("-" * 78)
    print("  A passage only one component returned contributes 0 from the other.")
    print("  Under min-max 0 is the floor of the observed range; under z-score it")
    print("  is the mean. The imputation is not the same penalty.\n")
    from indicrag.index.hybrid import _minmax, _zscore

    for norm, normaliser in (("minmax", _minmax), ("zscore", _zscore)):
        pcts: list[float] = []
        for i in items:
            _lexical, den = runs[i.id]
            scores = {r.passage_id: r.score for r in den}
            if len(scores) < 2:
                continue
            normed = normaliser(scores)
            vals = sorted(normed.values())
            below = sum(1 for v in vals if v < 0.0)
            pcts.append(below / len(vals))
        mean_below = sum(pcts) / len(pcts) if pcts else 0.0
        print(
            f"  {norm:<10}an absent dense component ranks above "
            f"{mean_below:.1%} of the candidates it is compared against"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
