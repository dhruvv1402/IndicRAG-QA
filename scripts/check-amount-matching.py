"""Would canonicalising amounts in the retrieval tokenizer help?

Section IV-B implies it already happens. It does not: canonical_amounts is
called by answer scoring, not by index/tokenize.py. This measures the gap on
the gold items whose question carries an amount.
"""
import sys
from pathlib import Path
ROOT = Path(r"C:\Users\xcite\Downloads\IndicRAG-QA")
sys.path.insert(0, str(ROOT / "src"))
import indicrag.index.tokenize as tk
from indicrag.config import get_settings
from indicrag.corpus.normalize import canonical_amounts
from indicrag.evaluation.retrieval import evaluate
from indicrag.evaluation.stats import bootstrap_ci, paired_randomization_test
from indicrag.index.lexical import LexicalIndex
from indicrag.models import Passage, QAItem, read_jsonl

cfg = get_settings()
passages = list(read_jsonl(cfg.passages_path, Passage))
items = [i for i in read_jsonl(ROOT / "evals" / "gold.jsonl", QAItem)
         if i.answerable and i.gold_passage_ids]
amt = [i for i in items if canonical_amounts(i.question)]
print(f"{len(items)} answerable gold items; {len(amt)} carry an amount\n")

base_tokenize = tk.tokenize
plain = LexicalIndex.build(passages)

def augmented(text, *, stem=True, drop_stopwords=True):
    out = base_tokenize(text, stem=stem, drop_stopwords=drop_stopwords)
    return out + [f"amt{a}" for a in canonical_amounts(text)]

tk.tokenize = augmented
import indicrag.index.lexical as lx
lx.tokenize = augmented
richer = LexicalIndex.build(passages)

def run(index, label, subset):
    return evaluate(label, subset, lambda i: (index.search_bm25(i.question, 5), 0.0))

for name, subset in (("amount-bearing", amt), ("all answerable", items)):
    if not subset:
        continue
    lx.tokenize = base_tokenize
    a = run(plain, "plain", subset)
    lx.tokenize = augmented
    b = run(richer, "amounts", subset)
    print(f"--- {name}  (n={len(subset)})")
    print(f"    BM25 plain           {bootstrap_ci(a.outcomes, lambda o: o.recall_at(5))}")
    print(f"    BM25 + amount key    {bootstrap_ci(b.outcomes, lambda o: o.recall_at(5))}")
    print(f"    paired               "
          f"{paired_randomization_test(b.outcomes, a.outcomes, lambda o: o.recall_at(5))}")
    print()
