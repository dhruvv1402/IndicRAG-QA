# IndicRAG-QA

*Evidence-grounded cross-lingual question answering for Indic and code-mixed queries.*

A user asks in English, in Hindi, or — most often in practice — in Romanized Hindi-English
code-mix (*"Scholarship ke liye minimum eligibility kya hai?"*). The document that answers them
was published in the other language. This project builds and, more importantly, **measures** a
retrieval-augmented pipeline over a bilingual corpus of Indian government scholarship schemes,
with one hard constraint: an answer must be supported by a retrieved passage, or the system must
say *"Insufficient information available in the provided documents."*

The deliverable is not a chatbot. It is a set of controlled comparisons — lexical vs dense vs
hybrid retrieval, closed-book vs retrieved vs oracle-context generation, four answerability
signals — each reported per language type, with confidence intervals, and with errors attributed
to the component that caused them.

Course project for CSET 346 (Natural Language Processing), Bennett University.

---

## Documents

| Document | Read it for |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | What is being built and why: hypotheses, corpus and dataset specification, the output contract, success criteria |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How it is built: pipeline, model registry, fusion, answerability signals, caching, CLI |
| [`docs/PLAN.md`](docs/PLAN.md) | Order of work: six phases with exit criteria, the full experiment matrix, results tables, risk register |

Suggested reading order: `PRD` → `ARCHITECTURE` → `PLAN`.

---

## Setting up

The machine this was developed on is CPU-only (Intel Iris Xe, no CUDA), 16 GB RAM, with about
32 GB free on the system drive. Models and caches therefore live on a second drive. Run the
environment script **before** the first install or model pull:

```powershell
. .\scripts\dev-env.ps1     # PowerShell -- dot-sourced, so the variables persist
uv sync --group dev
```

```bash
source scripts/dev-env.sh   # Git Bash / POSIX
uv sync --group dev
```

`torch` resolves from the CPU-only wheel index (~200 MB) rather than the default PyPI wheel,
which bundles ~2.5 GB of CUDA libraries this machine cannot use. That routing lives in
`pyproject.toml` under `[tool.uv.sources]`.

The generator is optional and installed separately, because the corpus and retrieval half of the
project must be usable without it:

```bash
uv sync --group dev --extra llm
```

---

## Running it

```bash
indicrag corpus fetch                  # download sources, write the provenance manifest
indicrag corpus extract                # text extraction + Devanagari integrity validation
indicrag corpus segment                # -> data/passages.jsonl (frozen v1, 694 passages)
indicrag corpus stats                  # counts by scheme and language, length histogram

indicrag index lexical                 # fit TF-IDF and BM25
indicrag index build --all             # encode passages with every registered encoder

indicrag ask "Scholarship ke liye minimum eligibility kya hai?"

indicrag eval retrieval --report evals/report-retrieval.txt
indicrag eval all --report evals/      # corpus, retrieval, errors, answerability reports
indicrag eval qa --split test --sample 72 --nli --gguf <qwen2.5-3b.gguf>   # Module 4
python scripts/regenerate-reports.py --check   # every committed report, diffed against evals/
```

Every `eval` subcommand takes `--no-model` (skip the generator entirely), `--rng-seed`, and
`--report <path>`. Reports are written through the same formatter that prints to the console, so
what is committed under `evals/` is byte-identical to what was shown.

---

## What has been measured

On the gold set: 393 items (313 answerable, 80 unanswerable), **verified by a language
model, not by a person** -- see "Status" below. Every figure here is from the sealed test
split (216 answerable, 273 in all), scored once, with tuning on the 120-item dev split.
Every report under `evals/` carries the same disclosure in its banner.

**Lexical retrieval does not cross the script boundary.** BM25 Recall@5 is 0.970 on
monolingual queries and 0.153 cross-lingual: a Devanagari question and an English passage
share essentially no tokens. multilingual-e5-base reaches 0.661 cross-lingual and is the
best single retriever overall (0.720 against BM25's 0.581, p=0.0001).

**Plain rank fusion throws that away, and a one-line correction restores it.** Reciprocal
Rank Fusion of BM25 and e5 scores 0.274 cross-lingual, less than half of e5 alone. BM25 can
only return passages in the query's script, so under RRF a cross-script passage is docked a
full vote for the lexical retriever's *blindness* rather than its own irrelevance.
`script_aware_rrf` scores each passage over the retrievers eligible to return it: 0.685
cross-lingual (+0.411 over plain RRF, p=0.0001), 0.750 overall (+0.132, p=0.0001), and no
monolingual cost (0.964 under both).

**Against e5 alone, the corrected hybrid gains little.** +0.030 overall (p=0.052), and
significant only on code-mixed queries (+0.085, p=0.004), where Romanized questions carry
English scheme names that BM25 can match. The finding is that standard fusion does harm and
the correction removes it, not that fusion is a large win. The alpha sweep finds no
weighting better than pure dense (best alpha 0.1 at 0.720 against 0.720).

**Indic pretraining is not retrieval training.** MuRIL, pretrained on 17 Indian languages,
reaches 0.048 cross-lingual with mean pooling and 0.024 with CLS; the pooling choice does not
explain it.

**The lexical score is a usable first-stage abstention signal; the fused score is not.** A
BM25 top-score threshold fitted on dev reaches unanswerable F1 0.525 on test (AUC 0.791),
catching every out-of-scope question but only 0.522 of near-misses -- questions about a
scheme the corpus covers, asking for a fact it does not state. The script-aware RRF score
carries almost no signal (medians 0.0328 against 0.0325), because rank fusion discards score
magnitudes.

**Retrieval-augmented generation works; measured so far only on unverified items.**
Module 4 compares four arms -- closed-book, dense RAG, script-aware RAG, and an oracle
given the gold passage -- with one generator (Qwen2.5-3B, 4-bit). On a 72-item sample of
the unverified set, citation support rose from 0.162 closed-book to 0.762 with retrieval
(+0.593, p=0.0001), the choice of retriever did not show up downstream, and generation
error (0.597) was three times retrieval error (0.193). The re-run on the sealed test split
is 186 of 288 generations in. Until it finishes, `evals/report-qa.txt` and
`report-answerability-signals.txt` hold the unverified run; the second says so in its
banner, and the first predates banners and does not.

See `evals/report-retrieval-test.txt`, `evals/report-fusion-test.txt`,
`evals/report-answerability-bm25.txt`, `evals/report-answerability-signals.txt`,
`evals/report-qa.txt`, `evals/report-errors-test.txt` and `evals/report-observed-votes.txt`. The earlier 180-probe results
(`report-retrieval-probes.txt`, `report-script-aware-fusion.txt`) are where the fusion
mechanism was found; where they disagree with the gold set, the gold set stands.

## Status

All six phases have run. `docs/PLAN.md` §10 carries the phase table and what is
not built, and that list is kept honest.

**The gold set was verified by a model, not by a person.** The owner asked for the
verification to be completed without them. Two independent model passes read every item
against its evidence; every item records `annotator: model:...`, and every report says so.
The blind second pass is also a model instance (kappa 1.000 over 59 items), which measures
self-consistency rather than the inter-annotator agreement PRD §6.5 asks for. A person
confirming at least the test split is the most valuable piece of outstanding work;
`indicrag dataset verify` records a human annotator item by item, and the banners will then
report the mix.

The paper (`paper/paper.md`, IEEE `.docx` and `.pdf`), the slide deck and every figure are
built by script from committed data -- `scripts/build-paper.py`, `build-figures.py`,
`build-docx.py`, `build-pptx.py` -- so a figure cannot quietly disagree with the table
beside it. `python scripts/regenerate-reports.py --check` regenerates every committed report
and diffs it.

## Attribution

Source documents are published by the Government of India and its ministries, generally under
GODL-India or equivalent permissive terms. This repository commits derived passages and a
provenance manifest with source URLs, not the original PDFs. Each scheme is attributed to its
issuing ministry in `data/corpus_manifest.jsonl`.

This system reports what those documents say. It is not a source of financial, legal or
eligibility advice.
