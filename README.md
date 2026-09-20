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
indicrag corpus segment                # -> data/passages.jsonl
indicrag corpus stats                  # counts by scheme and language, length histogram

indicrag index lexical                 # fit TF-IDF and BM25
indicrag index build --all             # encode passages with every registered encoder

indicrag ask "Scholarship ke liye minimum eligibility kya hai?"

indicrag eval retrieval --report evals/report-retrieval.txt
indicrag eval all --report evals/      # regenerate every table from cache
```

Every `eval` subcommand takes `--no-model` (skip the generator entirely), `--rng-seed`, and
`--report <path>`. Reports are written through the same formatter that prints to the console, so
what is committed under `evals/` is byte-identical to what was shown.

---

## Status

Early. `docs/PLAN.md` §10 lists what is not built yet, and that list is kept honest.

## Attribution

Source documents are published by the Government of India and its ministries, generally under
GODL-India or equivalent permissive terms. This repository commits derived passages and a
provenance manifest with source URLs, not the original PDFs. Each scheme is attributed to its
issuing ministry in `data/corpus_manifest.jsonl`.

This system reports what those documents say. It is not a source of financial, legal or
eligibility advice.
