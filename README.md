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

## What has been measured

Preliminary, on synthetic probes rather than the human-verified gold set, and
labelled as such in every report under `evals/`. Three findings so far:

**Lexical retrieval does not cross the language boundary at all.** BM25 Recall@5
is 0.740 monolingual but 0.006 cross-lingual and 0.028 code-mixed. A Romanized or
Devanagari query and an English passage share essentially no tokens.

**Naive hybrid fusion makes that worse, not better.** Plain RRF lost every one of
the 8 error cases where dense retrieval alone had found the gold passage. BM25
can only return passages sharing a script with the query, so in rank fusion a
cross-script passage is docked roughly 2:1 for the *lexical retriever's
blindness* rather than its own irrelevance. `script_aware_rrf` scores each
passage over the systems eligible to retrieve it, which takes cross-lingual
Recall@5 from 0.022 to 0.153 (p=0.0001) and code-mixed from 0.028 to 0.173
(p=0.0002), at a real cost of -0.044 monolingual (p=0.032).

**A retrieval-score threshold carries no answerability signal at all here.**
Answerable and unanswerable questions have indistinguishable top scores, and the
unanswerable ones score marginally *higher*: 13.55 against 13.82 under BM25,
0.1424 against 0.1485 under TF-IDF, 0.0327 against 0.0325 under script-aware
RRF. Precision stays at the 0.20 base rate across the entire threshold sweep.
The cause is the taxonomy, not the retriever: 55 of the 80 unanswerable items
are deliberately about schemes that *are* in the corpus, so they retrieve
exactly as well as answerable ones. A retrieval threshold detects corpus
absence, and answerability is not corpus absence.

**Pick the encoder by query script, not by language coverage.** LaBSE was
expected to lead the cross-lingual slice and does not (0.097, below MiniLM at
0.136). It instead dominates code-mixed at 0.332 — 2.5× the next best system,
and above script-aware fusion's own 0.173. On code-mixed queries a better
encoder beats correcting the fusion over a weaker one. MuRIL, meanwhile,
pretrained on 17 Indian languages, scores exactly 0.000 on both cross-lingual
and code-mixed: pretraining-language coverage does not substitute for retrieval
training.

**Retrieval-augmented generation works, and the generator is the bottleneck.**
Four arms over a 72-item stratified sample, one generator throughout, differing
only in the evidence supplied. Citation Support Rate -- the share of answered
questions whose answer the cited evidence supports -- goes 0.162 closed-book to
0.762 with dense retrieval, paired delta +0.593 at p=0.0001. The closed-book arm
answers 37 of 72 and only 16% of those answers are supported: it recites
eligibility thresholds from memory and is usually wrong.

*Which* retriever does not show up downstream. Script-aware fusion against dense
alone is +0.026 citation support (p=1.00) and +0.003 token-F1 (p=0.89), paired
over the questions both arms answered. The unpaired rates look like a win
(0.826 against 0.762) only because the two arms answer different numbers of
questions.

The oracle arm then bounds the whole thing. Generation error is 0.597 against
retrieval error of 0.193, so fixing retrieval entirely would buy less than a
third of what the 3B generator is losing. The fusion result is a claim about
retrieval measured as retrieval; it is not a claim that retrieval is what limits
answer quality here.

See `evals/report-retrieval-probes.txt`, `evals/report-script-aware-fusion.txt`,
`evals/report-qa.txt`, `evals/report-answerability.txt` and
`evals/report-errors.txt`.

## Status

P0–P3 complete, P4 running, P5 blocked, P6 substantially drafted. `docs/PLAN.md`
§10 carries the phase table and what is not built yet, and that list is kept
honest.

**The one real blocker is human verification: 0 of 400 items.** PRD §10.2
excludes unverified items from any reported result, so every number in this
repository is `[PROBE]` and every report says so in its own banner.
`indicrag dataset split` enforces this by refusing to run — it splits verified
items only. That is annotation work, not engineering.

The paper is drafted in full (`paper/paper.md`, §I–§X), with a 14-slide deck, 4
figures and an IEEE `.docx`. All three are built by script from the committed
data — `scripts/build-paper.py`, `build-figures.py`, `build-docx.py` — so a
figure cannot quietly disagree with the table beside it.

## Attribution

Source documents are published by the Government of India and its ministries, generally under
GODL-India or equivalent permissive terms. This repository commits derived passages and a
provenance manifest with source URLs, not the original PDFs. Each scheme is attributed to its
issuing ministry in `data/corpus_manifest.jsonl`.

This system reports what those documents say. It is not a source of financial, legal or
eligibility advice.
