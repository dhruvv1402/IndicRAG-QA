# PLAN — IndicRAG-QA

*The order of work, the experiments to run, the tables to fill, and what can go wrong.*

`docs/PRD.md` states what is being built and fixes the dataset numbers. `docs/ARCHITECTURE.md` states how it is built. This document states in what order, with an explicit exit criterion for each phase, so that "am I ready to move on" is a checkable question rather than a feeling.

**Schedule anchor: Week 1 begins Monday 2026-09-22.** The six phases span six weeks. This anchor is provisional — re-pin every date against the actual submission deadline before starting, and if the deadline gives fewer than six weeks, cut from the "optional" column in §1.8 rather than compressing P3 (dataset) or P5 (analysis), which are the two phases that cannot be rushed without invalidating the results.

---

## 1. Phases

Each phase lists its tasks, its dependencies, an effort estimate, and an **exit criterion** — the thing that must be demonstrably true before the next phase starts.

### 1.1 P0 — Environment and scaffold · Week 1, days 1–2 · ~6 h

| # | Task |
|---|---|
| 0.1 | Run `scripts/dev-env.ps1` to redirect `HF_HOME`, `UV_CACHE_DIR` and `INDICRAG_DATA_DIR` off the system drive **before any download**. With 31.6 GB free, this is a gate, not a nicety |
| 0.2 | `uv init`; `pyproject.toml` with hatchling, Python 3.10, ruff (E,F,I,UP,B,SIM; line-length 110), pytest with `pythonpath = ["src"]`, and the `indicrag` CLI entry point |
| 0.3 | Install CPU-only torch explicitly: `--index-url https://download.pytorch.org/whl/cpu` (~200 MB, against ~2.5 GB for the CUDA build this machine cannot use) |
| 0.4 | Install `transformers`, `sentence-transformers`, `rank_bm25`, `scikit-learn`, `pymupdf`, `indic-nlp-library`, `indic-transliteration`, `llama-cpp-python`, `typer`, `pydantic-settings` |
| 0.5 | Scaffold the package tree from ARCHITECTURE §2; `config.py` and `models.py` real, everything else a stub |
| 0.6 | **Benchmark the hardware.** Encode 200 dummy passages with E5-base; generate 80 tokens with Qwen2.5-1.5B-Q4 and 3B-Q4. Record measured tok/s |
| 0.7 | Replace every estimate in ARCHITECTURE §19 with the measured number, and re-check this plan's schedule against it |
| 0.8 | `.gitignore`, `.gitattributes` (`eol=lf` for `*.sh`, `*.yml`; binary for `*.npy`, `*.pkl`), `.env.example`, `git init` |

**Exit criterion:** `indicrag --help` runs; `pytest` passes on an empty suite; `ruff check src tests` is clean; the measured throughput table is written into ARCHITECTURE §19; total disk used by models and caches is confirmed to be off the system drive.

**If 0.6 shows the 3B model below ~6 tok/s**, the full 4-arm × 3B sweep (~5 h) becomes ~10 h. Decide now, not in week 5: either run the 3B arm on a 150-item subsample, or drop the 3B arm and make 1.5B primary. Record the decision and its reason.

---

### 1.2 P1 — Corpus · Week 1 day 3 – Week 2 day 2 · ~14 h

| # | Task |
|---|---|
| 1.1 | Identify 12–20 schemes with **both** an English and a Hindi official document. Record every candidate in `data/corpus_manifest.jsonl` with source URL, ministry, licence, `retrieved_at`, `sha256` |
| 1.2 | Prefer HTML sources over PDF wherever the same content exists in both — it sidesteps the font problem entirely |
| 1.3 | Implement `corpus/extract.py` (PyMuPDF text layer, Tesseract `hin`+`eng` fallback) |
| 1.4 | Implement `corpus/devanagari.py` — the five integrity checks in ARCHITECTURE §3.2 — and run it over every Hindi document |
| 1.5 | **Read 30 randomly sampled Hindi passages by eye.** The automated checks catch systematic corruption; they do not catch subtle loss |
| 1.6 | Implement `corpus/normalize.py` (NFC, digit mapping, danda, ZWNJ, indic-nlp normalizer, currency canonicalization) |
| 1.7 | Implement `corpus/segment.py` — structure-aware split, 120–220 token packing, 25% overlap, tables kept intact, stable `passage_id` |
| 1.8 | `indicrag corpus stats`: counts by scheme and language, token-length histogram, OCR share |

**Exit criterion:** `data/passages.jsonl` holds 800–1500 passages across 12–20 schemes in both languages; every Hindi document passes the integrity check or is recorded as OCR-derived; the 30-passage manual read found no corruption; `corpus stats` shows both languages represented for at least 10 schemes.

**This phase gates everything.** A corrupted corpus produces plausible numbers for weeks before anyone notices. Do not start P2 with a failed integrity check outstanding.

---

### 1.3 P2 — Indexing and retrieval · Week 2 days 3–5 · ~12 h

| # | Task |
|---|---|
| 2.1 | `index/tokenize.py` — script-aware dispatch (English Snowball, Hindi indic-nlp + light suffix stripping, per-token routing for mixed) |
| 2.2 | `index/lexical.py` — TF-IDF cosine and BM25-Okapi (`k1=1.2, b=0.75`) |
| 2.3 | `index/encoders.py` — the five-model registry from ARCHITECTURE §7.1, with E5's `query: `/`passage: ` prefixes applied inside the encoder, never left to the caller |
| 2.4 | `index/dense.py` — batch encode, L2-normalize, cache to `.npy` with a `.meta.json` carrying the passage-file hash; **reject a cache whose hash does not match** |
| 2.5 | Encode all five models (~35–50 min estimated, one-time) |
| 2.6 | `index/hybrid.py` — weighted-α fusion over min-max normalized scores, and RRF at `k=60` |
| 2.7 | `query/langid.py` — Unicode-ratio rules plus the Hindi function-word lexicon; the brief's own example must classify as Code-Mixed |
| 2.8 | `query/normalize.py` and `query/translit.py` — scheme aliases, spelling variants, Roman→Devanagari with a scheme-name exception list |
| 2.9 | `indicrag ask` end to end against a handful of manual queries, before any gold data exists |

**Exit criterion:** all five embedding caches built and validated; `indicrag ask "Scholarship ke liye minimum eligibility kya hai?"` returns a sensible passage with `query_type: Code-Mixed`; a dozen hand-written spot-check queries in all three languages return plausible evidence.

Spot-checking before the gold set exists is deliberate: it catches gross pipeline errors (wrong prefixes, misaligned embeddings, broken tokenizer) while they are cheap, rather than after they have been baked into 400 annotated items.

---

### 1.4 P3 — QA dataset · Week 3 · ~20 h *(the long pole)*

| # | Task |
|---|---|
| 3.1 | `indicrag dataset generate` — bootstrap 2–3 candidate questions per passage with the local model, in the passage's own language |
| 3.2 | Produce Hindi and Hinglish variants. **Hinglish is hand-written or hand-corrected** — machine code-mix is not how people type |
| 3.3 | `indicrag dataset verify` — the review loop. Confirm answerability, correct the gold answer to the document's exact wording for numbers, dates and names, record `gold_passage_ids` (a list — both language versions count) |
| 3.4 | Hand-author the 80 unanswerable items against the four-class taxonomy in PRD §6.3. Write the 30 near-miss items **while looking at the passage they are designed to nearly match** |
| 3.5 | Track coverage against the PRD §6.2 matrix continuously with `dataset stats`; annotation drifts from a target matrix and it is far cheaper to correct at item 150 than at item 380 |
| 3.6 | Second pass on a 15% random sample (60 items) for the `answerable` label; **compute and record Cohen's κ** |
| 3.7 | Compute question↔gold-passage Jaccard overlap; record the distribution; define the low-overlap tertile subset (PRD §6.6) |
| 3.8 | `indicrag dataset split --seed 20260922` — stratified by (query language × answerability × scheme): dev 120, test 280 |

**Exit criterion:** `evals/gold.jsonl` has 400 items, every one `verified: true`; the language-pair matrix matches PRD §6.2 within ±3 per cell; the unanswerable taxonomy matches within ±2 per class; **κ ≥ 0.70**; the overlap distribution is recorded; `evals/splits.json` is written and the test split is sealed.

**If κ < 0.70**, stop. The answerability boundary is not well defined, and every Module 5 number downstream will be measuring annotator inconsistency rather than system behaviour. Tighten the guideline — most often the near-miss/under-specified boundary — re-label the sample, and re-measure.

**Sealing the test split** means exactly that: from here until P5's final run, no metric is computed on it. Every threshold, α, k and prompt is tuned on the 120 dev items.

---

### 1.5 P4 — Generation and answerability · Week 4 · ~16 h

| # | Task |
|---|---|
| 4.1 | `rag/providers.py` — `LlamaCppProvider`, `OpenAICompatProvider`, `ExtractiveProvider` behind one protocol |
| 4.2 | Download Qwen2.5-1.5B-Instruct-Q4_K_M and Qwen2.5-3B-Instruct-Q4_K_M (~3 GB combined) |
| 4.3 | `rag/prompts.py` — the four grounding instructions from ARCHITECTURE §11.3, passages numbered with IDs and `section_path`, best evidence first |
| 4.4 | `rag/grammar.py` — the GBNF grammar. Verify parse failures are **zero** on 50 dev generations with the 1.5B model |
| 4.5 | `rag/generate.py` — arms A (closed-book), B (RAG dense), C (RAG hybrid), D (oracle); disk cache keyed by prompt hash |
| 4.6 | Prompt iteration **on dev only**. Check specifically that Hindi queries receive Hindi answers — small models default to English, and if they do the entire language analysis is meaningless |
| 4.7 | `answerability/signals.py` — threshold, self-report, NLI (mDeBERTa-v3 XNLI, ~1.1 GB) |
| 4.8 | `answerability/calibrate.py` — logistic regression over the eight-feature vector in ARCHITECTURE §12.2, fit on dev |
| 4.9 | `evaluation/grounding.py` — Citation Support Rate, both ROUGE-L and NLI variants |

**Exit criterion:** all four arms run end to end on the 120 dev items with zero JSON parse failures; Hindi and Hinglish queries receive answers in the right language at a verified rate above 90% on a 30-item manual check; τ and the regression boundary are fit and recorded; no test-split item has been scored.

---

### 1.6 P5 — Experiments and analysis · Week 5 · ~18 h, much of it unattended

| # | Task |
|---|---|
| 5.1 | Full retrieval sweep on test: 2 lexical × 5 dense × 2 fusion methods × k ∈ {1,3,5,10}, with the α-sweep at its dev-selected value |
| 5.2 | Full QA sweep: 4 arms × 2 model sizes × 400 items. **Run overnight** (~4–5 h for 3B, ~2–3 h for 1.5B). Everything cached |
| 5.3 | Answerability evaluation: 4 signals, per-class and per-query-type breakdowns, confusion matrix, PR curve |
| 5.4 | Grounding metrics across all arms — this is the H3 measurement |
| 5.5 | Bootstrap 95% CIs (1000 resamples, seeded) on every headline number; paired tests between compared systems |
| 5.6 | Repeat the headline retrieval comparison on the **low-overlap subset** (PRD §6.6) and compare against the full-set result |
| 5.7 | Module 6: select and write up ≥15 difficult cases against the §4 taxonomy |
| 5.8 | Optional: reranking arm; optional: API-model arm |
| 5.9 | `indicrag eval all --report evals/` — regenerate every committed report from cache; confirm it reproduces |

**Exit criterion:** every table in §3 is filled; every headline number carries a confidence interval; each of H1–H4 has an explicit supported / not-supported verdict with the evidence beside it; `evals/report-*.txt` committed; `eval all` reproduces them byte-identically from cache.

**A hypothesis that comes out unsupported is a result, not a problem.** H2 in particular may well fail — hybrid fusion does not always beat its best component. Report it, and analyse why, rather than tuning α until it wins.

---

### 1.7 P6 — Paper, slides, demo · Week 6 · ~20 h

| # | Task |
|---|---|
| 6.1 | Write the IEEE paper against the outline in §6 |
| 6.2 | Generate figures: α-sweep curve, PR curve, per-language bar charts, confusion matrix |
| 6.3 | Build the slide deck (§7) |
| 6.4 | Root `README.md`: what it is, how to run it, a doc-to-"read it for" table, what works and what does not |
| 6.5 | Rehearse the demo. `indicrag ask` with the 1.5B model, process kept warm; prepare 6–8 queries covering all three languages, one cross-lingual case, and one deliberate unanswerable |
| 6.6 | Optional: the web interface |
| 6.7 | Final repository pass — clean history, working instructions, verify a fresh clone runs |

**Exit criterion:** every item in PRD §11 exists at its stated path; the demo has been run start to finish at least twice without intervention; a fresh clone plus the documented commands reproduces the reported tables.

### 1.8 Effort summary

| Phase | Weeks | Effort | Can be cut? |
|---|---|---|---|
| P0 Environment | 1 (days 1–2) | ~6 h | No |
| P1 Corpus | 1–2 | ~14 h | No — gates everything |
| P2 Indexing | 2 | ~12 h | Encoder count reducible from 5 to 3 |
| P3 Dataset | 3 | ~20 h | Reducible to 300 items (PRD floor), not below |
| P4 Generation | 4 | ~16 h | 3B arm droppable |
| P5 Experiments | 5 | ~18 h | Reranking and API arms droppable |
| P6 Write-up | 6 | ~20 h | No |
| **Total** | **6 weeks** | **~106 h** | |

Cut order if the schedule compresses: reranking arm → API-model arm → 3B generator arm → encoders 5→3 (keep E5, LaBSE, MuRIL — MuRIL stays because H6 needs it) → dataset 400→300. **Never** cut the Devanagari validation, the second-pass κ, the dev/test discipline, or the confidence intervals — those are what make the results trustworthy rather than decorative.

---

## 2. Experiment matrix

**Every run enumerated in advance, so that "which experiments are left" is a lookup rather than a recollection.**

### 2.1 Split discipline

| Split | n | Use |
|---|---|---|
| **dev** | 120 | Tuning α, τ, k, prompts, pooling choice, calibration. Look freely |
| **test** | 280 | Scored **once**, at P5. Sealed from the end of P3 |

Stratified by (query language × answerability × scheme), seed `20260922`, recorded in `evals/splits.json`.

### 2.2 Retrieval runs (M1, M2, M3)

| Dimension | Values | n |
|---|---|---|
| Lexical | TF-IDF, BM25 | 2 |
| Dense | E5-base, MiniLM-L12, LaBSE, MuRIL, IndicBERT | 5 |
| Fusion | weighted-α (dev-selected), RRF k=60 | 2 |
| **Systems** | 2 + 5 + 2 = | **9** |
| k | 1, 3, 5, 10 | 4 |
| Slices | all, EN→EN, HI→HI, EN→HI, HI→EN, Hinglish→EN, Hinglish→HI, monolingual, cross-lingual, code-mixed | 10 |

Plus, on dev only: the α-sweep (11 points), MuRIL and IndicBERT with mean vs CLS pooling (4 runs), and the transliteration ablation for BM25 on code-mixed queries (H5).

### 2.3 Generation runs (M4)

| Dimension | Values |
|---|---|
| Arm | A closed-book · B RAG-dense · C RAG-hybrid · D oracle |
| Generator | Qwen2.5-1.5B-Q4, Qwen2.5-3B-Q4 |
| Items | 400 |
| **Total generations** | 4 × 2 × 400 = **3200**, cached |

Optional: `ExtractiveProvider` as a fifth non-neural arm (free, and a genuinely useful floor); an API model as a sixth.

### 2.4 Answerability runs (M5)

| Dimension | Values |
|---|---|
| Signal | threshold τ · generator self-report · NLI entailment · calibrated LR |
| τ sweep (dev) | 20 points across the observed similarity range |
| Report slices | overall; per unanswerable class (4); per query type (3) |

### 2.5 Statistics

Bootstrap 95% CI, 1000 resamples, seed `20260922`, on every headline number. Paired bootstrap between: BM25 vs best dense (H1) · best single vs best hybrid (H2) · arm A vs arm C on grounding (H3) · monolingual vs code-mixed within each system (H4).

---

## 3. Results tables to fill

**Pre-shaped so that analysis is a fill-in rather than a redesign.** One table per required module. Every cell gets a number and a bootstrap interval; every table gets a prose paragraph stating what it does and does not prove.

### 3.1 M1 — Lexical vs dense retrieval

| System | R@1 | R@5 | R@10 | P@5 | MRR | nDCG@10 | s/query |
|---|---|---|---|---|---|---|---|
| TF-IDF | | | | | | | |
| BM25 | | | | | | | |
| MiniLM-L12 | | | | | | | |
| E5-base | | | | | | | |
| LaBSE | | | | | | | |
| MuRIL (best pooling) | | | | | | | |
| IndicBERT (best pooling) | | | | | | | |

### 3.2 M2 — Query type × evidence language (Recall@5)

| System | EN→EN | HI→HI | EN→HI | HI→EN | Hing→EN | Hing→HI | Mono | Cross | Code-mix |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | | | | | | | | | |
| BM25 + translit | | | | | | | | | |
| E5-base | | | | | | | | | |
| LaBSE | | | | | | | | | |
| Best hybrid | | | | | | | | | |

### 3.3 M3 — Lexical vs dense vs hybrid

| System | R@5 (all) | R@5 (cross) | R@5 (code-mix) | MRR | Δ vs best single |
|---|---|---|---|---|---|
| Best lexical | | | | | — |
| Best dense | | | | | — |
| Hybrid α=__ (dev-selected) | | | | | |
| Hybrid RRF k=60 | | | | | |
| + reranker *(optional)* | | | | | |

Accompanied by the α-sweep curve (dev), with the endpoints marked.

### 3.4 M4 — Direct LLM vs RAG

| Arm | Model | EM | token-F1 | Semantic sim | Correctness (judged) | Citation Support Rate |
|---|---|---|---|---|---|---|
| A closed-book | 1.5B / 3B | | | | | |
| B RAG dense | 1.5B / 3B | | | | | |
| C RAG hybrid | 1.5B / 3B | | | | | |
| D oracle | 1.5B / 3B | | | | | |

`D − C` is retrieval error. `1 − D` is generation error. That decomposition is the table's whole purpose.

### 3.5 M5 — Answerability and hallucination

| Signal | Acc | P (UNANS) | R (UNANS) | F1 (UNANS) | Over-abstention |
|---|---|---|---|---|---|
| Threshold τ | | | | | |
| Generator self-report | | | | | |
| NLI entailment | | | | | |
| Calibrated LR | | | | | |

**Recall by unanswerable class** — the table that matters most, because a good aggregate can hide total failure on near-misses:

| Signal | Out-of-scope (25) | Near-miss (30) | False premise (15) | Under-specified (10) |
|---|---|---|---|---|
| Best signal | | | | |

**Abstention rate by query type** — guards the failure in ARCHITECTURE §12.3, where a global threshold refuses Hinglish users because their scores run systematically lower:

| Query type | Abstention rate | Over-abstention rate |
|---|---|---|
| English | | |
| Indic | | |
| Code-Mixed | | |

Plus a 2×2 confusion matrix and the PR curve across τ.

### 3.6 M6 — Error analysis

Rendered from `evals/errors.jsonl`; see §4.

### 3.7 Sanity tables

**Leakage check** (PRD §6.6) — if the dense-vs-lexical gap collapses on the low-overlap subset, the headline result was an annotation artefact:

| Subset | BM25 R@5 | Best dense R@5 | Gap |
|---|---|---|---|
| Full test set (280) | | | |
| Low-overlap tertile | | | |

**Annotation quality:** Cohen's κ on the 60-item second pass; question↔passage overlap distribution; OCR-derived share of Hindi passages.

---

## 4. Module 6 — error analysis protocol

**At least 15 cases, selected against a fixed taxonomy rather than picked for being interesting.** Selecting cases after seeing which ones look good is how an error analysis becomes an anecdote.

Selection rule: for each category below, take the highest-scoring failure — the case where the system was most confident and most wrong. Confident failures are more informative than borderline ones, and they are the ones a user would actually be harmed by.

| # | Category | What it probes |
|---|---|---|
| 1 | Code-mixing | Romanized Hindi with English technical terms |
| 2 | Transliteration variation | `Yasasvi / Yashasvi / यशस्वी` |
| 3 | Spelling variation | Misspelled English terms in a Hinglish query |
| 4 | Named entities | Scheme and ministry names, acronyms |
| 5 | Numbers and amounts | `Rs. 3,50,000` vs `₹3.5 lakh` vs Devanagari digits |
| 6 | Dates and deadlines | Relative dates, "as notified annually" |
| 7 | Ambiguity | Question answerable for one scheme, not another |
| 8 | Near-duplicate passages | Same clause across several schemes with different numbers |
| 9 | Cross-lingual gap | Evidence exists only in the other language |
| 10 | Devanagari extraction artefacts | Where §3.2 validation passed but text is degraded |
| 11 | Long-tail schemes | Schemes with few passages |
| 12 | Language mismatch in output | Hindi query answered in English |
| 13 | Over-abstention | Answerable question refused |
| 14 | Under-abstention | Near-miss unanswerable confidently answered |
| 15 | Citation mismatch | Answer correct, cited passage wrong |

Each case recorded in `evals/errors.jsonl`:

```json
{
  "case_id": "E07",
  "category": "ambiguity",
  "query": "...",
  "query_type": "Code-Mixed",
  "gold_answer": "...",
  "gold_passage_ids": ["..."],
  "retrieved": { "bm25": ["..."], "e5": ["..."], "hybrid": ["..."] },
  "generated": { "arm_c": "...", "arm_d": "..." },
  "answerability": { "predicted": "ANSWERABLE", "gold": "UNANSWERABLE", "confidence": 0.71 },
  "diagnosis": "Prose: what went wrong and why",
  "failing_component": "answerability",
  "would_fix": "Prose: what change would address it"
}
```

`failing_component` ∈ {`extraction`, `segmentation`, `query_processing`, `retrieval`, `generation`, `answerability`, `annotation`} — and `annotation` is a real option. Some error cases turn out to be gold-label mistakes, and recording that honestly is more valuable than quietly correcting it.

Rendered to `evals/report-errors.txt` by `indicrag eval errors --report`.

---

## 5. Risk register

| # | Risk | Trigger / early signal | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **Devanagari extraction corruption** | Integrity check fails, or the 30-passage manual read finds garbled text | Fatal — poisons index and gold answers, silently | Prefer HTML sources; PyMuPDF over alternatives; five automated checks (ARCH §3.2); mandatory human read; OCR fallback; record OCR share in the paper |
| R2 | **Scanned image-only PDFs** | PyMuPDF returns near-empty text | Medium | Tesseract `hin`+`eng`; if OCR quality is poor, substitute another scheme — there are more candidates than needed |
| R3 | **No Hindi version of a chosen scheme** | Found during P1 manifest build | Medium — weakens the bilingual corpus | Over-collect candidates in 1.1; require both languages before committing a scheme |
| R4 | **CPU throughput below estimate** | P0 task 0.6 benchmark | High — P5 schedule doubles | Decide at P0: 3B on a subsample, or drop the 3B arm. Never discover this in week 5 |
| R5 | **Disk exhaustion** (31.6 GB free) | Download fails mid-model | High | `dev-env` redirect as a P0 gate; CPU-only torch wheel; check free space before each model pull |
| R6 | **MuRIL / IndicBERT underperform badly** | Visible in the first P2 spot-check | **None — this is H6** | Report it as a finding. Evaluate both pooling strategies so it cannot be dismissed as a pooling artefact |
| R7 | **Question leakage inflates BM25** | High question↔passage Jaccard overlap in 3.7 | High — invalidates the headline H1 claim | Paraphrase-instructed generation; publish the overlap distribution; report the low-overlap-subset result alongside the full-set result |
| R8 | **Single-annotator bias** | κ < 0.70 on the second pass | High for M5 | 15% second pass with κ reported; stop and tighten guidelines if κ is low; state the limitation in the paper regardless |
| R9 | **`llama-cpp-python` install fails on Windows** | P0 task 0.4 | Medium | Prebuilt CPU wheels exist for Python 3.10/Windows. Fallbacks: `ctransformers`, or a local `llama.cpp` server behind `OpenAICompatProvider` |
| R10 | **Small model answers Hindi queries in English** | P4 task 4.6 manual check | High — makes the language analysis meaningless | Explicit language instruction in the prompt; verify on 30 dev items; if it persists, add a language-consistency post-check and report the rate |
| R11 | **Malformed JSON from the 1.5B model** | P4 task 4.4 | Medium — silently biases results toward easy items | GBNF constrained decoding (ARCH §11.4); assert zero parse failures on 50 dev generations before proceeding |
| R12 | **Dataset drifts from the §6.2 matrix** | `dataset stats` during P3 | Medium — weakens M2 | Check coverage continuously, not at the end; correcting at item 150 is cheap, at item 380 it is not |
| R13 | **Test-split contamination** | Any metric computed on test before P5 | Fatal to the paper's credibility | Seal the split at end of P3; all tuning on dev; state the discipline in the paper |
| R14 | **Government documents revised mid-project** | `sha256` mismatch on re-fetch | Low | Snapshot once, record `sha256` and `retrieved_at`, do not re-fetch mid-project |
| R15 | **Schedule compression** | Deadline earlier than six weeks | Medium | Cut in the order given in §1.8; never cut validation, κ, split discipline, or confidence intervals |

---

## 6. Paper plan

**IEEE conference format, two columns, 6–8 pages.** Each section names the experiment that feeds it, so writing is assembly rather than invention.

| Section | Content | Fed by |
|---|---|---|
| Abstract | Problem, approach, headline numbers, main finding | §3 tables |
| I. Introduction | Motivation, the code-mixed access gap, contributions as a numbered list | PRD §1 |
| II. Related Work | Multilingual dense retrieval; Indic NLP (MuRIL, IndicBERT, IndicNLP); RAG; answerability and abstention; code-mixed IR | Literature |
| III. Dataset | Corpus construction, bilingual property, the §6.2 matrix, the unanswerable taxonomy, annotation protocol, κ, leakage control | PRD §5–6 |
| IV. System | Pipeline, segmentation, the three retrieval families, fusion, the grounded generator, the four answerability signals | ARCH §1–12 |
| V. Experimental Setup | Splits, metrics, hardware, model registry, statistical protocol | ARCH §7, §19; PLAN §2 |
| VI. Results | M1–M5 tables with confidence intervals; explicit verdict on H1–H4 | §3.1–3.5 |
| VII. Error Analysis | The 15+ cases by category, with the component attribution | §4 |
| VIII. Discussion | What the numbers support; the MuRIL negative result; where hybrid helps and where it does not | §3, §4 |
| IX. Limitations | One language, one domain, 400 items, single annotator, small generator, no fine-tuning | PRD §10.3, §14 |
| X. Conclusion | Findings and future work | |

### 6.1 Figures

1. Pipeline diagram (from ARCH §1).
2. Recall@5 by query type × system — the H1/H4 figure.
3. α-sweep curve on dev, endpoints marked — the H2 figure.
4. Citation Support Rate by arm — the H3 figure.
5. Precision-recall curve for answerability across τ.
6. Confusion matrix for the best answerability signal.
7. Question↔passage overlap distribution — the leakage-control figure.

Figure 7 is unusual in a student paper and is worth keeping: showing the artefact you controlled for is more persuasive than claiming there wasn't one.

### 6.2 Contributions, as claimed

1. A bilingual (Hindi/English) evidence-grounded QA dataset over Indian scholarship schemes, with an explicit query-language × evidence-language matrix and a four-class unanswerable taxonomy.
2. A controlled comparison of lexical, five multilingual dense encoders, and two hybrid fusion methods across monolingual, cross-lingual and code-mixed queries.
3. A decomposition of end-to-end QA error into retrieval and generation components via an oracle-context arm.
4. An evaluation of four answerability signals, including per-unanswerable-class recall and per-language abstention rates.
5. A negative result on off-the-shelf Indic MLM checkpoints for retrieval, with the pooling confound controlled.

---

## 7. Slide outline (~12 slides)

1. Title — the problem in one line, with the Hinglish example query.
2. The access gap — one language published, another asked; why keyword search fails on all three mismatches.
3. Research question and the four hypotheses.
4. Corpus — bilingual by construction; counts.
5. Dataset — the language-pair matrix and the unanswerable taxonomy.
6. System — the pipeline diagram.
7. M1 — lexical vs dense.
8. M2 — the per-query-type breakdown; the code-mixed penalty.
9. M3 — hybrid, and the α-sweep.
10. M4 — RAG vs closed-book vs oracle; the error decomposition.
11. M5 — answerability, with the per-class recall table.
12. Error analysis — three representative cases.
13. Findings, limitations, and a live demo.

---

## 8. Demo plan

**The CLI is the deliverable; the web interface is a convenience.**

`indicrag ask` with the 1.5B model and a warm process — roughly 4–6 s per query, acceptable live. Prepared queries:

1. English, answerable, monolingual.
2. Hindi (Devanagari), answerable, monolingual.
3. **Hinglish** — the brief's own example, *"Scholarship ke liye minimum eligibility kya hai?"*
4. Cross-lingual — Hindi query, English evidence, showing the citation is an English passage.
5. Near-miss unanswerable — the abstention, with the retrieved-and-rejected passages shown.
6. A named-entity case where the scheme alias table earns its place.

Each shows the full output object: detected type, answerability, answer, confidence, citation with `doc_id`, and the explanation. Have cached outputs available as a fallback in case the machine is slow on the day, and say so if the fallback is used.

---

## 9. Definition of done

| Deliverable | Done when |
|---|---|
| **Source code** | Fresh clone plus documented commands reproduces every committed table; `ruff check` clean; `pytest` green; `README.md` maps each doc to "read it for" |
| **Dataset** | 400 verified items; matrix within ±3 per cell; κ ≥ 0.70 recorded; `corpus_manifest.jsonl` committed with licences and hashes |
| **Results** | `evals/report-*.txt` committed; every headline number has a CI; H1–H4 each have an explicit verdict |
| **Error analysis** | ≥15 cases across ≥10 categories in `evals/errors.jsonl`, each with a diagnosis and a component attribution |
| **Paper** | IEEE format, Word and PDF, all figures, limitations section written honestly |
| **Slides** | Deck complete; talk rehearsed |
| **Demo** | Run start to finish twice without intervention; fallback outputs prepared |

---

## 10. Status and what is not built yet

**Updated 2026-09-20.** P0–P2 are done, P3 is partly done, P4–P6 have not started.

| Phase | State |
|---|---|
| **P0** Environment | Done. CPU-only stack, caches redirected to G:, measured throughput written into ARCHITECTURE §19 |
| **P1** Corpus | Done, with a documented source change (PRD §5.4). 40 schemes × EN/HI = 80 documents → 694 passages, 80/80 passing integrity validation |
| **P2** Indexing | Done. TF-IDF, BM25, four dense encoders, two fusion methods, all evaluated. First results in `evals/report-retrieval-probes.txt` |
| **P3** Dataset | **Partly done.** Infrastructure complete and tested; 80/80 unanswerable items scaffolded; 0/320 answerable items generated |
| **P4** Generation | Not started |
| **P5** Experiments | Not started (probe-based retrieval results exist, but they are not gold-set results) |
| **P6** Write-up | Not started |

### 10.1 The P3 blocker

The 320 answerable items need a generator, and the generator is not running yet. Two independent obstacles, both environmental rather than design:

1. **`llama-cpp-python` will not build here.** No prebuilt wheel exists for this Python/platform, so it compiles. Three attempts failed: MSVC's environment was not initialised; then, once `vcvars64` was sourced and MSVC 19.44 was correctly detected, CMake resolves to MinGW's CMake 4.0 against an MSVC/Ninja toolchain and the compiler ABI check fails on the mismatch. The backend was switched to `transformers`, which needs no compiler and was already installed for the MuRIL arm.
2. **The model download is crawling.** HuggingFace is serving at roughly 250 KB/s with intermittent stalls, so the 3.1 GB `Qwen2.5-1.5B-Instruct` snapshot has not landed. This is transient — the 4.5 GB of encoders downloaded quickly earlier in the same session.

Everything downstream is ready and tested against a scripted stub. When the weights land:

```bash
indicrag dataset generate --merge-into evals/gold.jsonl   # 320 answerable candidates
indicrag dataset stats                                    # confirm matrix coverage
indicrag dataset verify --annotator a1                    # the human pass, resumable
indicrag dataset split                                    # seals the test split
```

### 10.2 Cost of the backend change

Switching from llama.cpp to `transformers` loses **GBNF grammar-constrained decoding**, which ARCHITECTURE §11.4 calls the highest-value detail in the generation stage. The concern was never malformed JSON in itself — it is that unparseable generations are not a random sample. They skew towards longer, messier passages, so dropping them silently biases the dataset towards easy ones.

`TransformersProvider` mitigates rather than ignores this: it extracts the first balanced JSON object from whatever the model emits, retries once with a stricter instruction, and **counts every failure**, so `parse_failure_rate` is reported alongside the dataset. A measured bias is a caveat that belongs in the limitations section; an unmeasured one is a flaw.

### 10.3 Still not built

Items already expected to remain unbuilt at submission, recorded here so they are not mistaken for oversights:

- **Reranking arm** — first cut if the schedule slips.
- **API-model comparison** — requires a key that does not exist on this machine; bonus work only.
- **Web demo interface** — the CLI is the deliverable.
- **Fine-tuning of any component** — PRD §4 explains why.
- **Languages beyond Hindi** — PRD §14.
- **Multi-hop question answering** — excluded at annotation time.
- **A Hindi morphological analyser** — the light suffix stripper is a known approximation, and its cost to BM25 recall on Hindi should be noted when that number is read.
