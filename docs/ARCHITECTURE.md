# ARCHITECTURE — IndicRAG-QA

*How the system is built: components, data models, model choices, and the reasons behind each.*

This document describes the design. `docs/PRD.md` states what is being built and fixes the dataset numbers this design assumes; `docs/PLAN.md` states the order of work. Where this document describes something not yet implemented, §21 says so explicitly rather than leaving the reader to guess.

Every non-obvious choice below is accompanied by the constraint or failure it answers to. The dominant constraint throughout is that the development machine has **no CUDA GPU**: an Intel i5-11320H with 15.7 GB RAM, Intel Iris Xe integrated graphics, and 31.6 GB free on the system drive.

---

## 1. Pipeline overview

**The pipeline follows brief §8, with one addition: an explicit error-attribution path.**

```
                    ┌──────────────────────────────────────────────┐
                    │  OFFLINE  (once per corpus revision)          │
                    └──────────────────────────────────────────────┘

  source PDFs / HTML
  (EN + HI, 12-20 schemes)
          │
          ▼
  ┌───────────────┐   ┌────────────────┐   ┌──────────────────┐   ┌──────────────┐
  │ fetch +       │──▶│ extract text   │──▶│ clean & normalize│──▶│ segment into │
  │ manifest      │   │ (+ Devanagari  │   │ (NFC, digits,    │   │ passages     │
  │ (sha, licence)│   │  integrity chk)│   │  danda, ZWNJ)    │   │ 120-220 tok  │
  └───────────────┘   └────────────────┘   └──────────────────┘   └──────┬───────┘
                                                                          │
                          ┌───────────────────────────────────────────────┤
                          ▼                                               ▼
                 ┌──────────────────┐                         ┌────────────────────┐
                 │ LEXICAL INDEX    │                         │ DENSE INDEX        │
                 │ TF-IDF + BM25    │                         │ 5 encoders ->      │
                 │ script-aware tok │                         │ data/emb/<m>.npy   │
                 └──────────────────┘                         └────────────────────┘

                    ┌──────────────────────────────────────────────┐
                    │  ONLINE  (per query)                          │
                    └──────────────────────────────────────────────┘

   user query
       │
       ▼
  ┌─────────────────────┐
  │ query processing    │  script/lang ID -> English | Indic | Code-Mixed
  │ langid, normalize,  │  optional Roman->Devanagari transliteration
  │ transliterate       │
  └──────────┬──────────┘
             │
             ├──────────────▶ lexical retrieve ─┐
             │                                   ├──▶ ┌──────────┐
             └──────────────▶ dense retrieve ────┘    │  FUSION  │ weighted-alpha | RRF
                                                       └────┬─────┘
                                                            │ top-k passages
                                            (optional) ┌────▼─────┐
                                                       │ reranker │ cross-encoder, top-20
                                                       └────┬─────┘
                                                            ▼
                                                  ┌───────────────────┐
                                                  │ RAG GENERATION    │  grounded prompt,
                                                  │ llama.cpp + GBNF  │  JSON-constrained
                                                  └────────┬──────────┘
                                                           ▼
                                                  ┌───────────────────┐
                                                  │ ANSWERABILITY     │  4 signals ->
                                                  │ threshold / self- │  calibrated decision
                                                  │ report / NLI / LR │
                                                  └────────┬──────────┘
                                                           ▼
                                             evidence-grounded answer + citations
                                             or "Insufficient information available
                                              in the provided documents."
```

### 1.1 Where errors originate

The reason the evaluation is split by component is that four distinct failures produce the same symptom — a wrong answer:

| Failure | Symptom | Isolated by |
|---|---|---|
| The gold passage was never retrieved | Answer is wrong or abstained | Retrieval Recall@k |
| The gold passage was retrieved but ranked below the cut | Answer is wrong | Recall@k vs Recall@1, MRR |
| Good context retrieved, generator ignored or misread it | Answer is wrong despite correct evidence | **Oracle arm (D)** vs RAG arm (C) |
| Generator answered when it should have abstained | Confident hallucination | Citation Support Rate, answerability F1 |

Arm D (oracle context) is the load-bearing piece here: the gap between arm C and arm D is retrieval error, and arm D's own error rate is generation error. Without it, the two are inseparable.

### 1.2 Required modules to components

**The brief's six required modules map onto sections of this document as follows.** A module with no section is a design gap.

| Module (brief §5) | Components | Sections |
|---|---|---|
| **M1** TF-IDF/BM25 vs multilingual dense retrieval | `index/lexical.py`, `index/dense.py`, `index/encoders.py` | §6, §7 |
| **M2** Monolingual vs cross-lingual vs code-mixed queries | `query/langid.py`, `query/translit.py`, per-slice reporting | §10, §14 |
| **M3** Lexical vs dense vs hybrid retrieval | `index/hybrid.py`, optional `index/rerank.py` | §8, §9 |
| **M4** Direct LLM vs retrieval-augmented QA | `rag/generate.py` arms A/B/C/D | §11 |
| **M5** Hallucination and answerability | `answerability/`, `evaluation/grounding.py` | §12, §13 |
| **M6** Qualitative analysis of difficult cases | `evaluation/errors.py` | §15, and `docs/PLAN.md` §4 |

---

## 2. Package layout

**src-layout, one package, mirroring the conventions of the author's other project (`lawbot-counter`).**

```
IndicRAG-QA/
├── README.md                      maps each doc to "read it for"
├── pyproject.toml                 hatchling, uv, ruff; CLI entry point
├── .env.example                   committed; .env is gitignored
├── docs/
│   ├── PRD.md                     what and why
│   ├── ARCHITECTURE.md            this file
│   └── PLAN.md                    order of work
├── scripts/
│   ├── dev-env.ps1                redirects HF_HOME, UV caches, venv off C:
│   └── dev-env.sh                 POSIX twin of the above
├── src/indicrag/
│   ├── __init__.py                __version__
│   ├── config.py                  pydantic-settings, @lru_cache get_settings()
│   ├── models.py                  Document, Passage, QAItem, Retrieved, Answer
│   ├── cli.py                     Typer app, nested sub-apps
│   ├── corpus/
│   │   ├── fetch.py               download + manifest + sha256
│   │   ├── extract.py             PDF/HTML -> text, OCR fallback
│   │   ├── devanagari.py          integrity validation (see §3.2)
│   │   ├── normalize.py           NFC, digits, danda, ZWNJ, whitespace
│   │   ├── segment.py             structure-aware chunking
│   │   └── store.py               passages.jsonl read/write
│   ├── index/
│   │   ├── tokenize.py            script-aware tokenizers
│   │   ├── lexical.py             TF-IDF and BM25
│   │   ├── encoders.py            model registry + loading
│   │   ├── dense.py               encode, cache, exact search
│   │   ├── hybrid.py              weighted-alpha and RRF fusion
│   │   └── rerank.py              optional cross-encoder
│   ├── query/
│   │   ├── langid.py              English | Indic | Code-Mixed
│   │   ├── normalize.py           query-side cleanup
│   │   └── translit.py            Roman -> Devanagari
│   ├── rag/
│   │   ├── providers.py           LLMProvider protocol + backends
│   │   ├── prompts.py             grounded prompt templates
│   │   ├── grammar.py             GBNF JSON grammar
│   │   ├── schemas.py             generation request/response
│   │   └── generate.py            arms A/B/C/D, disk cache
│   ├── answerability/
│   │   ├── signals.py             threshold, self-report, NLI
│   │   ├── calibrate.py           logistic regression over signals
│   │   └── decide.py              final ANSWERABLE/UNANSWERABLE
│   ├── evaluation/
│   │   ├── gold.py                QAItem I/O, split, summarise
│   │   ├── retrieval.py           Recall/Precision/Hit/MRR/nDCG
│   │   ├── qa.py                  EM, token-F1, semantic sim
│   │   ├── answerability.py       accuracy, P/R/F1, confusion
│   │   ├── grounding.py           Citation Support Rate
│   │   ├── stats.py               bootstrap CIs, paired tests
│   │   ├── errors.py              Module 6 case capture
│   │   └── report.py              formatters -> list[str]
│   └── demo/
│       └── app.py                 optional local web interface
├── evals/
│   ├── gold.jsonl                 the 400-item QA set
│   ├── splits.json                dev/test assignment + seed
│   ├── errors.jsonl               Module 6 cases
│   └── report-*.txt               committed plaintext results
├── tests/                         flat, one file per module
├── data/                          gitignored (see §17)
└── paper/                         IEEE paper, slides
```

Two conventions carried over deliberately: **formatters return `list[str]`** so that console output and `--report <path>` share one code path and the formatting is unit-testable, and **tests are flat with sentence-length names** (`test_devanagari_validator_rejects_reordered_matras`).

---

## 3. Document processing

**Text extraction is the highest-risk stage in the whole system, and it is the one most likely to fail silently.**

### 3.1 Extraction order

1. **HTML source, if one exists.** Government scheme pages often mirror their PDFs. HTML avoids the font problems entirely and is preferred wherever available.
2. **PDF text layer** via PyMuPDF (`fitz`), which is already installed and handles Devanagari better than most alternatives.
3. **OCR fallback** via Tesseract with the `hin` and `eng` traineddata, only for documents that are image-only or that fail the integrity check in §3.2. OCR output is flagged in the manifest so that any downstream anomaly can be traced to it.

### 3.2 Devanagari integrity validation

**A Hindi PDF can extract into a string that renders as plausible Devanagari but is character-level nonsense.** Legacy fonts and subset embeddings produce reordered matras, dropped conjuncts, or characters mapped to the wrong codepoints. This does not raise an error; it produces text that a non-reader would accept. If it enters the index, retrieval quality collapses in a way that looks like a model problem.

`corpus/devanagari.py` runs these checks on every extracted Hindi document and refuses the extraction if any fail badly:

| Check | What it catches |
|---|---|
| Dependent vowel sign (matra, U+093E–U+094C) appearing before any consonant in a cluster | The classic reordering bug |
| Virama (U+094D) in word-final position at an implausible rate | Dropped conjunct joining |
| Ratio of Devanagari codepoints to total non-space characters below a floor | Font mapped to Latin private-use area |
| Proportion of tokens matching a Hindi stopword list | Mojibake that is still technically Devanagari |
| Presence of replacement character U+FFFD | Encoding failure |

A document that fails is routed to OCR and re-checked. **In addition, 30 randomly sampled passages are read by a human before the corpus is frozen** — the automated checks catch systematic corruption, not subtle loss, and no amount of validation code substitutes for reading the text once.

### 3.3 Normalization

Applied to both passages and queries, so that the two sides match:

- Unicode **NFC** composition, applied uniformly.
- **Devanagari digits ↔ ASCII digits** (०–९ → 0–9), with the original preserved in a `raw` field. Amounts are the single most common answer type in this corpus, and a query written with ASCII digits must match a passage written with Devanagari ones.
- **Danda and double danda** (।, ॥) normalized to sentence boundaries for segmentation, retained in the displayed text.
- **ZWJ / ZWNJ** (U+200D / U+200C) stripped except where they are semantically required, since their inconsistent presence otherwise fragments tokens.
- Whitespace collapsed; soft hyphens and PDF line-break artefacts repaired.
- `indic-nlp-library`'s `DevanagariNormalizer` for nukta and variant-form canonicalization.
- Currency and number surface forms normalized to a canonical form for *matching only* (`Rs. 3,50,000`, `₹3,50,000`, `3.5 lakh`, `3,50,000/-` → one key), with the original text displayed.

Normalization is a pure function with no configuration branches at query time; the same code path runs on both sides, which is the only way to guarantee they agree.

---

## 4. Passage segmentation

**Segmentation determines the ceiling on retrieval quality — a fact split across a chunk boundary can never be retrieved intact.**

Strategy, in priority order:

1. **Structure-aware split first.** Government circulars are numbered: clause headings, bullet lists, tabular eligibility criteria. Split on detected headings and clause markers so that a passage corresponds to a self-contained provision.
2. **Length-bounded packing second.** Accumulate structural units into passages of 120–220 tokens, never splitting a unit that fits.
3. **Overlap of ~25%** between adjacent passages where a structural split was not available, so that a fact near a boundary appears whole in at least one passage.
4. **Tables kept intact** where detected, even if they exceed the length target — an eligibility table split in half is worse than a long passage.

Each passage carries:

```
passage_id = f"{doc_id}#p{index:04d}"     stable across re-runs given the same input
doc_id, scheme, lang, section_path, page, char_span, token_count, text, text_raw
```

`section_path` (e.g. `"3. Eligibility > 3.2 Income criteria"`) is carried because it is genuinely useful context for the generator and for the human reading an error case, and because it makes near-duplicate passages across schemes distinguishable in the error analysis.

Passage IDs must be **stable across re-runs**: the error analysis, the gold set and the cached embeddings all reference them, and a re-segmentation that renumbers passages invalidates all three. Re-segmentation therefore requires an explicit corpus version bump, recorded in the manifest.

---

## 5. Data models

**Plain dataclasses with tolerant JSONL serialization.** No ORM, no database — the corpus is under 1500 passages and files are easier to inspect, diff and hand-edit.

| Model | Key fields |
|---|---|
| `Document` | `doc_id, scheme, ministry, lang, title, source_url, retrieved_at, sha256, licence, format, pages, ocr_used, notes` |
| `Passage` | `passage_id, doc_id, scheme, lang, section_path, page, char_span, token_count, text, text_raw` |
| `QAItem` | as specified in PRD §6.4 |
| `Retrieved` | `passage_id, score, rank, method, component_scores` |
| `Answer` | the output contract in PRD §9, plus `latency_ms, model, prompt_hash` |

Each carries `as_dict()` / `from_dict()`. Deserialization is **tolerant**: unknown keys are dropped rather than raising, and blank lines in JSONL are skipped. This exists for one concrete reason — the gold set is hand-edited during annotation, and a schema addition mid-project must not invalidate the file the annotator is halfway through.

---

## 6. Lexical retrieval

**Two lexical baselines, because the brief names both and because they fail differently.**

- **TF-IDF + cosine** (`sklearn.TfidfVectorizer`) — the simpler baseline; sensitive to document length.
- **BM25-Okapi** (`rank_bm25`, or a compact in-repo implementation) with `k1=1.2, b=0.75` — the stronger baseline and the one the headline comparison uses.

### 6.1 Script-aware tokenization

A single tokenizer cannot serve three scripts. `index/tokenize.py` dispatches on detected script:

| Input | Tokenizer | Stemming |
|---|---|---|
| English | lowercase, Unicode word split, English stopwords | Snowball |
| Hindi (Devanagari) | `indic-nlp-library` trivial tokenizer, Hindi stopword list | Light suffix stripping (a full Hindi stemmer is out of scope; the light stripper handles the common case-marker suffixes) |
| Mixed | per-token script detection, each token routed to its own path | as above |

### 6.2 The expected result, stated in advance

A Romanized Hinglish query shares **almost no tokens** with a Devanagari passage. BM25 on that pair should score near zero except where English loanwords (`scholarship`, `income`, `NMMS`) appear in both. This is not a bug to be fixed; it is the measurement that motivates dense retrieval, and H4 predicts it. It is stated here so that the result is read as confirmation rather than as a broken baseline.

### 6.3 The transliteration bridge

To separate *script* mismatch from *semantic* mismatch, an ablation arm transliterates Romanized queries to Devanagari (`indic-transliteration`, ITRANS/HK schemes, with a scheme-name exception list so `NMMS` and `YASASVI` are not mangled) before lexical retrieval. If BM25 recovers substantially on this arm, the code-mixed penalty is largely orthographic; if it does not, the penalty is semantic. This is H5, and it is one of the more interesting numbers the project can produce cheaply.

---

## 7. Dense retrieval

### 7.1 Model registry

**Five encoders, chosen to make a specific comparison rather than to maximise one number.** Sizes are approximate fp32 on-disk footprints.

| Model | Params | Disk | Role | Why it is here |
|---|---|---|---|---|
| `intfloat/multilingual-e5-base` | 278M | ~1.1 GB | **Primary** | Strong cross-lingual retrieval; trained with a retrieval objective. Requires `query: ` / `passage: ` prefixes — omitting them measurably degrades it, so the prefix is applied in `dense.py`, not left to the caller |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | ~470 MB | Speed baseline | Fastest CPU option; establishes what a small model gives up |
| `sentence-transformers/LaBSE` | 471M | ~1.8 GB | Cross-lingual specialist | Trained for bitext mining across 109 languages; the arm most likely to win the EN↔HI cross-lingual slice |
| `google/muril-base-cased` | 236M | ~950 MB | Indic-pretrained | Named in the brief. **An MLM checkpoint, not a sentence encoder** |
| `ai4bharat/indic-bert` | 33M | ~135 MB | Indic-pretrained, small | Named in the brief. Same caveat |

Total encoder footprint ≈ **4.5 GB**, plus a CPU-only torch wheel (~200 MB) and `transformers`. Well inside the 12 GB budget in PRD NFR-3, provided `HF_HOME` is redirected off the system drive (§16).

### 7.2 The MuRIL and IndicBERT caveat

**MuRIL and IndicBERT are masked-language-model checkpoints.** They were pretrained with MLM objectives, not with a contrastive sentence-similarity objective, and they have no pooling layer trained for sentence representation. Mean-pooling their token embeddings produces usable-looking vectors whose similarity structure is dominated by lexical and positional overlap rather than by meaning — the well-documented anisotropy problem in raw BERT-family embeddings.

They are expected to lose, and they are included anyway for three reasons: the brief names them; practitioners do reach for them by name, so the negative result is useful; and the contrast makes the point that *retrieval training matters more than pretraining-language coverage*. Both are evaluated with mean pooling and with CLS pooling, and the better of the two is reported, so that the result cannot be dismissed as a pooling artefact. This is H6.

### 7.3 Index structure

At 800–1500 passages, an **exact** search is the right answer: embeddings are L2-normalized into a single `float32` matrix of shape `(n_passages, dim)`, and search is one matrix-vector product followed by `argpartition`. For 1500 × 768 this is well under a millisecond and is *exactly* correct, where an approximate index would introduce recall error that is indistinguishable from model error in the results.

FAISS is deliberately **not** used. It adds an install dependency that is awkward on Windows, and it buys nothing at this scale. It is noted as the path forward only if the corpus grows past roughly 10⁵ passages.

Embeddings are cached to `data/emb/<model-slug>.npy` alongside a `<model-slug>.meta.json` recording the model revision, normalization settings, passage count and a hash of the passage file. **A cache whose passage hash does not match the current `passages.jsonl` is rejected rather than used** — a stale embedding matrix silently misaligned with the passage list is the kind of bug that produces plausible but meaningless numbers for days.

---

## 8. Hybrid fusion

**Two fusion methods, compared, because they have different failure modes.**

### 8.1 Weighted score fusion

The brief's formula, applied after normalizing each component's scores over the candidate pool:

```
Score(p) = α · norm(Score_lexical(p)) + (1 − α) · norm(Score_dense(p))
```

`norm` is min-max over the union of the two candidate lists. Min-max is chosen over z-score because BM25 score distributions are strongly right-skewed and a z-score leaves the fused ranking dominated by lexical outliers. Both are implemented and the choice is an ablation, since this is exactly the kind of detail that silently changes a headline number.

`α` is swept over `{0.0, 0.1, …, 1.0}` **on the dev split only**. The endpoints α=1 and α=0 recover pure lexical and pure dense retrieval, which means the α-sweep curve doubles as the evidence for or against H2: if no interior α beats both endpoints, H2 is false and the paper says so.

### 8.2 Reciprocal Rank Fusion

```
RRF(p) = Σ_systems 1 / (60 + rank_system(p))
```

Rank-based, so it needs no score normalization at all and is immune to the skew problem above. `k=60` is the standard value and is not tuned, to keep it an honest baseline against the tuned α.

### 8.3 Why both

Weighted fusion can exploit score magnitude when the two systems are well calibrated; RRF is more robust when they are not. Reporting both — and reporting which one wins — is more informative than reporting a single "hybrid" number, and costs almost nothing since the candidate lists are already computed.

---

## 9. Reranking (optional)

A cross-encoder reranker over the top-20 fused candidates, as the brief's bonus "transformer reranking" item:

- `BAAI/bge-reranker-v2-m3` (568M, ~2.2 GB) — multilingual, strong, and the slowest thing in the system on CPU.
- Budget: roughly 2–4 s per query for 20 pairs, therefore ~20–25 minutes for a full 400-query pass. Feasible as a batch arm, **not** feasible in the interactive demo, and the CLI reflects that by leaving it off by default.

This arm is explicitly optional. If the schedule slips, it is the first thing cut, and §21 records it as not built.

---

## 10. Query processing

### 10.1 Language and script identification

**Rule-based, because it is fast, deterministic, explainable, and on this three-class problem it is at least as accurate as a statistical classifier.**

```
devanagari_ratio = |chars in U+0900-U+097F| / |non-space chars|

devanagari_ratio > 0.60                          -> Indic
devanagari_ratio < 0.10 and hindi_marker_hits>=1 -> Code-Mixed
devanagari_ratio < 0.10 and hindi_marker_hits==0 -> English
otherwise                                        -> Code-Mixed (mixed-script)
```

`hindi_marker_hits` counts matches against a closed-class Hindi function-word lexicon written in Latin script — the words that survive code-mixing because they are grammatical rather than lexical:

`kya, hai, hain, ke, ki, ka, ko, se, liye, mein, me, kitna, kitni, kaise, kab, kahan, kaun, nahi, nahin, chahiye, hoga, hogi, karna, milega, wala, aur, ya, par, tak, bhi, sakta, sakte`

This lexicon is the single most important piece of the classifier: a query like *"eligibility criteria kya hai"* is Latin-script and lexically English except for the two Hindi words that determine its type.

A character-n-gram classifier is trained as a **cross-check**, not as the primary path, and disagreements between the two are logged. Disagreement cases are strong candidates for the Module 6 error analysis, since a misdetected query type also means the answer is generated in the wrong language.

The three emitted classes are exactly `English`, `Indic`, `Code-Mixed`, as brief §7 requires.

### 10.2 Query normalization

The same normalization as §3.3 (NFC, digits, punctuation), plus query-specific handling:

- **Scheme-name variants.** `YASASVI / Yashasvi / यशस्वी`, `NMMS / N.M.M.S.`, `Pragati / प्रगति` are mapped to canonical forms through a small hand-built alias table. Named entities are the most common retrieval failure in this corpus and the cheapest to fix.
- **Spelling variants.** Common misspellings of high-frequency English terms (`scholarship / scholership / schlorship`) via an edit-distance match against a corpus vocabulary, applied only to tokens absent from the vocabulary.
- **Transliteration**, when enabled (§6.3), producing a Devanagari variant of the query that is issued to the lexical retriever alongside the original.

### 10.3 What the query type is used for

Three things: selecting the answer language for the generator; slicing every evaluation table by language type (brief §6 requires this); and populating `query_type` in the output contract. It deliberately does **not** switch the retrieval method — using one pipeline for all query types is what makes the per-type comparison meaningful.

---

## 11. RAG generation

### 11.1 Provider interface

```python
class LLMProvider(Protocol):
    name: str
    def complete(self, prompt: str, *, grammar: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.0,
                 seed: int = 0) -> Completion: ...
```

| Backend | Use |
|---|---|
| `LlamaCppProvider` | **Default.** `llama-cpp-python` with a local GGUF file. Offline, free, deterministic at `temperature=0` with a fixed seed |
| `OpenAICompatProvider` | Optional. Any OpenAI-compatible endpoint (Groq, Gemini's compatibility endpoint, a local server) for the "larger instruction-tuned model" bonus comparison |
| `ExtractiveProvider` | No model. Returns the highest-scoring sentence from the top passage. Powers the `--no-model` fast path, makes CI meaningful without a 2 GB download, and serves as a genuine non-neural QA baseline |

### 11.2 Generator models

| Model | Quant | Disk | Role | Throughput (est., this CPU) |
|---|---|---|---|---|
| `Qwen2.5-3B-Instruct` | Q4_K_M | ~2.0 GB | **Primary.** Batch evaluation arm | ~8–12 tok/s |
| `Qwen2.5-1.5B-Instruct` | Q4_K_M | ~1.0 GB | Small-model arm; the interactive demo | ~18–25 tok/s |

Qwen2.5 is chosen over comparable small models for its Devanagari and Hindi handling, which is materially better than most sub-3B alternatives, and because both sizes come from the same family — making the size comparison a clean ablation rather than a confounded one.

Throughput figures are **estimates from the hardware profile, not measurements**. They must be replaced with measured numbers in P0 of the plan; if the 3B model turns out slower than about 6 tok/s, the full sweep is re-budgeted or the 3B arm is reduced to a subsample.

### 11.3 Prompt design

The grounded prompt carries four instructions, each answering a specific observed failure of small models:

1. **Answer only from the numbered passages below.** — against parametric leakage.
2. **Cite the passage ID you used.** — makes grounding checkable, and an uncitable answer is detectable.
3. **Answer in the same language and script as the question.** — small models default to English; without this, Hindi queries get English answers and the language analysis is meaningless.
4. **If the passages do not contain the answer, set `answerable` to false.** — gives the self-report signal something to report.

Passages are presented numbered with their IDs and `section_path`, in retrieval order, with the highest-scoring passage first. Placing the best evidence first matters for small models, which attend unevenly across a long context.

### 11.4 Constrained decoding

**The dominant failure mode of a 1.5B model asked for JSON is malformed JSON.** `llama.cpp` supports GBNF grammars, and `rag/grammar.py` defines one that makes the required object shape the only reachable output:

```
root ::= "{" ws "\"answerable\"" ws ":" ws bool ws ","
             ws "\"answer\"" ws ":" ws string ws ","
             ws "\"citations\"" ws ":" ws idlist ws ","
             ws "\"confidence\"" ws ":" ws number ws "}"
```

With the grammar active, parse failures go to zero by construction. Without it, a meaningful share of 1.5B generations would need a repair pass or would be dropped — and dropping malformed outputs silently biases the results towards the easy questions. This is the single highest-value implementation detail in the generation stage.

### 11.5 Generation arms

| Arm | Context given | What it measures |
|---|---|---|
| **A — Closed-book** | None. Question only | Parametric knowledge; the hallucination baseline for H3 |
| **B — RAG dense** | Top-k from the best dense encoder | Standard RAG |
| **C — RAG hybrid** | Top-k from the best fusion method | The full system |
| **D — Oracle** | The gold passage(s), retrieval bypassed | The generation ceiling; `D − C` is retrieval error, `1 − D` is generation error |

### 11.6 Caching

Every generation is written to `data/gen/<model>/<arm>.jsonl`, keyed by `sha256(prompt + model + params + seed)`. Re-running the evaluation reads the cache and touches no model. This is what makes NFR-6 (one command reproduces every table) affordable: the expensive pass happens once, and analysis iterates freely afterwards.

---

## 12. Answerability

**Four signals, evaluated independently and then combined, so the paper can say which signal actually carries the decision.**

| # | Signal | Computation | Cost |
|---|---|---|---|
| 1 | **Retrieval threshold** | `max_sim < τ` → UNANSWERABLE. τ fit on dev | free |
| 2 | **Generator self-report** | The `answerable` field and `confidence` from the constrained JSON | free (already generated) |
| 3 | **NLI entailment** | Does the generated answer follow from the cited passage? | ~0.3 s/pair on CPU |
| 4 | **Calibrated combination** | Logistic regression over all of the above | negligible |

### 12.1 The NLI check

`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (278M, ~1.1 GB) scores `P(entailment)` for the pair *(cited passage as premise, generated answer as hypothesis)*. It is multilingual, so it handles the cross-lingual case where the answer is Hindi and the passage English — which is exactly the case a monolingual checker would fail on.

This doubles as the brief's bonus **citation verification** item: an answer the NLI model does not consider entailed by its own cited passage is, by definition, an unsupported citation.

### 12.2 Calibrated combination

A logistic regression fit on the dev split over the feature vector:

```
[ max_sim,  top1_score − top2_score,  mean_top_k_sim,
  generator_confidence,  generator_answerable_flag,
  nli_entailment_prob,  answer_token_length,  citation_count ]
```

The `top1 − top2` margin is included because it captures something `max_sim` alone does not: a near-miss unanswerable question typically produces *several* similarly-scoring passages (the topic is present, the fact is not), whereas a genuinely answerable question usually has one clear winner. A small margin with a high max score is the signature of the hardest unanswerable class.

Logistic regression, not a larger model, because there are 120 dev items. Anything with more capacity would memorise them, and the coefficients of a linear model are directly reportable — which signal dominates is itself a finding.

### 12.3 Threshold selection and reporting

τ and the regression decision boundary are chosen on dev by maximising F1 on the UNANSWERABLE class. The test split reports: accuracy, precision, recall, F1, the confusion matrix, the precision-recall curve across τ, **per-unanswerable-class recall** (out-of-scope vs near-miss vs false-premise vs under-specified), and **per-query-type abstention rate**.

That last breakdown guards against a specific and likely failure: if code-mixed queries retrieve at lower similarity scores across the board, a global threshold will abstain on them disproportionately — the system would refuse to answer Hinglish users not because the evidence is missing but because the scores are systematically lower. Reporting abstention rate per query type is what surfaces it. If it appears, a per-language-type threshold is the obvious remedy, and its necessity is itself worth reporting.

---


### 12.4 Signal 1 is measured, and it does not work here

**Added 2026-09-21.** The table above lists the retrieval threshold first
because it is free and standard. On this corpus it carries no information at
all, and the design should not be read as if it does.

Measured over all 400 gold items:

| Retriever | answerable (median) | unanswerable (median) | best F1 | precision at best |
|---|---|---|---|---|
| BM25 | 13.55 | **13.82** | 0.366 | 0.235 |
| TF-IDF | 0.1424 | **0.1485** | 0.370 | 0.231 |
| Script-aware RRF | 0.0327 | 0.0325 | 0.351 | 0.216 |

The classes do not separate, and under two of the three retrievers the
*unanswerable* questions score higher. Across the full τ sweep precision stays
between 0.19 and 0.22 against an unanswerable base rate of 0.20, and the best F1
is reached at 87% abstention.

The cause is the dataset, not the retriever. 55 of the 80 unanswerable items —
near-miss, false-premise and under-specified — are written about schemes that
*are* in the corpus, so they retrieve as well as answerable questions do. A
retrieval-score threshold is a corpus-absence detector; only the 25 out-of-scope
items are corpus-absent, and those it catches at 0.688.

Two consequences for the rest of this section. First, §12.3's instruction to fit
τ by maximising F1 on the UNANSWERABLE class is still right, but the fitted
value is not meaningful in isolation here: it moved from 0.000 recall to 0.93
across two splits of the same data while the underlying curve stayed flat, so
the reporting now always carries the separation table and the full sweep beside
any fitted point.

Second, the rationale in §12.2 for including the `top1 − top2` margin has now
been measured, and it is *directionally right and far too weak to use*.

**Feature separation under BM25, all 400 items.** AUC is P(a random answerable
question scores above a random unanswerable one); 0.500 is chance.

| Feature | answerable | unanswerable | AUC |
|---|---|---|---|
| `score_spread` | 5.2127 | 3.7732 | 0.599 |
| `margin` | 2.1769 | 1.2750 | 0.594 |
| `max_score` | 13.5509 | 13.8179 | 0.545 |
| `mean_top_k` | 10.2447 | 10.3221 | 0.510 |
| `scheme_agreement` | 0.4000 | 0.4000 | 0.504 |

The prediction was that near-miss questions retrieve several similarly-scoring
passages and so show a smaller margin, and they do: 1.28 against 2.18. The
reasoning was sound. But an AUC of 0.594 means the ordering holds on 59% of
pairs where chance is 50%, which is nowhere near enough to threshold on.

**Signal 4 has now been fit over exactly these features**, on the same split:

| Signal | test F1 | abstains on | AUC |
|---|---|---|---|
| 1. threshold | 0.345 | 87.0% | — |
| 4. calibrated | 0.360 | 37.5% | 0.663 |

The prediction that a near-chance feature set cannot be rescued by combining it
held for F1, and understated the rest. Combining does extract more than any
single feature — AUC 0.663 against 0.599 — and it reaches a comparable F1 at a
far more usable operating point, refusing 37% of questions rather than 87%.

The fitted weights are the sharper result: `score_spread` 0.739, `mean_top_k`
−0.350, `margin` 0.138, `scheme_agreement` 0.104, **`max_score` 0.011**. The
feature signal 1 thresholds is the least informative of the five.

It also sharpens why signals 2 and 3 matter: both read the passage *text*, which
is the only place the missing information can be. Every retrieval-side feature
is a summary of score geometry, and the score geometry of a question about a
present scheme looks the same whether or not the specific fact is there.

## 13. Grounding metrics

**H3 claims RAG reduces unsupported answers. That requires a measurement of "unsupported", not an assumption.**

| Metric | Definition |
|---|---|
| **Citation Support Rate** | Of answered questions, the fraction whose answer is supported by the cited passage. Supported = ROUGE-L precision of answer against passage above a threshold, **or** NLI entailment probability above a threshold. Both variants reported |
| **Unsupported Answer Rate** | `1 − CSR`. For the closed-book arm, computed against the gold passage, since arm A cites nothing |
| **Abstention Rate** | Fraction of all questions answered with the refusal string, per arm and per query type |
| **Over-abstention** | Answerable questions incorrectly refused — the cost side of a conservative threshold |

The ROUGE-L variant is cheap and language-agnostic but rewards copying; the NLI variant is semantic but has its own error rate. Reporting both, and reporting where they disagree, is more honest than picking the one that gives the better number.

---

## 14. Evaluation harness

**Report objects with named metric methods; formatters that return `list[str]`.** The console path and the `--report <path>` path call the same formatter, so what is committed to `evals/report-*.txt` is byte-identical to what was printed.

```python
@dataclass
class RetrievalReport:
    outcomes: list[Outcome]
    def recall_at(self, k: int, *, slice: str | None = None) -> float: ...
    def precision_at(self, k: int, *, slice: str | None = None) -> float: ...
    def hit_rate_at(self, k: int, *, slice: str | None = None) -> float: ...
    def mrr(self, *, slice: str | None = None) -> float: ...
    def ndcg_at(self, k: int, *, slice: str | None = None) -> float: ...
    def seconds_per_query(self) -> float: ...

def format_retrieval(r: RetrievalReport) -> list[str]: ...
def format_misses(r: RetrievalReport, limit: int) -> list[str]: ...
```

`slice` takes a language-type or scheme key, which is how every table gets its per-language breakdown without a second code path.

### 14.1 Question-answering metrics

Exact Match and token-level F1 both depend on a normalizer, and **a normalizer written for English is wrong for Hindi**. `evaluation/qa.py` implements a script-aware one: casefold and strip articles for English; strip nothing article-like for Hindi (it has none) but normalize the common postposition spacing (`के लिए` / `केलिए`); unify digit systems and currency surface forms on both sides; strip punctuation including danda. The normalizer has its own tests, because a quiet bug in it shifts every EM and F1 number in the paper.

Semantic similarity uses LaBSE cosine between predicted and gold answers — chosen because it is already in the model registry and because it is cross-lingual, so a correct Hindi answer to an English gold answer still scores well.

### 14.2 Statistics

Every headline comparison carries a **bootstrap 95% confidence interval** (1000 resamples over queries, seeded) and, between two systems, a **paired bootstrap or approximate randomization test**. With 280 test items, differences under roughly 3–4 points are unlikely to be distinguishable from noise, and a project that reports "dense beats BM25 by 2.1 points" without that context is overstating its evidence. The report format prints the interval next to every headline number.

---

## 15. Testing strategy

Flat `tests/`, one file per source module, sentence-length test names, no network access in any test.

The tests that matter most, because they guard silent-corruption failures rather than crashes:

| Test | Guards |
|---|---|
| `test_devanagari_validator_rejects_reordered_matras` | §3.2 — the highest-risk failure |
| `test_normalizer_is_idempotent_and_matches_on_both_sides` | §3.3 — query/passage normalization divergence |
| `test_passage_ids_are_stable_across_resegmentation` | §4 — invalidated caches and gold references |
| `test_embedding_cache_is_rejected_when_passage_hash_differs` | §7.3 — misaligned embeddings |
| `test_langid_classifies_the_brief_example_as_code_mixed` | §10.1 — the brief's own example is a test case |
| `test_qa_normalizer_treats_devanagari_and_ascii_digits_as_equal` | §14.1 — every EM/F1 number |
| `test_answerable_response_always_carries_a_citation` | PRD §9.1 — the output contract |
| `test_unanswerable_response_uses_the_exact_required_string` | PRD FR-11 |

Fixtures build a tiny synthetic bilingual corpus (roughly 20 passages) so that retrieval and evaluation are testable end to end in under a second without loading a model.

---

## 16. Configuration

`pydantic-settings` `BaseSettings` with `env_file=".env"`, plus an explicit `load_dotenv(find_dotenv(usecwd=True), override=False)` at module import — because third-party SDKs and `llama-cpp-python` read `os.environ` directly and never see the settings object.

Derived paths are `@property`; the accessor is `@lru_cache def get_settings()`.

Configuration groups (names only; `.env.example` is committed, `.env` is not):

```
INDICRAG_DATA_DIR            where corpus, embeddings and caches live
Encoders                     ENCODER_PRIMARY, ENCODER_LIST, ENCODER_BATCH_SIZE
Retrieval                    RETRIEVAL_TOP_K, HYBRID_ALPHA, HYBRID_METHOD, RERANK_ENABLED
Generation                   LLM_BACKEND, LLM_GGUF_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_SEED
Answerability                ANSWERABILITY_TAU, ANSWERABILITY_MODE, NLI_MODEL
Optional API                 LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
Caches                       HF_HOME, TRANSFORMERS_CACHE  (set by scripts/dev-env, not here)
```

### 16.1 Disk redirection

**With 31.6 GB free on the system drive, model caches cannot live in the default `~/.cache/huggingface`.** `scripts/dev-env.ps1` and `scripts/dev-env.sh` set `HF_HOME`, `UV_CACHE_DIR` and `INDICRAG_DATA_DIR` to a chosen data directory before anything downloads. This is the first thing run in a fresh environment, and P0 of the plan treats it as a gate, not a nicety — a half-downloaded 2 GB model on a full drive is a bad afternoon.

The CPU-only torch wheel is installed explicitly (`--index-url https://download.pytorch.org/whl/cpu`, ~200 MB) rather than the default, which pulls the ~2.5 GB CUDA build that this machine cannot use.

---

## 17. Storage layout and caching

```
data/                                    gitignored in full
├── corpus_manifest.jsonl                provenance (committed — small, and it is the citation record)
├── raw/<doc_id>.{pdf,html}              source snapshots
├── text/<doc_id>.txt                    extracted, pre-normalization
├── passages.jsonl                       the segmented corpus + passages.sha256
├── emb/<model-slug>.npy                 float32 (n, dim), L2-normalized
├── emb/<model-slug>.meta.json           model revision, passage hash, settings
├── lex/{tfidf,bm25}.pkl                 fitted lexical indices
└── gen/<model>/<arm>.jsonl              cached generations, keyed by prompt hash
```

The invariant: **nothing expensive is computed twice, and every cache carries the hash of its input.** A cache whose input hash does not match is rejected and rebuilt rather than used. `data/` is gitignored except `corpus_manifest.jsonl`, which is committed because it is what lets someone else reconstruct the corpus.

---

## 18. CLI surface

One Typer app, nested sub-apps, `no_args_is_help=True`.

**As built, 2026-09-21.** This section previously described a surface that
was partly aspirational; four of the commands below did not exist, including
`index build`, so the embeddings every retrieval number depends on could only be
regenerated by an ad-hoc script. That is an NFR-6 failure, and the list is now
kept against the code.

```
indicrag corpus fetch                      download sources, write manifest
indicrag corpus validate                   Devanagari integrity checks
indicrag corpus segment [--version 1]      passages.jsonl; v1 is the frozen corpus, v2 the corrected one
indicrag corpus audit [--report <path>]    passage-level invariants (§4 metadata, token bounds)
indicrag corpus repair-spans [--apply]     re-derive char_span/text_raw by alignment; never moves IDs
indicrag corpus stats                      counts by scheme, language, length histogram

indicrag index lexical                     fit TF-IDF and BM25
indicrag index build --encoder <name>      encode + cache; --all for every registry entry
                                           [--pooling mean|cls] [--batch-size 16]

indicrag ask "<query>" [--method hybrid] [--k 5] [--gguf <path>]
                                           the demo path; prints the §9 object.
                                           Extractive without --gguf, generated with it.

indicrag eval retrieval     [--report <p>] [--gold <p>] [--alpha 0.4]
                                           [--sweep/--no-sweep]   alpha sweep, H2
indicrag eval probes        [--out <p>] [--per-shape 60]   rebuild the probe set
indicrag eval qa            [--report <p>] [--arms A,B,C,D] [--k 5] [--sample N]
                                           [--gguf <path>] [--no-model] [--nli]
indicrag eval answerability [--report <p>] [--method hybrid] [--dev-fraction 0.3]
                                           [--gguf <p>]   adds signals 2 and 3
                                           [--sample N] [--enrich]
indicrag eval errors        [--report <p>] [--out <p>] [--limit 20]
indicrag eval all           [--report evals/]     regenerates every table

indicrag dataset generate   [--gguf <p>] [--merge-into <p>]   candidate bootstrapping
indicrag dataset merge                             fold candidates into the gold set
indicrag dataset scaffold-unanswerable             the 80 unanswerable items
indicrag dataset verify [--assist <p>]             annotation review loop, resumable; shows pre-review notes
indicrag dataset review [--notes <p> --reviewed-by <who>] [--out <p>] [--report <p>]
                                                   rule checks + reading-pass notes; advice, never verifies
indicrag dataset second-pass [--draw|--compare]    blind 15% sample, Cohen's kappa
indicrag dataset split      [--seed 20260922]      stratified dev/test, verified only
indicrag dataset stats                             matrix coverage vs PRD §6.2
indicrag dataset reanchor --out <p> [--to-version 2] [--report <p>]
                                                   map gold citations v1 -> v2; proposes, never overwrites
```

`--report <path>` is available on every `eval` subcommand; a formatter returning
`list[str]` means the file and the console are the same bytes. `--no-model` is
specific to `eval qa`, which is the only evaluation that needs a generator.

There is no `corpus extract`: after the source change in PRD §5.4 the corpus
arrives as text through the MediaWiki API, so extraction and OCR fall away and
`corpus validate` carries the integrity checks that command was going to run.
`ask --explain` is not built; the response object already carries the citation
and snippet that flag would have printed.

`indicrag dataset stats` exists for a specific reason: PRD §6.2 fixes a language-pair matrix, and annotation drifts from it. The command prints actual counts against target counts so the drift is visible while it is still cheap to correct.

---

## 19. Performance budget

**Measured on the development machine, 2026-09-20; generation figures added 2026-09-21.** Encoding and generation figures are real. The final table is the original estimate, retained so the misses are visible.

| Encoder | Params | 694 passages | Rate |
|---|---|---|---|
| MiniLM-L12 | 118M | 149.6 s | 4.6/s |
| multilingual-e5-base | 278M | ~380 s | ~1.8/s |
| LaBSE | 471M | ~600 s | ~1.2/s |
| MuRIL (mean) | 236M | 632.3 s | 1.1/s |
| MuRIL (CLS) | 236M | 622.2 s | 1.1/s |
| **All five** | | **~40 min** | one-time, cached |

The estimate in the original version of this table was 35–50 minutes for the full
registry, and the measured total landed inside it. What the estimate got wrong is
the *shape*: throughput does not scale with parameter count as assumed. MuRIL at
236M is slower than LaBSE at 471M, because the MLM path runs through a
hand-rolled `transformers` loop while the sentence models use SentenceTransformer's
optimised batching. Anything added to the MLM arm should be budgeted at MuRIL's
rate, not interpolated from its size.

### 19.1 Generation, measured

**Measured 2026-09-21** on Qwen2.5-3B-Instruct-Q4_K_M, 4 threads, over a
stratified 72-item Module 4 run.

| Arm | Prompt | Measured | n |
|---|---|---|---|
| A closed-book | no passages, ~60 tokens | **8.9 s** | 72 |
| B RAG-dense | k=5 passages, ~1500 tokens | **57.7 s** | 11 |

The original estimate of 8–12 s per generation was right for the closed-book arm
and wrong by roughly 5× for the RAG arms, because it budgeted output tokens and
ignored prefill. At k=5 the prompt is ~1500 tokens against ~80 generated, so
**prefill dominates and cost scales with the context, not the answer**. The
practical lever is therefore k and the per-passage excerpt length, not
`max_tokens` — halving the output budget would save almost nothing.

Revised full-sweep figure: 320 answerable items × 4 arms ≈ **16 h**, against the
4–5 h in the estimate below. This is why Module 4 runs on a stratified sample
first (`eval qa --sample N`) and why generations are cached per item rather than
per run.

One related measurement, since it looks like a contradiction: removing
schema-constrained decoding sped the closed-book arm up 2.7× (24 s → 8.9 s) but
the RAG arms only ~1.2× (71 s → 57.7 s). Grammar checking is a per-*generated*-
token cost, so it is diluted exactly where prefill dominates.

The table below is the original per-stage budget, kept for comparison. Its
generation rows are the estimates corrected above.

| Stage | Scale | Estimate | Notes |
|---|---|---|---|
| PDF extraction | 40 docs | 2–5 min | One-time |
| Segmentation | 40 docs → ~1200 passages | < 1 min | One-time |
| Dense encoding, MiniLM (118M) | 1200 passages | ~2–4 min | One-time per model |
| Dense encoding, E5-base (278M) | 1200 passages | ~6–10 min | One-time per model |
| Dense encoding, LaBSE (471M) | 1200 passages | ~12–18 min | One-time per model |
| All five encoders | 1200 passages | **~35–50 min total** | One-time, cached |
| Query encoding | 1 query | < 100 ms | |
| Exact dense search | 1 query over 1200 | < 1 ms | Matrix-vector product |
| BM25 search | 1 query | < 5 ms | |
| Generation, 1.5B Q4 | ~80 tokens | ~4–6 s | Interactive path |
| Generation, 3B Q4 | ~80 tokens | ~8–12 s | Batch path |
| Full QA sweep, 400 q × 4 arms, 3B | 1600 generations | **~4–5 h** | Run once, cached |
| Full QA sweep, 400 q × 4 arms, 1.5B | 1600 generations | **~2–3 h** | Run once, cached |
| NLI scoring | 400 pairs | ~2–3 min | |
| Reranking, if enabled | 400 q × 20 pairs | ~20–25 min | |
| Evaluation from cache | all tables | < 30 s | No model loaded |

The two multi-hour entries are why §11.6 caching exists and why they are scheduled as overnight runs in the plan rather than as interactive work.

---

## 20. Extension points

Listed because the brief's §9 bonus items map onto them, and because each is a clean seam rather than a rewrite:

| Extension | Seam |
|---|---|
| A larger instruction-tuned model | `OpenAICompatProvider` already exists; add the key, add the arm |
| Query translation instead of direct multilingual embedding | A query-side transform, sits beside `translit.py` |
| Additional Indic languages | Add documents; the encoders and pipeline are language-agnostic. The langid lexicon and the QA normalizer need per-language additions |
| Multi-document question answering | Retrieval already returns k passages from multiple documents; the prompt and the gold schema need multi-passage answers |
| Approximate search at scale | Swap the matrix in `dense.py` for FAISS behind the same interface |
| A different domain | Only `corpus/` and the scheme alias table are domain-specific |

---

## 21. Known gaps

**Nothing in this document is implemented yet.** It is a design, written before the code, and it will be wrong in places that only implementation will reveal. Specific items already known to be uncertain:

- **Throughput numbers in §19 are estimates.** They come from the hardware profile, not from measurement. P0 replaces them, and the schedule in `docs/PLAN.md` depends on the result.
- **Reranking (§9) is optional** and is the first thing cut if the schedule slips.
- **The light Hindi stemmer (§6.1)** handles common case-marker suffixes only. A proper Hindi morphological analyser is out of scope, and its absence will cost some lexical recall on Hindi queries — which should be noted when reading the BM25-on-Hindi number.
- **OCR quality is unmeasured.** If a meaningful share of Hindi documents require OCR, extraction quality becomes a confound in the cross-lingual comparison, and the paper must report what share of Hindi passages came from OCR.
- **The scheme alias table (§10.2) is hand-built** and will be incomplete. Missing aliases show up as named-entity failures in Module 6, which is at least a visible failure mode.
- **No fine-tuning of any component.** PRD §4 explains the reasoning.
- **The web demo (§2, `demo/app.py`) is optional.** The CLI `ask` command is the deliverable demo; the web interface is a convenience if time allows.
- **Single-passage answers only.** The generator receives k passages but the gold set contains no questions requiring composition across them, so nothing here measures multi-hop behaviour.
