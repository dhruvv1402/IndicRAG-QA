# PRD — IndicRAG-QA

*Evidence-grounded cross-lingual question answering for Indic and code-mixed queries over Indian government scholarship schemes.*

This document states **what** is being built and **why**, and fixes the numbers that the architecture and the experiment plan depend on. `docs/ARCHITECTURE.md` states how it is built; `docs/PLAN.md` states in what order. Where this document and the course brief (`PJ_09.pdf`) disagree, the brief wins and this document is wrong and must be corrected.

Course: CSET 346 — Natural Language Processing. Instructor: Dr. Ankur Kumar Singhal, Bennett University.

---

## 1. Motivation and problem statement

**Public-interest information in India is published in one language and asked about in another.** A student eligible for a post-matric scholarship reads a circular issued in English, but asks a question in Hindi, or — far more commonly in practice — in Romanized Hindi-English code-mix: *"Scholarship ke liye minimum eligibility kya hai?"*. Keyword search fails on all three mismatches at once: different language, different script, and different surface words for the same concept.

Retrieval-augmented generation can bridge this, but introduces its own failure: a fluent language model will answer a question it has no evidence for. For a document set that governs who receives public money, a confident wrong answer is worse than no answer. The system must therefore be able to say *"Insufficient information available in the provided documents."* and be measured on how often it says so correctly.

Formally, given a document collection `D = {d₁, d₂, …, dₙ}` segmented into passages, a retriever `R` returns the top-k passages for a query, and a generator `f` produces an answer conditioned only on those passages:

```
D = {d₁, d₂, …, dₙ}      R(q) = {p₁, p₂, …, p_k}      A = f(q, R(q))
```

The constraint that matters is the *only* in "conditioned only on": `A` must be derivable from `R(q)`, and when it is not, the system must abstain rather than fall back on parametric knowledge.

### 1.1 Central research question

> How effectively can multilingual semantic retrieval and RAG improve question answering for Indic and code-mixed queries compared with lexical retrieval, while keeping generated answers grounded in evidence?

### 1.2 What makes this a research project rather than an application

The deliverable is not a working chatbot. It is a set of controlled comparisons — lexical vs dense vs hybrid retrieval, closed-book vs retrieved vs oracle-context generation, four answerability signals — each reported with per-language breakdowns, confidence intervals and an error analysis that attributes failures to a specific component. A demo that answers questions but cannot say *which* component fails on Hinglish queries does not satisfy the brief.

---

## 2. Research questions and hypotheses

**Each hypothesis is falsifiable by a specific experiment and maps to a required module.** Stating them before running anything is what prevents the results section from becoming a post-hoc narrative.

| # | Hypothesis | Falsified if | Module | Experiment |
|---|---|---|---|---|
| **H1** | Multilingual dense retrieval outperforms lexical retrieval, and the margin is widest on cross-lingual and code-mixed queries | Dense Recall@5 ≤ BM25 Recall@5 on the cross-lingual slice | M1, M2 | Retrieval sweep, per language-pair |
| **H2** | Hybrid retrieval beats either component alone, mainly by recovering exact-match named entities and numbers that dense retrieval blurs | No α ∈ (0,1) beats both endpoints on dev | M3 | α-sweep and RRF, tuned on dev |
| **H3** | RAG reduces unsupported answers relative to direct closed-book LLM question answering | Citation Support Rate for RAG ≤ that of the closed-book arm | M4, M5 | Arms A/B/C/D plus grounding metrics |
| **H4** | Code-mixed queries are harder than monolingual queries for every retrieval method, but the *relative* penalty is smaller for dense than for lexical retrieval | Hinglish Recall@5 ≥ monolingual Recall@5 | M2, M6 | Per-language-type breakdown |

Two secondary questions, carried as bonus work rather than core claims:

- **H5** — Romanized-to-Devanagari transliteration of the query recovers a meaningful share of the lexical retriever's loss on code-mixed input. This isolates *script* mismatch from *semantic* mismatch, which the headline dense-vs-lexical comparison conflates.
- **H6** — Indic-pretrained masked-language-model checkpoints (MuRIL, IndicBERT), used off-the-shelf with mean pooling, underperform smaller models trained explicitly for sentence retrieval (E5, LaBSE, MiniLM). If true this is a useful negative result for anyone reaching for "the Indic model" by name.

---

## 3. Users and usage scenarios

**Two audiences, and the second one is the reason the output contract is shaped the way it is.**

| User | Scenario | What they need from the output |
|---|---|---|
| **Applicant** — a student or parent checking eligibility | Asks *"NMMS ke liye income limit kitni hai?"* on a phone, in Romanized Hindi | A short answer in the language they asked in, and a visible source so they can verify before acting |
| **Operator / evaluator** — the project author, the instructor, or an administrator auditing the system | Needs to know whether a given answer came from the corpus or from the model's memory | Passage IDs, document IDs, retrieval scores, and an abstention when evidence is thin |

The applicant is why answers must be in the query language. The operator is why every answer carries citations and a confidence score, and why the evaluation splits retrieval from generation.

Explicitly **not** a user: anyone treating the output as authoritative advice. The system reports what the documents say; the documents themselves are the authority. This caveat belongs in the README and the demo interface, not only here.

---

## 4. Scope

| In scope | Out of scope |
|---|---|
| English, Hindi (Devanagari), Hindi-English code-mix (Romanized) | Any other Indian language; Devanagari-script English |
| Single-passage extractive and abstractive answers | Multi-hop reasoning across passages requiring composition |
| A static, pre-fetched document snapshot with recorded retrieval dates | Live scraping of government portals at query time |
| Text queries typed by a user | Speech input, OCR of user-supplied images |
| Retrieval, generation, answerability, and their evaluation | Fine-tuning any encoder or generator |
| A CLI demo, optionally a local web interface | Deployment, authentication, multi-user serving, persistence of user queries |

**On fine-tuning.** The brief permits it but does not require it. On CPU-only hardware, fine-tuning a 278M-parameter encoder would take longer than the rest of the project combined and would risk overfitting a 400-item dataset. The project instead compares five off-the-shelf encoders, which is the more useful result for a practitioner choosing one. This is a deliberate trade and is stated in the paper's limitations section.

---

## 5. Corpus specification

**The corpus is bilingual by construction, which is what makes cross-lingual retrieval a real measurement rather than a simulation.** Indian government scholarship schemes publish the same guidelines in English and Hindi. Because both versions are in the index, a Hindi query can legitimately retrieve an English passage and vice versa, and we can measure whether the retriever crosses the language boundary when the better evidence is on the other side.

### 5.1 Target size

| Property | Target | Rationale |
|---|---|---|
| Schemes | 12–20 | Enough near-duplicate structure (every scheme has an eligibility clause) to make retrieval non-trivial |
| Documents | 25–40 | Roughly each scheme in both English and Hindi |
| Passages | 800–1500 | Large enough that top-k retrieval is a real choice; small enough that exact dense search needs no approximate index |
| Passage length | 120–220 tokens | Long enough to contain a complete clause; short enough that a 3B model's context holds five of them |

### 5.2 Candidate schemes

National Scholarship Portal (NSP) master guidelines · PM-YASASVI · National Means-cum-Merit Scholarship (NMMS) · Pre-Matric and Post-Matric Scholarships for SC / ST / OBC / Minority students · Central Sector Scheme of Scholarships (CSSS) · AICTE Pragati (girl students) · AICTE Saksham (differently-abled students) · INSPIRE Scholarship (DST) · PM Vidyalaxmi and education-loan interest subsidy · National Scholarship for Persons with Disabilities.

These are chosen for a specific property: they **overlap heavily**. Several schemes have similar income ceilings, similar documentation lists and similar renewal rules but differ in the details. That near-duplication is precisely what stresses a retriever, and it is where the hardest Module 6 error cases will come from.

### 5.3 Provenance record

Every source document is recorded in `data/corpus_manifest.jsonl` before extraction:

```
{ "doc_id", "scheme", "ministry", "lang", "title", "source_url",
  "retrieved_at", "sha256", "licence", "format", "pages", "notes" }
```

`retrieved_at` and `sha256` exist so the paper can state exactly which snapshot produced the numbers — government circulars are revised without changing their URL.

### 5.4 What was actually built, and why it differs

**The corpus is built from parallel English/Hindi Wikipedia articles on Indian
government schemes, not from ministry guideline PDFs.** This is a deviation from
§5.2 above, forced by two findings on first contact with the official sources —
both of them risks R1 and R2 in `docs/PLAN.md` §5, which duly materialised:

1. The Hindi paths on `socialjustice.gov.in` return HTTP 200 with a *"Page not
   found"* body — a soft 404. The bilingual premise does not survive there.
2. `PMS_for_SCs_Scheme_Guidelines.pdf` (18 pages, 2.8 MB) yields **17 characters**
   of extractable text. It is a scan, as most Hindi circulars of this kind are,
   and this machine has no Tesseract, so OCR was not available.

An English-only text-layer corpus would have defeated the cross-lingual research
question entirely, which is the thing the project exists to measure. Wikipedia
pairs are genuinely parallel (resolved through the MediaWiki `langlinks` API
rather than matched by hand), text-layer by construction, section-structured, and
CC BY-SA licensed.

**What it costs, stated here so it reaches the paper's limitations rather than a
reader's own inference:** the register is *encyclopedic rather than regulatory*.
Wikipedia describes schemes; it does not legislate them. Eligibility criteria and
amounts appear, but as secondary prose rather than normative clauses, and they
may lag the current circular. Every hypothesis in §2 remains testable — the
corpus is bilingual, dense with near-duplicate scheme descriptions, and rich in
amounts, dates and named entities — but **no claim about retrieval over statutory
text is supported by these results.**

As built: **40 schemes × 2 languages = 80 documents → 694 passages** (491 English,
203 Hindi). The Hindi/English passage imbalance is inherent: Hindi Wikipedia
articles run roughly a third the length of their English counterparts. 694 falls
below the 800 floor in §5.1; the shortfall is recorded rather than papered over,
and it slightly reduces the pool from which Hindi-evidence questions can be drawn.

### 5.5 Licensing and redistribution

Indian government documents of this kind are generally published under the Government Open Data Licence – India (GODL-India) or equivalent permissive terms, which allow reuse with attribution. The repository therefore commits **derived passages and the manifest with source URLs**, not the original PDFs, and attributes each scheme to its issuing ministry. Any document whose licence cannot be established is excluded rather than assumed permissive, and the exclusion is recorded in the manifest's `notes` field.

---

## 6. Dataset specification

**This is the section the rest of the project is bolted to.** The language-pair matrix is what makes Module 2 measurable, and the unanswerable taxonomy is what makes Module 5 more than a formality.

### 6.1 Size and composition

**400 instances = 320 ANSWERABLE + 80 UNANSWERABLE (20%).**

This sits inside the brief's recommended 300–500 range. The 20% unanswerable share is a design choice, not a natural rate: it is high enough that the answerability classifier has enough examples of the positive class to produce a meaningful F1, and it is reported as a design parameter so that nobody reads the resulting accuracy figure as a real-world abstention rate.

### 6.2 Answerable language-pair matrix (320 items)

Query language is what the user typed; passage language is the language of the gold evidence passage.

| Query ↓ / Gold passage → | English | Hindi | Row total |
|---|---|---|---|
| **English** | 70 *(monolingual)* | 45 *(cross-lingual)* | 115 |
| **Hindi (Devanagari)** | 45 *(cross-lingual)* | 60 *(monolingual)* | 105 |
| **Hinglish (code-mixed)** | 55 *(code-mixed, cross-script)* | 45 *(code-mixed, cross-script)* | 100 |
| **Column total** | 170 | 150 | **320** |

Three groupings fall out of this matrix directly, and they are the three Module 2 comparisons:

- **Monolingual** (130) — query and evidence in the same language and script.
- **Cross-lingual** (90) — query and evidence in different languages, each in its native script.
- **Code-mixed** (100) — Romanized Hindi-English query; evidence in either language, never in the query's script.

The Hinglish row is deliberately never "monolingual", because Romanized Hindi is not the script of any corpus document. Code-mixed queries therefore always cross a script boundary, which is exactly why they are expected to be hardest.

### 6.3 Unanswerable taxonomy (80 items)

**A question about a topic the corpus has never heard of is trivially rejected and proves nothing.** The value is in the near-misses. The 80 items are stratified so that answerability results can be read per class:

| Class | n | Definition | Example |
|---|---|---|---|
| **Out-of-scope** | 25 | Topic absent from the corpus entirely | *"What is the current home loan interest rate?"* |
| **Near-miss distractor** | 30 | Topic and scheme are in the corpus; the specific fact is not | *"What is the exact last date for NMMS renewal in 2025?"* when the document says only "as notified annually" |
| **False premise** | 15 | Presupposes a benefit or clause that does not exist | *"Pragati scholarship ke tahat laptop allowance kitna hai?"* when no laptop allowance exists |
| **Under-specified** | 10 | Cannot be answered without disambiguation | *"What is the income limit?"* with no scheme named, where schemes differ |

The near-miss class is the real hallucination test: retrieval will confidently return a highly similar passage, the generator will have plausible-looking context, and only a correctly calibrated answerability check will abstain. Reporting a single aggregate answerability F1 without this breakdown would hide that.

The 80 unanswerable items are distributed across query types in roughly the same proportion as the answerable set (≈36% English, ≈33% Hindi, ≈31% Hinglish) so that per-language abstention rates are comparable.

### 6.4 Record schema

One JSON object per line in `evals/gold.jsonl`, written with `ensure_ascii=False`:

```json
{
  "id": "nmms-0042",
  "question": "NMMS ke liye family income limit kitni hai?",
  "query_lang": "hinglish",
  "script": "latin",
  "passage_lang": "en",
  "answerable": true,
  "unanswerable_class": null,
  "answer_gold": "Rs. 3,50,000 per annum",
  "answer_gold_hi": "3,50,000 रुपये प्रति वर्ष",
  "gold_passage_ids": ["nmms-en#p017", "nmms-hi#p019"],
  "scheme": "NMMS",
  "difficulty_tags": ["number", "code-mix", "named-entity"],
  "split": "test",
  "annotator": "a1",
  "second_pass": false,
  "verified": true,
  "notes": ""
}
```

`gold_passage_ids` is a list, not a scalar. When the same fact appears in both the English and the Hindi version of a document, **both count as correct evidence**, and scoring recall against only one of them would penalise a retriever for choosing the other. This single decision materially changes cross-lingual Recall@k and must be settled before any numbers are produced, not after.

### 6.5 Annotation protocol

1. **Candidate generation.** For each passage, prompt the local model for two or three candidate questions plus the answer span, in the passage's own language.
2. **Translation and code-mixing.** Produce the Hindi and Hinglish variants for the designated share of items. Hinglish variants are hand-written or hand-corrected — machine-generated code-mix is not representative of how people actually type.
3. **Human verification of every item.** The annotator confirms the question is answerable from the cited passage, corrects the gold answer to the document's exact wording wherever the answer is a number, date or name, and records `gold_passage_ids`. No item enters the set unverified; items with `verified: false` are excluded from every reported metric.
4. **Unanswerable authoring.** The 80 unanswerable items are written by hand against the taxonomy in §6.3, with the near-miss items authored while looking at the passage they are designed to nearly match.
5. **Second pass.** A 15% random sample (60 items) is independently re-labelled for the `answerable` field and **Cohen's κ is reported**. A κ below 0.70 means the answerability boundary is not well defined and the guidelines need tightening before the main results can be trusted.
6. **Splitting.** Stratified by (query language × answerability × scheme), seeded: **dev = 120, test = 280**. Every threshold, α, k and prompt is tuned on dev. The test split is scored once, at the end.

### 6.6 Leakage control

**LLM-generated questions copy the wording of the passage they were generated from, which silently inflates lexical retrieval.** If BM25 scores well only because the question reuses the passage's rare terms, the dense-vs-lexical comparison is measuring an annotation artefact rather than retrieval quality.

Mitigations, all reported rather than merely applied:

- The generation prompt instructs paraphrase, not extraction, for the question stem.
- After annotation, compute the token overlap (Jaccard over content words) between each question and its gold passage, and **publish the distribution**.
- Define a **low-overlap subset** (bottom tertile) and report the headline retrieval comparison on it as well as on the full set. If the dense-vs-lexical gap widens on the low-overlap subset, the main result is conservative; if it collapses, the main result was an artefact and the paper must say so.

---

## 7. Functional requirements

**Every required module in brief §5 maps to at least one requirement here; a module with no requirement is a gap.**

| ID | Requirement | Module |
|---|---|---|
| FR-1 | Ingest source documents, record provenance, and extract text with a validated Devanagari integrity check | — |
| FR-2 | Segment documents into passages with stable `doc_id` / `passage_id` and carried metadata (scheme, language, section path, page) | — |
| FR-3 | Provide a TF-IDF cosine retriever and a BM25 retriever, both with script-aware tokenization | M1, M3 |
| FR-4 | Provide multilingual dense retrieval over at least four encoders, including one Indic-pretrained checkpoint named in the brief | M1 |
| FR-5 | Provide hybrid retrieval by weighted score fusion `α·lex + (1−α)·dense` **and** by reciprocal rank fusion, with α and k tuned on dev only | M3 |
| FR-6 | Detect and report query type as one of English / Indic / Code-Mixed | M2 |
| FR-7 | Normalize queries, including Roman→Devanagari transliteration of code-mixed input as a switchable ablation | M2, bonus |
| FR-8 | Generate an answer from retrieved passages only, in the query's language, emitting structured output with citations | M4 |
| FR-9 | Support a closed-book (no-retrieval) generation arm and an oracle-context arm for error attribution | M4 |
| FR-10 | Classify each query ANSWERABLE / UNANSWERABLE via at least three independent signals plus a calibrated combination | M5 |
| FR-11 | Return the literal string `Insufficient information available in the provided documents.` when UNANSWERABLE | M5 |
| FR-12 | Score retrieval, question answering and answerability **separately**, with per-language-type breakdowns | M1–M5 |
| FR-13 | Report bootstrap confidence intervals and paired significance tests between compared systems | M1–M5 |
| FR-14 | Record and render at least 15 difficult cases against a fixed error taxonomy | M6 |
| FR-15 | Expose the whole pipeline through a CLI, including one command that reproduces every reported table | — |

### 7.1 Requirement-to-module coverage

M1 → FR-3, FR-4 · M2 → FR-6, FR-7, FR-12 · M3 → FR-3, FR-5 · M4 → FR-8, FR-9 · M5 → FR-10, FR-11 · M6 → FR-14. No module is unmapped.

---

## 8. Non-functional requirements

**The hardware is the binding constraint, and every choice in the architecture answers to it.** The development machine is an Intel i5-11320H (4 cores / 8 threads), 15.7 GB RAM, **no CUDA GPU** (Intel Iris Xe integrated graphics only), with **31.6 GB free on the system drive** and Python 3.10.

| ID | Requirement | Consequence |
|---|---|---|
| NFR-1 | Every model must run on CPU | No encoder above roughly 600M parameters; generation via 4-bit quantized GGUF |
| NFR-2 | The full pipeline must run offline after a one-time model download | No API dependency in the default path; network is needed only for corpus fetch and model pull |
| NFR-3 | Total disk footprint (models, caches, data) under 12 GB | `HF_HOME` redirected off the system drive; CPU-only torch wheel (~200 MB, against ~2.5 GB for the CUDA build) |
| NFR-4 | Every expensive artefact cached and keyed by content hash | Passage embeddings and model generations are computed once; re-running the analysis never re-runs a model |
| NFR-5 | Runs are deterministic given a seed | All sampling, splitting and bootstrapping is seeded, and the seed is recorded in each report header |
| NFR-6 | Every reported table reproducible by one command | `indicrag eval all --report evals/` regenerates everything from cached artefacts |
| NFR-7 | Interactive query latency under roughly 10 s with the 1.5B model | Acceptable for a live demo; the 3B model is a batch-mode arm |
| NFR-8 | A model-free fast path exists for tests and CI | `--no-model` runs retrieval and evaluation without loading any generator |

**On latency.** The 10 s target covers embedding one query, exact search over ~1500 passages, and generating roughly 80 tokens at 15–25 tokens/second. It excludes model load time, which is amortised by keeping the process warm in the demo.

---

## 9. Output contract

**Every query returns the same object, whether answerable or not.** This is the structure required by brief §7.

```json
{
  "query": "Scholarship ke liye minimum eligibility kya hai?",
  "query_type": "Code-Mixed",
  "query_lang": "hi-en",
  "answerability": "ANSWERABLE",
  "answer": "Class 9 me admission liya ho aur family income Rs. 3,50,000 se kam ho.",
  "answer_lang": "hi-en",
  "confidence": 0.81,
  "citations": [
    {
      "passage_id": "nmms-en#p017",
      "doc_id": "nmms-en",
      "scheme": "NMMS",
      "lang": "en",
      "score": 0.79,
      "snippet": "…students whose parental income from all sources does not exceed Rs. 3,50,000 per annum…"
    }
  ],
  "explanation": "The cited passage states the parental income ceiling and the class-of-study condition; both parts of the answer come from it.",
  "retrieval": { "method": "hybrid-rrf", "k": 5, "top_score": 0.79, "margin": 0.11 }
}
```

### 9.1 Field rules

- `query_type` ∈ {`English`, `Indic`, `Code-Mixed`} — exactly the three classes the brief names.
- `answerability` ∈ {`ANSWERABLE`, `UNANSWERABLE`}.
- When `UNANSWERABLE`, `answer` is the literal string **`Insufficient information available in the provided documents.`**; `citations` may still be populated, showing what was retrieved and then rejected, which is useful to the operator; and `confidence` reports confidence in the abstention.
- `answer` is written in the language the query was asked in. For code-mixed queries the answer may be code-mixed; the evaluation normalizer handles both scripts.
- `citations` is never empty for an `ANSWERABLE` response. An answer with no citation is a bug, and there is a test asserting it.
- `explanation` is best-effort and may be omitted by the smaller model; it is not scored as a primary metric.

---

## 10. Evaluation requirements and success criteria

**Retrieval and generation are scored separately so that a bad answer can be attributed to the component that caused it.** This is a requirement of the brief, not a nicety: the oracle-context arm exists specifically so that "the retriever missed" and "the generator ignored good evidence" are distinguishable.

### 10.1 Metrics

| Component | Metrics | Breakdown |
|---|---|---|
| **Retrieval** | Recall@k, Precision@k, Hit Rate@k for k ∈ {1, 3, 5, 10}; MRR; nDCG@10 | Per language pair; per scheme; full set and low-overlap subset |
| **Question answering** | Exact Match, token-level F1 (script-aware normalization), semantic similarity (LaBSE cosine to gold), LLM-judged answer correctness on a sampled subset | Per query type; per arm (A/B/C/D) |
| **Answerability** | Accuracy, Precision, Recall, F1 on the UNANSWERABLE class; confusion matrix; precision-recall curve over τ | Per unanswerable class; per query type |
| **Grounding** | Citation Support Rate; Unsupported Answer Rate | Per arm — this is what tests H3 |
| **Language analysis** | Every table above, broken out for English / Hindi / cross-lingual / code-mixed | Required by brief §6 |

`MRR = (1/N) Σᵢ 1/rankᵢ`, with reciprocal rank counted as 0 for queries where no gold passage appears in the returned list.

### 10.2 Success criteria

These are targets for judging whether the system works, **not** predictions of the result. A missed target is a finding to analyse, not a failure to hide.

| Criterion | Target | What a miss would mean |
|---|---|---|
| Dense Recall@5, monolingual | ≥ 0.85 | Segmentation or encoder choice is wrong |
| Dense Recall@5, cross-lingual | ≥ 0.70 | The encoder is not genuinely aligning the two languages |
| Dense Recall@5, code-mixed | ≥ 0.60 | Query normalization and transliteration need work |
| Hybrid Recall@5 vs best single retriever | ≥ +2 points | Fusion is not adding signal; report honestly and analyse why |
| Answerability F1 (UNANSWERABLE class) | ≥ 0.75 | Threshold calibration or the signal set is inadequate |
| Citation Support Rate, RAG arms | ≥ 0.85 | The generator is drifting away from its context |
| Citation Support Rate, closed-book arm | expected far below the RAG arms | If it is not, RAG is adding nothing and H3 is false |
| Cohen's κ, answerability second pass | ≥ 0.70 | The annotation guideline is ambiguous; fix it before trusting any result |

### 10.3 What the numbers will not prove

A 400-item, single-domain, two-language dataset annotated largely by one person supports statements about *this corpus under this protocol*. It does not support claims about Indic question answering in general, about languages not tested, or about how these model rankings would hold at production scale. Every reported table carries this caveat, and the paper's limitations section states it in full.

---

## 11. Deliverables

| Brief §11 item | Repository path | Status |
|---|---|---|
| Source code with README and execution instructions | `src/indicrag/`, `README.md` | Not started |
| Research paper (IEEE style, Word and PDF) | `paper/IndicRAG-QA.docx`, `paper/IndicRAG-QA.pdf` | Not started |
| Presentation slides | `paper/slides.pptx` | Not started |
| Dataset / document collection description and QA set | `docs/PRD.md` §5–6, `evals/gold.jsonl`, `data/corpus_manifest.jsonl` | This document; data not started |
| Retrieval, QA, answerability and error-analysis results | `evals/report-*.txt` | Not started |
| Working demonstration | `indicrag ask`, optional local web interface | Not started |

---

## 12. Risks

**Summary only; the full register with triggers and owners is in `docs/PLAN.md` §5.** The three that can sink the project:

1. **Devanagari text extraction corruption.** Hindi government PDFs frequently use legacy or subset-embedded fonts that extract with reordered matras or dropped conjuncts. Text that looks like Devanagari but is character-level garbage would poison both the index and the gold answers, and would not be obvious at a glance. Detected by an automated well-formedness check plus manual inspection of a 30-passage sample; mitigated by preferring HTML sources and, failing that, OCR.
2. **Annotation artefact inflating the lexical baseline.** Addressed by the leakage control in §6.6.
3. **CPU throughput.** A four-arm generation sweep over 400 questions at two model sizes is several hours of compute. Mitigated by aggressive caching (NFR-4) and by treating the 3B model as batch-only.

---

## 13. Glossary

| Term | Meaning here |
|---|---|
| **Passage** | A retrievable chunk of a document, 120–220 tokens, with a stable identifier |
| **Code-mixed / Hinglish** | Hindi-English mixing written in Latin script, e.g. *"eligibility kya hai"* |
| **Cross-lingual retrieval** | A query in one language retrieving evidence written in another |
| **Closed-book** | Generation with no retrieved context — the model answers from its parameters alone |
| **Oracle context** | Generation given the gold passage directly, bypassing retrieval; the generation ceiling |
| **Answerability** | The binary decision to answer or abstain |
| **Citation Support Rate** | Fraction of answered questions whose answer is supported by the passage it cites |
| **Near-miss** | An unanswerable question whose topic *is* in the corpus but whose specific fact is not |
| **RRF** | Reciprocal Rank Fusion, `Σ 1/(60 + rank)` — rank-based hybrid fusion needing no score normalization |

---

## 14. Known gaps and non-goals

**Stated plainly so that no reader mistakes an untested area for a validated one.**

- **One Indic language.** Hindi only. Nothing here demonstrates behaviour on Dravidian or other Indo-Aryan scripts, and encoder quality varies substantially across them.
- **One domain.** Scholarship schemes. Legal, medical or conversational text would behave differently.
- **No fine-tuning.** All encoders and generators are off-the-shelf; §4 explains why.
- **A single primary annotator.** Mitigated by a 15% second pass with κ reported, but not eliminated.
- **A small generator.** A 1.5B or 3B quantized model is weaker than a frontier API model. The comparison against a larger instruction-tuned model is listed as bonus work and may not be completed.
- **No multi-hop question answering.** Questions requiring composition across two passages are excluded at annotation time, and the exclusion is recorded.
- **A snapshot, not a live system.** Answers reflect the documents as retrieved on a recorded date; schemes change.
- **Not advice.** The system reports what documents say. It is not a source of financial, legal or eligibility guidance, and the demo states this.
