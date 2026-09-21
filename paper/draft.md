# IndicRAG-QA — paper draft

> **PLANNING DOCUMENT. Not the paper.**
>
> This file is the outline, the figure list, the writing order and the notes on
> what each section is waiting for. It is kept because that reasoning is still
> useful, but its Abstract and Section I are **superseded** by
> `sections-I-abstract.md` and are left here only as the drafts they came from.
>
> The paper is assembled by `python scripts/build-paper.py` into `paper.md`,
> from `sections-I-abstract.md`, `section-II-related-work.md`,
> `sections-III-IV-V.md`, `sections-VI-results.md` (pending) and
> `sections-VII-X.md`. Edit those, not `paper.md`, and not this file.


**Status: working draft.** Every number below is tagged with its provenance.
`[PROBE]` marks a result from the 180 synthetic retrieval probes; `[GOLD]` marks
one from the human-verified 400-item set, which is still being built. No `[GOLD]`
number exists yet. Nothing tagged `[PROBE]` may appear in the submitted paper as
a headline result — the probes exist to catch pipeline errors early, and their
composition (120 of 180 monolingual, one shape quoting the passage directly)
flatters lexical retrieval in a way the gold set will not.

Target: IEEE conference format, two columns, 6–8 pages.
Course: CSET 346, Bennett University.

---

## Title

**Script-Aware Fusion for Cross-Lingual Retrieval: Why Naive Hybrid Retrieval
Fails on Indic and Code-Mixed Queries**

*Alternative, if the fusion result does not survive the gold set:*
IndicRAG-QA: Evidence-Grounded Cross-Lingual Question Answering for Indic and
Code-Mixed Queries

The first title is preferred. It names a specific, falsifiable mechanism rather
than describing a system, and the mechanism is the part of this work that is
novel. A paper titled after its system invites the reader to ask "so what?"; a
paper titled after a failure mode answers that in advance.

---

## Abstract

*(~180 words. Draft — numbers to be replaced with `[GOLD]` values.)*

> Retrieval-augmented generation over multilingual corpora is commonly built by
> fusing a lexical retriever with a dense one, on the assumption that their
> errors are complementary. We show that this assumption breaks in a specific
> and correctable way when the query and the evidence are in different scripts.
> A lexical retriever cannot return a passage whose script differs from the
> query's, because the two share almost no tokens; under rank fusion, such a
> passage collects one vote where a same-script passage collects two, and is
> therefore penalised for the lexical retriever's *blindness* rather than for
> its own irrelevance. On a bilingual Hindi–English corpus of Indian government
> welfare schemes, plain Reciprocal Rank Fusion scores 0.022 Recall@5 on
> cross-lingual queries against 0.125 for its own dense component. We propose
> **script-aware fusion**, which scores each passage over the retrievers
> eligible to return it, and recover 0.153 cross-lingual and 0.173 code-mixed
> Recall@5 (p = 0.0001, p = 0.0002), at a cost of 0.044 on monolingual queries.
> We further report that a retrieval-score answerability threshold detects 0.000
> of false-premise questions, and release a 400-item bilingual QA set with an
> explicit query-language × evidence-language matrix.

---

## I. Introduction

**Material available: complete.** Draft below.

The framing, in four moves:

1. **The access gap.** Public-interest information in India is published in one
   language and asked about in another. A citizen checking eligibility for a
   welfare scheme reads a circular issued in English but asks in Hindi, or — far
   more commonly — in Romanized Hindi-English code-mix: *"Scholarship ke liye
   minimum eligibility kya hai?"* Keyword search fails on three mismatches at
   once: different language, different script, different surface words.

2. **The standard remedy, and why it is not enough.** Multilingual dense
   retrieval bridges the semantic gap, and hybrid retrieval is the accepted
   best practice for combining it with lexical precision. We show hybrid
   retrieval, built the usual way, *actively harms* the cross-lingual case.

3. **The grounding requirement.** For documents governing who receives public
   money, a confident wrong answer is worse than no answer. The system must
   abstain, and be measured on abstaining correctly.

4. **Contributions.** Enumerated list, below.

### Contributions, as claimed

1. **Script-aware fusion**, a correction to rank fusion that recovers
   cross-lingual retrieval on a bilingual corpus, with the structural cause and a
   significance test. `[PROBE]`, to be confirmed `[GOLD]`.
   **Claim narrowed after the literature pass** — see `section-II-related-work.md`.
   RRF's single-retriever penalty is intended behaviour, not an oversight, and
   prior cross-lingual QA work already restricts the sparse index to the query's
   language. The contribution is to show the fusion *arithmetic* still assumes a
   retriever's silence is evidence, to quantify that cost, and to correct it.
2. **A bilingual (Hindi/English) evidence-grounded QA dataset** over Indian
   government schemes, with an explicit query-language × evidence-language
   matrix and a four-class unanswerable taxonomy.
3. **A decomposition showing retrieval-score answerability thresholds cannot
   detect false premises** — 0.000 recall on that class — with the mechanism.
4. **A negative result on off-the-shelf Indic MLM checkpoints** for retrieval,
   with the pooling confound controlled.

---

## II. Related Work

**Literature pass complete (2026-09-21). Drafted in
`paper/section-II-related-work.md`, with a bibliography checklist.**

The gate result: the framing must be narrowed from "we show" to "we characterise
and correct". Details and the honest novelty statement are in that file.

Four threads to cover, with the specific gap each leaves:

| Thread | Key work to cite | The gap we address |
|---|---|---|
| Multilingual dense retrieval | LaBSE, multilingual-E5, mBERT, XLM-R, LASER | Evaluated mostly on bitext mining and monolingual IR; cross-script *fusion* behaviour is not examined |
| Indic NLP resources | MuRIL, IndicBERT, IndicNLPSuite, AI4Bharat | Pretraining coverage is reported; retrieval-readiness off the shelf is assumed rather than measured |
| Hybrid / sparse-dense retrieval | RRF (Cormack et al.), ColBERT, SPLADE, weighted score fusion | Fusion is studied on monolingual collections where both retrievers can see every document — the precondition our result shows is violated |
| Answerability and abstention | SQuAD 2.0, Natural Questions unanswerable, selective prediction, RAG hallucination | Unanswerable classes are usually undifferentiated; the near-miss / false-premise distinction is what our Module 5 result turns on |
| Code-mixed IR | Hinglish datasets, transliteration-based retrieval, GLUECoS | Typically monolingual-target; the query-script vs evidence-script interaction is not isolated |

**Note for the pass:** the RRF row is the one that matters. If prior work has
already observed the eligibility problem under a different name, the framing in
§I must change from "we show" to "we quantify on Indic data". Search terms:
*rank fusion missing documents*, *unjudged documents fusion*, *asymmetric
retriever coverage*, *sparse retriever recall ceiling multilingual*.

---

## III. Dataset

**Material available: corpus complete; QA set in progress.**

### A. Corpus

40 schemes × 2 languages = **80 documents → 694 passages** (491 English, 203
Hindi), 83% carrying a real section path, all 80 passing Devanagari integrity
validation.

**The source deviation must be stated in the paper, not buried.** Official
ministry guidelines were attempted first and failed twice: `socialjustice.gov.in`
serves its Hindi paths as soft 404s (HTTP 200 with a "Page not found" body), and
`PMS_for_SCs_Scheme_Guidelines.pdf` — 18 pages, 2.8 MB — extracts to **17
characters**, being a scan, with no OCR available. An English-only corpus would
have defeated the cross-lingual question entirely. The corpus is therefore
parallel English/Hindi Wikipedia articles on Indian government schemes, paired
through the MediaWiki `langlinks` API so parallelism is asserted by Wikipedia
rather than assumed by us.

The cost, stated in §IX: the register is **encyclopedic rather than regulatory**.
No claim about retrieval over statutory text is supported by these results.

### B. Devanagari integrity validation

Worth a short subsection because it is a methodological contribution other Indic
NLP work can reuse. Five checks: misplaced dependent vowel signs (the classic
legacy-font reordering bug), dangling word-final viramas, Devanagari codepoint
ratio, Hindi stopword rate (catches mojibake that is *inside* the Devanagari
block but is not Hindi), and replacement characters.

One finding worth reporting: three clean documents initially failed the virama
check because Hindi prose writes **ZWNJ after a virama** to force an explicit
halant over a conjunct ligature (उद्देश्‍य). ZWNJ sits outside the Devanagari
block, so it splits the word token and leaves the first half apparently ending in
a bare virama. Any Indic pipeline doing token-level validation will hit this.

### C. QA set

Target **400 = 320 answerable + 80 unanswerable**.

Answerable language-pair matrix (§6.2 of the PRD) — the design that makes the
Module 2 comparison possible:

| Query ↓ / Evidence → | English | Hindi | Total |
|---|---|---|---|
| English | 70 | 45 | 115 |
| Hindi | 45 | 60 | 105 |
| Hinglish | 55 | 45 | 100 |
| **Total** | 170 | 150 | **320** |

Unanswerable taxonomy: out-of-scope 25, **near-miss 30**, **false-premise 15**,
under-specified 10. The two bolded classes carry the Module 5 result; out-of-scope
questions are trivially rejected and prove nothing.

**Construction.** Candidates are bootstrapped with a local quantized model and
then human-verified; items remain `verified: false` until a person confirms the
question is answerable from the cited passage and corrects the gold answer to the
document's exact wording. Cross-lingual items translate the *question only* and
leave the gold passage in place — translating the passage would silently convert
a cross-lingual item into a monolingual one.

**Status:** 150/400 generated (70 `en→en`, 80 unanswerable); 250 in progress.
`[GOLD]` numbers pending. Cohen's κ on the second-pass sample: **pending**.

### D. Leakage control

LLM-generated questions tend to quote their source passage, which inflates BM25
and would make the dense-vs-lexical comparison an artefact. We measure
question↔passage Jaccard overlap and report the distribution; on the `en→en`
cell generated so far, median overlap is **0.075** (max 0.186) `[GOLD-partial]`.
The headline retrieval comparison is additionally reported on the low-overlap
tertile.

---

## IV. System

**Material available: complete.**

Pipeline: fetch → Devanagari validation → normalization → structure-aware
segmentation → {lexical, dense} indexing → query language ID → retrieval →
fusion → generation → answerability.

Subsections to write:

- **Normalization**, run identically on passages and queries. Devanagari↔ASCII
  digits, six surface forms of an amount collapsed to one key
  (`Rs. 3,50,000` / `₹3.5 lakh` / `३,५०,०००` / `350000/-`), danda as a sentence
  terminator, ZWNJ handling.
- **Query language identification**, rule-based, three classes. The
  ambiguous-marker split is worth one sentence: `the`, `is` and `to` are valid
  Romanizations of थे, इस and तो and also the commonest English words, so they
  corroborate but never trigger the Code-Mixed label.
- **Retrieval arms**: TF-IDF, BM25, five encoders, two fusions.
- **Script-aware fusion** — the core of §VI. Formalise as: for passage *p* and
  retriever set *R*, score over `{r ∈ R : eligible(r, p)}` rather than over all
  of *R*, where a lexical retriever is ineligible for a passage whose script
  differs from the query's.
- **Answerability signals**: threshold, margin, scheme agreement, and (pending)
  generator self-report and NLI entailment.

---

## V. Experimental Setup

**Material available: complete.**

- **Hardware**, and it is relevant rather than incidental: Intel i5-11320H,
  4 cores, **no GPU**, 15.7 GB RAM. Every model choice follows from this, and it
  makes the work reproducible on a student laptop, which is worth stating.
- **Measured throughput** for the encoder sweep (694 passages): MiniLM 149.6 s,
  E5-base ~380 s, LaBSE ~600 s, MuRIL 632 s. Note the non-obvious result that
  throughput does **not** track parameter count — MuRIL (236M) is slower than
  LaBSE (471M) because the MLM arm runs a hand-rolled loop rather than
  SentenceTransformer's batching.
- **Splits**: dev 120 for tuning α, τ, k; test 280 scored once. Seed 20260922.
- **Statistics**: percentile bootstrap (1000 resamples) for intervals; paired
  two-sided randomization test (10000 trials) between systems. A permutation
  test rather than a t-test because per-query recall is bounded and discrete.

---

## VI. Results

### A. Lexical vs dense retrieval (Module 1)

`[PROBE]` Recall@5: BM25 0.499, TF-IDF 0.501, E5-base 0.449, LaBSE 0.361,
MiniLM 0.303, MuRIL 0.238 (mean pooling, the better of two).

**This table must not be presented as supporting H1.** On the probe set BM25
leads overall, and that is an artefact of probe composition, not evidence
against dense retrieval. Say so in the text and defer to the per-slice table.

### B. The language-pair breakdown (Module 2) — the real M1 result

`[PROBE]` Recall@5:

| System | monolingual | cross-lingual | code-mixed |
|---|---|---|---|
| BM25 | 0.740 | **0.006** | **0.028** |
| MiniLM-L12 | 0.394 | 0.136 | 0.102 |
| E5-base | 0.610 | 0.125 | 0.132 |
| LaBSE | 0.435 | 0.097 | **0.332** |
| MuRIL (mean) | 0.357 | **0.000** | **0.000** |

Three things to draw out:

1. **BM25 does not cross the boundary at all.** 0.006 is not degradation, it is
   absence. A Romanized or Devanagari query and an English passage share
   essentially no tokens.
2. **MuRIL, the Indic-pretrained model, scores 0.000 on both cross-script
   slices.** The model pretrained on seventeen Indian languages is the one that
   cannot retrieve between two of them. Both pooling strategies were run and the
   better reported, so this is not a pooling artefact. Pretraining-language
   coverage is not a substitute for retrieval training.
3. **LaBSE dominates code-mixed (0.332) but not cross-lingual (0.097).** This
   contradicts our own prior expectation — LaBSE is a bitext-mining model and we
   expected it to lead the cross-lingual slice. If it survives the gold set, the
   practical implication is to select an encoder by **query script**, not by
   advertised language coverage. Flag explicitly as unexpected.

### C. Script-aware fusion (Module 3) — the paper's central result

`[PROBE]` Recall@5, script-aware vs plain RRF, paired randomization test:

| slice | plain RRF | script-aware | Δ | p |
|---|---|---|---|---|
| cross-lingual | 0.022 | **0.153** | +0.131 | 0.0001 |
| code-mixed | 0.028 | **0.173** | +0.145 | 0.0002 |
| monolingual | 0.712 | 0.668 | −0.044 | 0.0319 |
| all | 0.483 | 0.500 | +0.017 | 0.3195 |

Against **dense alone**, which is the question of whether hybrid earns its keep
at all: +0.050 overall, p = 0.0086.

**The H2 revision is the honest and interesting part.** An α-sweep initially
said hybrid retrieval was not supported: no interior α beat both endpoints by
more than 0.001. That verdict was correct about the method it tested and wrong
about the hypothesis. Hybrid *does* beat its components — but only once fusion
stops penalising passages for being written in the language the query was not.

**Report the monolingual cost plainly.** −0.044 at p = 0.032 is real. Removing
the lexical double-vote costs precision where both retrievers could see the same
passage. On this corpus it is a good trade because monolingual was already strong
and cross-lingual was near zero. It is a trade, not a free win.

### D. Direct LLM vs RAG (Module 4)

**Not yet run.** Requires the generation arms (A closed-book / B RAG-dense /
C RAG-hybrid / D oracle). Arm D is the one that separates retrieval error from
generation error and must not be cut.

### E. Answerability (Module 5)

`[PROBE + scaffold]` Retrieval-threshold signal: accuracy 0.695, precision 0.533,
recall 0.291, **F1 0.376** against a 0.75 target.

The aggregate understates how badly it fails. Per-class recall:

| class | recall | n |
|---|---|---|
| out-of-scope | 0.688 | 16 |
| under-specified | 0.286 | 7 |
| near-miss | **0.143** | 21 |
| false-premise | **0.000** | 11 |

**The 0.000 is the mechanism, not a bug.** *"How much laptop allowance does
scheme X provide?"* retrieves X's passages at high similarity, because the scheme
name matches and the topic is right. A retrieval score measures whether a
question is **about** something in the corpus; it cannot measure whether the
specific asserted fact **exists**. No choice of τ fixes this — the signal does
not carry the information. This is the argument for the NLI entailment check.

**A fairness result to report:** Code-Mixed queries abstain at 0.225 against
0.132 for Indic — roughly 70% more often — while their *over*-abstention is the
lowest at 0.091. A single global threshold refuses Hinglish users partly for
their query script rather than for missing evidence.

---

## VII. Error Analysis (Module 6)

**Material available: 20 cases across 10 taxonomy categories, committed.**

Selection is by rule, not by hand: within each category, the failure where the
system was most confident and most wrong. State this explicitly — picking
examples after seeing which look interesting is how an error analysis becomes a
set of anecdotes.

**The headline case (E02) is how the fusion result was found**, and the paper
should say so: the aggregate tables showed hybrid underperforming and gave no
hint why. Reading twenty cases did. Of the 20 failures, e5-base alone had
retrieved the gold passage in 8; plain RRF lost all 8.

Categories to illustrate with one case each: code-mixing, transliteration
variation, named entity, amount, date, ambiguity, near-duplicate schemes,
cross-lingual gap, long-tail scheme, short query.

---

## VIII. Discussion

To write once `[GOLD]` numbers exist. The argument:

1. Fusion assumes retriever errors are complementary. Across a script boundary
   they are not merely correlated, they are *structurally asymmetric* — one
   retriever has zero recall by construction.
2. This generalises beyond Hindi–English to any retrieval setting where one
   retriever cannot see part of the collection: cross-script pairs, modality
   gaps, access-controlled shards.
3. Practical guidance: check retriever *coverage* before fusing, not just
   retriever quality.

---

## IX. Limitations

Write this section properly; it is where the work earns trust.

- **Encyclopedic, not regulatory corpus.** §III-A. No claim about statutory text.
- **One Indic language.** Hindi only.
- **400 items, one primary annotator**, κ to be reported.
- **Probe-based preliminary results.** Any number still tagged `[PROBE]` at
  submission must be labelled as such in the paper.
- **Small quantized generator** (Qwen2.5-3B Q4_K_M), CPU-only.
- **No fine-tuning** of any component.
- **Script-aware fusion is evaluated on a two-script corpus.** Behaviour with
  three or more scripts, or with partially-overlapping scripts, is untested.
- **The monolingual regression** (−0.044) may matter more on a corpus where
  cross-lingual queries are rare.

---

## X. Conclusion

To write last.

---

## Figures

| # | Figure | Source | Status |
|---|---|---|---|
| 1 | Pipeline diagram | ARCHITECTURE §1 | ready to redraw |
| 2 | Recall@5 by query type × system (grouped bars) | §VI-B | data ready `[PROBE]` |
| 3 | Script-aware vs plain RRF by slice | §VI-C | data ready `[PROBE]` |
| 4 | α-sweep on dev, endpoints marked | evals/ | data ready `[PROBE]` |
| 5 | Per-class answerability recall | §VI-E | data ready `[PROBE]` |
| 6 | Confusion matrix, best answerability signal | §VI-E | data ready `[PROBE]` |
| 7 | Question↔passage overlap distribution | §III-D | partial |

Figure 7 is unusual in a student paper and should stay: showing the artefact you
controlled for is more persuasive than asserting there wasn't one.

---

## Writing order

1. §III, §IV, §V — fully determined by work already done, no dependency on
   pending numbers. **Write these first.**
2. §II — needs a literature pass; the RRF prior-art check gates the §I framing.
3. §VI, §VII — skeleton exists; swap `[PROBE]` for `[GOLD]` when the set lands.
4. §I, §VIII, §IX, §X, Abstract — write last, once the claims are settled.

## Deliverable note

The brief requires IEEE-style **Word and PDF**. This draft is Markdown for
editing; conversion to the IEEE two-column template is a separate step and should
happen only once the content is stable, because reflowing a two-column layout
around changing tables is wasted effort.
