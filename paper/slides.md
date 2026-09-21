# Slide deck — IndicRAG-QA

16 slides for a ~14 minute talk. One idea per slide; the speaker notes carry the
argument, the slide carries the evidence.

The deck is built around the fusion finding rather than around the system. A
system walkthrough invites "so what?" at the end; a failure mode answers it at
the start. Slides 5–8 are the spine and 11–12 are the payoff — if time runs short cut 3,
then 13, then 14. Never 12: it is the one that stops the talk being a sales
pitch.

Retrieval figures are `[PROBE]` (180 synthetic probes). Any figure still marked
`[PROBE]` on the day must be labelled on the slide, not just in the notes.

---

## 1. Title

**Script-Aware Fusion for Cross-Lingual Retrieval**
Why naive hybrid retrieval fails on Indic and code-mixed queries

IndicRAG-QA · CSET 346 · Bennett University

*Notes:* Name the mechanism, not the system. Fifteen seconds.

---

## 2. The query that motivates the work

> *"Scholarship ke liye minimum eligibility kya hai?"*

Three mismatches at once:

| | Query | Document |
|---|---|---|
| Language | Hindi | English |
| Script | Latin | Latin / Devanagari |
| Surface form | *eligibility* / *yogyata* / *योग्यता* | one spelling |

*Notes:* This is not a constructed example; it is how the question is actually
typed. Each mismatch is individually well studied. Their conjunction is the
ordinary case for a large population, and that is the setting we measure.

---

## 3. Why the stakes differ from open-domain QA

These documents govern **who receives public money**.

A confident wrong answer about an eligibility ceiling is worse than no answer.

→ The system must be able to abstain, and be **measured on abstaining in the
right places**.

*Notes:* Motivates Module 5 and the unanswerable taxonomy. Cuttable if short on
time, but it is what makes the answerability work more than a checkbox.

---

## 4. The corpus, and one honest deviation

- Parallel Hindi/English corpus, Indian government welfare schemes
- Bilingual **by construction**: each scheme has two independently authored
  documents, not a translation
- **400-item QA set**, query-language × evidence-language matrix, 6 cells
- 80 unanswerable items across 4 classes

**Deviation, stated up front:** intended to use official scheme PDFs. Soft 404s
and image-only scans — one 18-page document yielded 17 characters of extractable
text. Corpus is encyclopedic, not regulatory.

*Notes:* Say the deviation out loud. Someone will ask; answering before they do
is worth more than the slide costs.

---

## 5. The standard recipe

Language/script mismatch → **multilingual dense retrieval**

Dense retrieval weak on rare terms (scheme names, amounts, section numbers)
→ add a **lexical retriever** and fuse

Reciprocal Rank Fusion: `score(d) = Σᵢ 1/(k + rankᵢ(d))`

No normalisation needed. Standard, strong baseline.

*Notes:* Set this up as the obviously correct thing to do, because it is. The
next slide is the turn.

---

## 6. It does not merely fail to help

**Recall@5, cross-lingual queries** `[PROBE]`

| System | Recall@5 |
|---|---|
| Dense alone (e5-base) | **0.125** |
| Hybrid RRF (plain) | **0.022** |

Hybrid scores **below the dense retriever it contains**.

*Notes:* Pause here. This is the slide the talk exists for. Hybrid retrieval is
supposed to be a safe default; on the case this system exists to serve it is
worse than one of its own components.

---

## 7. The cause is structural, not a tuning problem

A lexical retriever has **zero recall by construction** across a script
boundary — a Devanagari passage and a Latin query share no tokens.

RRF treats agreement as evidence:

- same-script passage → **2 votes**
- cross-script passage → **1 vote**

The cross-script passage is docked for the **lexical retriever's blindness**,
not for its own irrelevance.

> BM25 saying nothing about a Devanagari passage is not evidence against it.
> It is no evidence at all.

*Notes:* The α sweep found no interior weighting that beat either endpoint. It
was right about the method and wrong about the hypothesis — α was never the
problem.

---

## 8. Script-aware fusion

Score each candidate over the retrievers **eligible** to return it, not those
that did.

- No training, no tuning
- One lookup per candidate
- Query-time cost unchanged

| Slice | Plain RRF | Script-aware | p |
|---|---|---|---|
| cross-lingual | 0.022 | **0.153** | 0.0001 |
| code-mixed | 0.028 | **0.173** | 0.0002 |
| monolingual | 0.712 | 0.668 | 0.032 |
| overall | 0.483 | 0.500 | 0.32 |

*Notes:* Read the monolingual row out loud. It is a real cost — removing the
double vote loses precision where agreement genuinely was evidence.

---

## 9. Report the trade, not the aggregate

Overall Recall@5: 0.483 → 0.500, **p = 0.32** — indistinguishable from noise.

Reporting only the aggregate would hide **both** the 7× cross-lingual gain
**and** the 0.044 monolingual cost.

Good trade *on this corpus*: monolingual was already strong, cross-lingual was
near zero. On a mostly-monolingual deployment it would be a bad one.

*Notes:* This is the slide that earns trust. Aggregates that conceal both the
win and the cost are the default failure of results sections.

---

## 10. How it was actually found

Aggregate tables: "hybrid underperforms." No hint why.

**20 error cases, selected by rule** (most confident × most wrong, 2 per
category across 10 categories):

- Dense alone found the gold passage in **8**
- Plain RRF lost **all 8**

*Notes:* Method point, and the one most transferable to the audience: the
aggregate metric did not merely fail to show the mechanism, it pointed away from
it. Selection by rule, fixed before looking, is what keeps this from being
anecdote.

---

## 11. Does it actually help the answers?

Four arms, same generator, differing only in the evidence given.

| Arm | token-F1 | abstains | Citation Support |
|---|---|---|---|
| closed-book | 0.065 | 0.486 | **0.162** |
| RAG dense | 0.207 | 0.417 | 0.762 |
| RAG hybrid | 0.210 | 0.361 | **0.826** |
| oracle | 0.403 | 0.194 | 0.948 |

Closed-book answers 37 of 72 questions. **16%** of those answers are supported
by the evidence.

Script-aware fusion propagates: hybrid beats dense on support *and* abstains
less, same model, same questions.

*Notes:* This is H3 and it is the cleanest result in the deck. The closed-book
arm is not refusing — it is confidently reciting eligibility thresholds from
memory and getting them wrong.

---

## 12. But retrieval is not the bottleneck

| | |
|---|---|
| oracle token-F1 | 0.403 |
| full system token-F1 | 0.210 |
| **retrieval error** | **0.193** |
| **generation error** | **0.597** |

Fixing retrieval *entirely* buys 0.193. The 3B generator is losing 0.597.

**Say this out loud:** our own contribution improves the smaller of the two.

*Notes:* The slide that stops the talk being a sales pitch. The fusion result is
a claim about retrieval, measured as retrieval. It is not a claim that retrieval
is what limits answer quality here. This ratio is a property of a 3B quantized
model, not a law — but it is why the oracle arm exists.

---

## 13. Where this generalises

Not about Hindi. Not about script.

Anywhere rank fusion runs over retrievers with **asymmetric coverage**:

- text × image/table retrieval — each blind to the other's modality
- access-controlled shards — one retriever permitted a subset
- specialist × general index over partly-covered collections

**Guidance:** before fusing, check retriever *coverage*, not only *quality*.

*Notes:* Two retrievers of equal accuracy can combine well or destructively
depending on whether each *could* have retrieved what the other did.

---

## 14. A second negative result: answerability

Retrieval-score thresholds carry **no information** about answerability here.

| | answerable | unanswerable |
|---|---|---|
| BM25 | 13.55 | **13.82** |
| TF-IDF | 0.1424 | **0.1485** |
| RRF | 0.0327 | 0.0325 |

Unanswerable questions score *higher*. Precision stays at the 0.20 base rate at
every threshold.

**Why:** 55 of 80 unanswerable items are about schemes that ARE in the corpus.
A retrieval threshold detects corpus absence — and answerability is not corpus
absence.

*Notes:* A property of the signal, not the calibration. The fitted operating
point swings from 0.000 recall to 0.93 depending only on the split, while the
curve underneath stays flat — which is why the separation table matters more
than any single number. Motivates the NLI signal.

---

## 15. Scope, stated plainly

- Encyclopedic corpus, **not** regulatory text
- **One** Indic language; two scripts only
- 400 items, single primary annotator, blind 15% second pass, κ gated at 0.70
- Retrieval numbers are `[PROBE]` pending gold-set verification
- Qwen2.5-3B Q4_K_M on CPU; nothing fine-tuned

*Notes:* Say these before the Q&A rather than during it.

---

## 16. Conclusion

Hybrid retrieval inverts cross-lingually because **fusion treats a retriever's
silence as evidence**.

Fix the arithmetic, not the model: **7× cross-lingual**, **6× code-mixed**, no
training.

Found by reading 20 cases, not by reading a table.

*Notes:* Three sentences, then stop.
