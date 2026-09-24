# Slide deck — IndicRAG-QA

16 slides for a ~14 minute talk. One idea per slide; the speaker notes carry the
argument, the slide carries the evidence.

The deck is built around the fusion finding rather than around the system. A
system walkthrough invites "so what?" at the end; a failure mode answers it at
the start. Slides 5–8 are the spine and 11–12 are the payoff — if time runs short cut 3,
then 13, then 14. Never 12: it is the one that stops the talk being a sales
pitch.

Every figure is from the model-verified gold set, on its sealed test split.
Figures that ever come from another set must be labelled on the slide, not just
in the notes.

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
- **393-item QA set** (313 answerable, 80 unanswerable in 4 classes),
  query-language × evidence-language matrix, 6 cells; sealed dev 120 / test 273
- **Verified by a language model, not a person** — disclosed on every report

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

**Recall@5, cross-lingual queries** (test split, n = 62)

| System | Recall@5 |
|---|---|
| Dense alone (e5-base) | **0.661** |
| Hybrid RRF (plain) | **0.274** |

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

*Notes:* The α sweep found no interior weighting significantly better than the better endpoint. It
was right about the method and wrong about the hypothesis — α was never the
problem.

---

## 8. Script-aware fusion

Score each candidate over the retrievers **eligible** to return it, not those
that did.

- No training, no tuning
- One lookup per candidate
- Query-time cost unchanged

Test split, Recall@5:

| Slice | Plain RRF | Script-aware | Dense alone |
|---|---|---|---|
| cross-lingual | 0.274 | **0.685** | 0.661 |
| code-mixed | 0.514 | **0.556** | 0.472 |
| monolingual | 0.964 | 0.964 | 0.976 |
| overall | 0.618 | **0.750** | 0.720 |

vs plain RRF: **+0.411** cross-lingual, +0.132 overall (p = 0.0001).
vs dense alone: +0.030 overall (p = 0.052); significant only code-mixed (+0.085, p = 0.004).

*Notes:* Two comparisons, two answers. The repair to plain fusion is large and
costs nothing monolingually. The gain over a good dense retriever is small —
say so before someone asks.

---

## 9. What verification changed

On 180 synthetic probes the method looked like a trade: +0.131 cross-lingual,
−0.044 monolingual. On the unverified questions it beat dense alone by
**+0.133** cross-lingually.

On the verified test split: the repair is **larger** (+0.411), the monolingual
cost is **gone**, and the gain over dense alone is **mostly gone** (+0.024
cross-lingual, p = 0.62).

Claim what survives: standard fusion does harm across scripts; the fix removes
it. Not: fusion beats a strong dense retriever.

*Notes:* This is the slide that earns trust. The probe numbers were not wrong
about the mechanism, only about the magnitudes.

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
Test split, 72-item stratified sample.

| Arm | token-F1 | abstains | Citation Support |
|---|---|---|---|
| closed-book | 0.050 | 0.319 | **0.184** |
| RAG dense | 0.446 | 0.250 | **0.870** |
| RAG hybrid | 0.418 | 0.208 | 0.860 |
| oracle | 0.641 | 0.083 | 0.985 |

Closed-book answers 49 of 72 questions; fewer than **1 in 5** of those answers
is supported. Paired: **+0.684, p = 0.0001**.

**Hybrid vs dense is NOT significant** (−0.027 token-F1, p = 0.55).
Retrieval-or-not is the large effect; *which* retriever does not show up
downstream at n = 72.

*Notes:* This is H3 and it is the cleanest result in the deck. The closed-book
arm is not refusing — it is confidently reciting scheme figures from memory and
getting them wrong. Say the second line out loud: the fusion gain is real as
retrieval, and small enough over dense alone that 72 questions cannot see it.

---

## 12. Where the errors are

Test split, 72-item sample.

| | |
|---|---|
| oracle token-F1 | 0.641 |
| full system token-F1 | 0.418 |
| **retrieval error** | **0.223** |
| **generation error** | **0.359** |

Both matter. The generator loses more, by 1.6×.

**Say this out loud:** before verification this was 0.597 against 0.193 — a
factor of three — and we advised fixing the generator first. On verified items
(and a different sample) it narrowed to 1.6. That advice did not survive.

*Notes:* The slide that stops the talk being a sales pitch: our headline number
moved when the data got better, and we show it moving. The ratio is a property
of a 3B quantized model, not a law — but it is why the oracle arm exists.

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

## 14. Answerability: which score you threshold matters

Median top retrieval score, verified set:

| | answerable | unanswerable |
|---|---|---|
| BM25 | **25.47** | 13.82 |
| TF-IDF | **0.238** | 0.145 |
| RRF (fused) | 0.0328 | 0.0325 |

BM25 threshold on test: F1 **0.525**, all out-of-scope caught — but only
**0.522** of near-miss questions. The fused RRF score carries almost nothing:
rank fusion discards the magnitudes.

Before verification the BM25 medians were reversed (13.55 vs 13.82): questions
that did not name their scheme retrieved weakly whether or not they were
answerable.

Four signals, same 85 held-out items (enriched: 57 unanswerable, 28 answerable).

| signal | catches unanswerable | answers answerable |
|---|---|---|
| threshold on fused score | 53/57 | **2/28** |
| calibrated combination | 44/57 | 11/28 |
| generator self-report | 52/57 | **20/28** |

Generator near-miss recall: **22 of 23**.

*Notes:* The threshold row is the one to dwell on — high recall, achieved by
refusing almost every question. Only the generator, which reads the passage,
catches near-misses while still answering. Adding NLI on top changes nothing:
of the 25 it answers, 5 are unanswerable, and no entailment threshold improves
on that. In deployment: BM25 floor first, generator abstention second.

---

## 15. Scope, stated plainly

- Encyclopedic corpus, **not** regulatory text
- **One** Indic language; two scripts only
- 393 items, **verified by a model, not a person**; blind second pass κ = 1.000
  is one model agreeing with itself, not the 0.70 inter-annotator gate
- Probe-set numbers kept only where the mechanism was found; results are test split
- Qwen2.5-3B Q4_K_M on CPU; nothing fine-tuned

*Notes:* Say these before the Q&A rather than during it.

---

## 16. Conclusion

Hybrid retrieval inverts cross-lingually because **fusion treats a retriever's
silence as evidence**.

Fix the arithmetic, not the model: cross-lingual **0.274 → 0.685**, no
monolingual cost, no training — and no large gain over dense alone.

Found by reading 20 cases, not by reading a table.

*Notes:* Three sentences, then stop.
