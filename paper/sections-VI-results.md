# Section VI — Results

All five modules are written from measured numbers.

**Three different bases, and they must not be conflated.** The retrieval figures
(§VI-B to §VI-F) are `[PROBE]`: 180 synthetic probes over the real corpus. The
question-answering figures (§VI-G) are measured on a 72-item sample of the gold
set, stratified across the six language-pair cells. The answerability figures
(§VI-H) use all 400 gold items. **None of the gold items is human-verified
yet**, so nothing here may be quoted as a verified gold-set result; §VI-A states
what the probes can and cannot support.

---

## VI. Results

### A. What the probe set can support

All results in this section are measured on 180 synthetic probes constructed
from real passages and real article titles, with ground truth by construction.
The human-verified 400-item set exists but has not completed verification, and
no number here is reported as a gold-set result.

The probe set's composition bounds what it can show. 120 of the 180 probes are
monolingual, and a substantial share are *fragment*-shaped, lifting phrasing
directly from the target passage. This flatters lexical retrieval considerably.
The cross-lingual probes have near-zero lexical overlap by construction and are
the only shape that speaks honestly to cross-lingual retrieval.

We therefore draw no conclusion from these probes about the *ranking* of
retrievers overall. What the probe set does support is the finding this paper
turns on, because that finding is structural rather than statistical: a lexical
retriever's recall across a script boundary is zero by construction, and a
probe set cannot manufacture that. The magnitudes below may move on the gold
set. The mechanism will not.

The corpus is 694 passages over 40 schemes in English and Hindi, with all 80
documents passing Devanagari integrity validation.

### A.1 A first look at the same experiments on the gold set

The retrieval sections below are probe-based. We have since run the identical
evaluation over the 320 answerable gold items, and report the comparison here
because two of the probe-based conclusions do not survive it. These items are
**not verified either**, so this is a second preliminary source rather than a
correction; the headline sections keep their probe numbers until verification
settles which to report.

**What the gold set confirms, more strongly than the probes did.** Lexical
retrieval still collapses across the language boundary: BM25 reaches 0.977
monolingual and 0.056 cross-lingual, a ratio worse than on the probes. MuRIL
still scores exactly 0.000 cross-lingually. And the fusion penalty is still
there and still visible in the same shape — plain RRF reaches 0.111 on the
cross-lingual slice while the dense retriever it contains reaches 0.367.

**What it contradicts.** Two things.

First, §VI-F's LaBSE result reverses. On the probes LaBSE led the code-mixed
slice at 0.332; on the gold set it scores 0.050 there, and lexical retrieval
leads at 0.230. The gold code-mixed questions are Romanized Hindi carrying
English scheme names, which BM25 can match directly against the English
passages, and the probe set's code-mixed shape evidently did not reproduce that.
§VI-F should be read as a claim about the probe set until verification decides.

Second, the α sweep reverses, and with it the H2 verdict. On the probes no
interior weighting beat the endpoints by more than 0.001. On the gold set
α = 0.4 reaches 0.547 against 0.484 for pure lexical and 0.416 for pure dense —
a margin of 0.062, comfortably outside the noise band. Weighted hybrid fusion
does help on these questions, which is the opposite of what §VI-E reports from
the probes.

That divergence is itself evidence for the caution in §VI-A. A probe set built
by lifting phrasing from target passages flatters lexical retrieval and
compresses the differences between fusion settings; the gold questions are
written to be answered rather than to be matched, and they separate the methods
that the probes could not. The full table is committed as
`evals/report-retrieval-goldset.txt`.

**Third, and most consequentially, the central result is larger on the gold set
and its cost disappears.** Script-aware fusion had never been measured on these
items at all — it was absent from the retrieval harness and its figures came
from a one-off script over the probes.

| Slice | plain RRF | script-aware | dense alone | SA − plain | SA − dense |
|---|---|---|---|---|---|
| cross-lingual (n=90) | 0.111 | **0.500** | 0.367 | **+0.389** * | **+0.133** * |
| code-mixed (n=100) | 0.210 | 0.230 | 0.090 | +0.020 | **+0.140** * |
| monolingual (n=130) | 0.969 | 0.962 | 0.962 | −0.008 | +0.000 |
| **all (n=320)** | 0.491 | **0.603** | 0.522 | **+0.113** * | **+0.081** * |

Figure 6 plots this beside Figure 3's probe version.

Three of the probe-based qualifications in §VI-E do not survive. The
cross-lingual gain roughly triples, from +0.131 to +0.389 (p = 0.0001). The
monolingual regression, which §VIII-C treats as the honest price of the method,
falls from −0.044 (p = 0.032) to −0.008 (p = 1.00) — on these questions there is
no measurable cost. And the overall gain, which the probes could not
distinguish from noise at p = 0.32, becomes +0.113 at p = 0.0001, making
script-aware fusion the best of the nine systems evaluated at 0.603 Recall@5.

The code-mixed gain moves the other way, from +0.145 to +0.020 (p = 0.72), for
the same reason LaBSE's advantage evaporated: these questions carry English
scheme names that the lexical retriever can already match, so plain RRF was not
being penalised much to begin with. Against dense retrieval alone the gain
remains significant on every cross-script slice.

We do not restate the headline numbers on this basis, because these items are
unverified and the probe figures at least rest on ground truth that is correct
by construction. But the direction is worth being plain about: on real
questions the correction helps more and costs less than the probe set
suggested.

### B. Lexical against dense retrieval (Module 1)

**Recall@5, overall, with 95% bootstrap confidence intervals** `[PROBE]`

| System | R@1 | R@5 | MRR | nDCG@10 | R@5 95% CI |
|---|---|---|---|---|---|
| BM25 | 0.358 | 0.499 | 0.481 | 0.476 | [0.429, 0.565] |
| TF-IDF | 0.347 | 0.501 | 0.477 | 0.471 | [0.430, 0.569] |
| MiniLM-L12 (multilingual) | 0.122 | 0.303 | 0.280 | 0.282 | [0.239, 0.366] |
| multilingual-e5-base | 0.210 | 0.449 | 0.406 | 0.420 | [0.386, 0.516] |
| LaBSE | 0.175 | 0.361 | 0.361 | 0.341 | [0.298, 0.433] |
| MuRIL (mean-pooled) | 0.144 | 0.238 | 0.206 | 0.220 | [0.178, 0.299] |
| MuRIL (CLS-pooled) | 0.059 | 0.143 | 0.127 | 0.134 | [0.099, 0.193] |

BM25 leads overall, and the best dense encoder trails it by 0.049
(p = 0.10, not significant). We do **not** read this as evidence that lexical
retrieval beats dense retrieval on this task. It is a property of the probe mix
described in §VI-A — two thirds monolingual, many quoting the target passage —
and the honest statement is that this comparison is not established either way
on this data. The aggregate row is reported for completeness and the
language-pair breakdown that follows is what carries the finding.

`ai4bharat/indic-bert`, named in the project brief, could not be evaluated: the
repository is gated and returns HTTP 401 without an accepted licence.

### C. The language-pair breakdown (Module 2)

The aggregate table conceals the result. Split by language group, the same
systems behave in categorically different ways.

**Recall@5 by language group** `[PROBE]`

| System | Monolingual (n=120) | Cross-lingual (n=30) | Code-mixed (n=30) |
|---|---|---|---|
| BM25 | **0.740** | 0.006 | 0.028 |
| TF-IDF | 0.745 | 0.006 | 0.019 |
| MiniLM-L12 | 0.394 | 0.136 | 0.102 |
| multilingual-e5-base | 0.610 | 0.125 | 0.132 |
| LaBSE | 0.435 | 0.097 | **0.332** |
| MuRIL (mean-pooled) | 0.357 | **0.000** | **0.000** |

Figure 2 plots the same numbers. Lexical retrieval does not cross the language
boundary at all: BM25 scores 0.740 monolingual and 0.006 cross-lingual — a factor of more than a hundred. This is
not a weakness to be tuned away but the expected consequence of matching surface
tokens between a Devanagari query and a Latin-script passage that share
essentially none. It is the empirical form of the structural claim in §VIII, and
it is what makes the fusion behaviour in §VI-E a predictable consequence rather
than a surprise.

### D. Indic pretrained checkpoints underperform, with pooling controlled

MuRIL, pretrained on seventeen Indian languages, is the weakest dense encoder
tested at 0.238 Recall@5. It sits below MiniLM-L12 at 0.303, a model with
roughly half the parameters but trained with a retrieval objective — though we
note that this particular ordering is **not statistically established**: paired
over the 180 probes the gap is +0.065 at p = 0.069, which does not clear the
0.05 level we report against elsewhere. We ran both mean and CLS pooling and
report the better, so the result is not a pooling artefact; CLS pooling is worse
still at 0.143.

The sharpest form of the result needs no test at all, and is in the breakdown:
MuRIL scores **exactly 0.000** on both the cross-lingual and the code-mixed
slices — not a small number, but no correct retrieval in 60 queries. The model
pretrained on seventeen Indian languages is the one that cannot retrieve between
two of them.

The explanation is the known anisotropy of masked-language-model representations.
An MLM checkpoint is trained to predict masked tokens, not to place
paraphrases near one another, and its mean-pooled output occupies a narrow cone
in which cosine similarity discriminates poorly. Pretraining-language coverage
does not substitute for retrieval training. We report this because the brief
names these checkpoints and because the assumption that an Indic-pretrained
model is the right choice for an Indic retrieval task is a natural one to make.

### E. Script-aware fusion (Module 3)

Plain Reciprocal Rank Fusion over BM25 and multilingual-e5-base scores 0.022 on
cross-lingual queries — **below the 0.125 of the dense retriever it contains**.
Fusion is actively destroying information on precisely the slice it was added to
help. §VIII gives the mechanism; the correction is to score each candidate over
the retrievers *eligible* to return it rather than those that did.

**Recall@5, script-aware against plain RRF, paired randomization test (10 000
trials)** `[PROBE]`

| Slice | Plain RRF | Script-aware | Δ | p |
|---|---|---|---|---|
| cross-lingual | 0.022 | **0.153** | +0.131 | 0.0001 * |
| code-mixed | 0.028 | **0.173** | +0.145 | 0.0002 * |
| monolingual | 0.712 | 0.668 | −0.044 | 0.032 * |
| overall | 0.483 | 0.500 | +0.017 | 0.32 |

Figure 3 plots the table, with the monolingual regression on the same axes as
the gains rather than in a separate panel. Against dense alone — the comparison
that decides whether hybrid retrieval earns its place at all — script-aware
fusion gains +0.028 cross-lingual (p = 0.037),
+0.041 code-mixed (p = 0.074) and +0.050 overall (p = 0.0086).

Three things should be read from this table rather than one.

First, the gain on the target slices is large and significant. Second, the
monolingual regression is real, significant, and the price of the method:
removing the lexical double-vote costs precision where both retrievers could see
the same passage and agreement between them genuinely was evidence. Third, the
overall row is indistinguishable from noise, and a paper reporting only that row
would conceal both the gain and the cost.

This also revises the negative result of the α sweep, plotted as Figure 4, which
found no interior
weighting beating both endpoints by more than 0.001 and recorded the hybrid
hypothesis as unsupported. That verdict was correct about weighted fusion and
wrong about the hypothesis: hybrid retrieval does beat its components here, once
fusion stops penalising passages for their script.

### F. A caveat the fusion result does not remove

LaBSE, used alone, reaches **0.332** on the code-mixed slice against 0.173 for
script-aware fusion over BM25 and e5-base. Paired over the 30 code-mixed
queries that is +0.159 at **p = 0.0094**, so on code-mixed queries choosing a
different encoder genuinely outperforms correcting the fusion over a weaker one.

The reverse holds elsewhere, and equally significantly. On the monolingual slice
LaBSE loses to the fused system by 0.233 (p = 0.0001), and on the cross-lingual
slice it is behind by 0.056, which is within noise (p = 0.34). LaBSE is not a
better retriever; it is a differently shaped one, and the shape happens to suit
Romanized Hindi.

We state this plainly because it qualifies the practical reading of §VI-E. The
fusion correction is orthogonal to the choice of encoder: it is a statement
about how rankings are combined, and nothing prevents applying it over LaBSE,
which we have not tested and would expect to be stronger than either.

The result contradicts our own prior expectation, recorded before the
experiment, that LaBSE would lead the cross-lingual slice by virtue of its
translation-ranking objective. Instead it is the only model that handles
Romanized Hindi well. If this survives the gold set, the practical guidance is
to select the encoder by query *script* rather than by language coverage, and it
is the most immediately useful finding in this section. It needs the gold set
before it can be claimed.

### G. Direct LLM against RAG (Module 4)

We run four arms over a 72-item sample stratified across the six
language-pair cells: **A** closed-book, the model answering from its own
parameters with no passages; **B** retrieval-augmented with dense retrieval;
**C** retrieval-augmented with script-aware fusion; and **D** an oracle given
the gold passage directly. The generator is Qwen2.5-3B-Instruct at 4-bit
quantization throughout, so the arms differ only in the evidence they receive.

**Module 4, 72 items per arm** `[PROBE]`

| Arm | EM | token-F1 | abstains | Citation Support |
|---|---|---|---|---|
| A closed-book | 0.014 | 0.065 | 0.486 | 0.162 |
| B RAG-dense | 0.139 | 0.207 | 0.417 | 0.762 |
| C RAG-hybrid | 0.139 | **0.210** | 0.361 | **0.826** |
| D oracle | 0.264 | 0.403 | 0.194 | 0.948 |

**H3 is supported, and by a wide margin.** Citation Support Rate — the share of
answered questions whose answer the cited evidence actually supports — rises
from 0.162 closed-book to 0.826 with retrieval, and to 0.948 given the gold
passage. The closed-book arm answers 37 of 72 questions and only 16% of those
answers are supported by the evidence that would have justified them. It is not
declining to answer; it is answering from memory about specific eligibility
thresholds and scheme parameters, and is usually wrong. This is the behaviour
the system exists to prevent, and retrieval prevents most of it.

Abstention falls monotonically as the evidence improves, from 0.486 to 0.417 to
0.361 to 0.194. The generator abstains less when it has something to work from,
which is the desired direction: the abstentions are responding to the absence of
evidence rather than to a fixed conservatism.

**The fusion result does not measurably propagate to generation, and the
unpaired table is misleading about it.** Arm C's citation support of 0.826
against arm B's 0.762 is computed over different denominators — the arms answer
46 and 42 questions respectively — and paired over the 39 questions both
answered, the difference is +0.026 at p = 1.00. On token-F1 it is +0.003 at
p = 0.89.

| Comparison | token-F1 | citation support |
|---|---|---|
| B vs A | +0.141, p = 0.0003 * | +0.593, p = 0.0001 * |
| C vs B | +0.003, p = 0.89 | +0.026, p = 1.00 |
| D vs C | +0.193, p = 0.0002 * | +0.143, p = 0.068 |

So the honest statement is narrower than the retrieval numbers invite. Retrieval
versus no retrieval is a large, significant effect on both metrics. *Which*
retriever, at this sample size and with this generator, is not distinguishable:
the 0.131 Recall@5 gain of §VI-E does not produce a detectable difference in
answer quality or grounding over dense retrieval alone.

We do not read this as evidence that the fusion gain is illusory — §VI-E
measures it directly and significantly, and the oracle row shows retrieval
headroom of 0.193 that is significant at p = 0.0002. We read it as the expected
consequence of a generator that loses 0.597 on its own (§VI-G.1): a retrieval
improvement has to survive that noise floor to show up downstream, and on 72
questions it does not. A larger sample, or a generator that wastes less of the
evidence it is given, would be the way to detect it if it is there.

### G.1 Where the errors actually are

The oracle arm decomposes the remaining failure, and the result qualifies this
paper's own contribution.

| Quantity | Value |
|---|---|
| oracle (D) token-F1 | 0.403 |
| full system (C) token-F1 | 0.210 |
| **retrieval error** (D − C) | **0.193** |
| **generation error** (1 − D) | **0.597** |

Generation error is three times retrieval error. Given the correct passage and
nothing to find, the 3B quantized generator still reaches only 0.403 token-F1.
The ceiling on what *any* retrieval improvement can buy on this corpus, with
this generator, is 0.193 — and script-aware fusion has already taken part of it.

We state this plainly because it bounds the practical reading of §VI-E. The
fusion correction is a real and large improvement to *retrieval*, measured as
retrieval; it is not a claim that retrieval is the binding constraint on
end-to-end answer quality here. On this corpus it is not. A reader whose
priority is answer quality rather than evidence selection should read the
0.597 first, and it points at the generator — its size, its quantization, or
its prompt — rather than at the retriever.

### G.2 Lexical and entailment support disagree

The two Citation Support variants rank the arms differently.

| Arm | CSR lexical | CSR entailment |
|---|---|---|
| A closed-book | 0.162 | 0.297 |
| B RAG-dense | 0.762 | 0.476 |
| C RAG-hybrid | 0.826 | 0.391 |
| D oracle | 0.948 | 0.569 |

H3 survives under both: every retrieval arm beats closed-book either way. But
the RAG arms lose roughly half their support when the criterion changes from
"the answer's tokens appear in order in the cited passage" to "the passage
entails the answer", and the entailment column puts arm C *below* arm B.

We checked the obvious artefact and it is not the cause. `NLIScorer` truncates
at 384 tokens, and the retrieved context for arms B and C runs to ~780 words,
which would truncate most of it away. But the premise actually scored is the
*cited* passage, not the whole context, and citations resolve for 40 of 42
answers in arm B and 46 of 46 in arm C. Median premise length is 139–168 words
and only 3 of 183 scored pairs exceed the limit, so the entailment figures rest
on short, comparable premises.

The gap is therefore a real property of the answers: they reuse the cited
passage's wording without saying what the passage says, which is the failure
mode ROUGE-L precision is structurally blind to. Whether the entailment model is
itself well calibrated on Hindi and code-mixed answers is a separate question we
have not established, so we report both columns rather than choosing one.

Exact Match is low throughout (0.264 even for the oracle) and we do not read
much into it. A 3B model asked an open question rarely reproduces a gold string
verbatim, so EM here measures formatting agreement more than correctness;
token-F1 is the informative column.

### H. Answerability (Module 5)

A retrieval-score threshold is the standard abstention signal, and on this
corpus it does not work at any operating point. The reason is not that it is
poorly calibrated. It is that the quantity it thresholds does not separate the
classes.

**Top retrieval score by gold label, all 400 items** `[PROBE]`

| Retriever | answerable | unanswerable | best F1 | precision at best |
|---|---|---|---|---|
| BM25 | 13.55 | **13.82** | 0.366 | 0.235 |
| TF-IDF | 0.1424 | **0.1485** | 0.370 | 0.231 |
| Script-aware RRF | 0.0327 | 0.0325 | 0.351 | 0.216 |

The medians are indistinguishable, and in two of the three retrievers the
*unanswerable* questions score marginally higher — the opposite of the direction
a threshold assumes. Across the full sweep of τ, plotted as Figure 5, precision
never leaves the neighbourhood of 0.21–0.24 against an unanswerable base rate of
0.20, which is the signature of a signal carrying no information: abstaining at random achieves
precision equal to the prevalence.

The best F1 of 0.366 is reached at 86% abstention. Those two numbers have to be
read together. A recall of 0.93 on the unanswerable class, bought by refusing
most of the answerable questions as well, is not detection; it is silence.

The top score is the quantity a threshold uses, but it is not the only
retrieval-side feature available, and the calibrated combination of §IV-F is fit
over all of them. None of them separates either.

**Feature separation under BM25, all 400 items.** AUC is P(a random answerable
question scores above a random unanswerable one); 0.500 is chance.

| Feature | answerable | unanswerable | AUC |
|---|---|---|---|
| top1 − top2 margin | 2.1769 | 1.2750 | 0.594 |
| score spread (top1 − topK) | 5.2127 | 3.7732 | 0.599 |
| max score | 13.5509 | 13.8179 | 0.545 |
| mean top-k | 10.2447 | 10.3221 | 0.510 |
| scheme agreement | 0.4000 | 0.4000 | 0.504 |

The margin deserves comment because it was the feature we expected to work. The
reasoning was that a near-miss question retrieves several similarly-scoring
passages — the topic is present, the specific fact is not — and so shows a
smaller top1-to-top2 gap than a question with one clear answer. The prediction
holds in direction: 1.28 against 2.18. It fails in magnitude. An AUC of 0.594
orders a random pair correctly 59% of the time against a chance rate of 50%,
which is not a basis for abstention.

We fit the calibrated combination of §IV-F over these features to see how much
of that is recoverable, on the same split, and report it beside the threshold.

| Signal | test F1 | abstains on | AUC |
|---|---|---|---|
| 1. retrieval threshold | 0.345 | 87.0% | — |
| 4. calibrated combination | **0.360** | **37.5%** | 0.663 |

Combining does extract more than any feature alone: AUC 0.663 against a best
single feature of 0.599. F1 barely moves, which is what a near-chance feature
set predicts, but the *operating point* improves substantially — a comparable F1
while refusing 37% of questions rather than 87%. The distinction matters for
deployment, where the cost of silence is borne by users who asked answerable
questions.

The fitted coefficients are the sharper result:

| Feature | weight |
|---|---|
| score spread | 0.739 |
| mean top-k | −0.350 |
| margin | 0.138 |
| scheme agreement | 0.104 |
| **max score** | **0.011** |

`max_score` receives the smallest weight of the five, and it is the only
quantity the standard threshold looks at. The conventional abstention signal
thresholds the least informative feature available to it. What little
information exists lies in the *geometry* of the score distribution — how far
the top result stands above the rest of the retrieved set — rather than in its
magnitude.

**This follows from the dataset rather than from the retriever.** Of the 80
unanswerable items, 55 — the near-miss, false-premise and under-specified
classes — are deliberately written about schemes that *are* in the corpus. A
false-premise question names a real scheme and asserts something untrue of it; a
near-miss question matches a scheme's subject matter without matching any
particular statement. Such questions retrieve exactly as well as answerable
ones, because the passages they retrieve are genuinely about the topic. Only the
25 out-of-scope items concern absent subjects, and those are the only ones a
retrieval score can see.

A corpus-absence detector is therefore what a retrieval threshold is, and
answerability is not corpus absence. The distinction matters for deployment: a
system fielding questions about schemes it documents will meet the hard classes
far more often than the easy one.

We draw two conclusions. The first is negative and firm: **retrieval-score
thresholds should not be used as answerability signals on corpora where
unanswerable questions concern present topics**, and reporting a single fitted
operating point conceals this, because the fitted point moves from 0.000 recall
to 0.93 depending only on the split while the underlying curve stays flat. The
second is the motivation for §IV-F: deciding whether a *specific claim* is
supported requires reading the passage, which is what the generator and the
entailment check do and what a retrieval score cannot.

### H.1 All four signals, on the same items

We evaluate all four on a 120-item sample enriched to hold every unanswerable
item plus 40 answerable ones, fitting on a seeded 36 and reporting on the
remaining 84. Enrichment is what makes the per-class table legible — a
proportional sample of affordable size leaves two or three false-premise items
in the reported half — and it moves the unanswerable base rate to 0.667, so the
precision figures below are not comparable to the 0.20-base-rate runs above.

| Signal | reads | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|
| 1. retrieval threshold | scores | 0.702 | 0.707 | 0.946 | 0.809 |
| 4. calibrated combination | scores | 0.667 | 0.667 | **1.000** | 0.800 |
| 2. generator self-report | passage | **0.821** | **0.825** | 0.929 | **0.874** |
| 3. NLI entailment | passage | 0.821 | 0.825 | 0.929 | 0.874 |

**The aggregate table is misleading and the confusion matrices are not.** Signal
4 reaches recall 1.000 by refusing every question in the reported set: 56 of 56
unanswerable caught, and 0 of 28 answerable answered. Its precision of 0.667 is
the base rate exactly, which is what a system abstaining unconditionally scores.
Signal 1 is the same failure in milder form, answering 6 of 28. Only the
generator's self-report produces a system that answers anything: 17 of 28
answerable, while still catching 52 of 56 unanswerable.

Per-class recall has to be read against that. Signal 4 scores 1.000 on every
unanswerable class and the number means nothing, because it abstains on
everything and every class is "caught" by construction. The generator's figures
are earned: false-premise 1.000, under-specified 1.000, near-miss 0.952,
out-of-scope 0.824.

### H.2 Entailment adds nothing once the generator has abstained

Signals 2 and 3 are bit-identical, and that is a result rather than a bug. Both
fits chose their degenerate threshold — `min_confidence` 0.00 and τ 0.000 — so
each reduces to the same decision: did the generator decline to answer. Neither
the model's self-reported confidence nor the entailment probability improves on
the binary abstention flag.

The reason is visible in the counts. Of the 120 sampled items the generator
answered only 25, having abstained on the rest, and of those 25 just **two** are
truly unanswerable. There is almost nothing left for a second filter to catch,
so no threshold on entailment can improve F1 and the fit correctly declines to
set one.

Entailment is not uninformative — on those 25 answers it separates the classes
with an AUC of 0.848, the two false positives scoring a median entailment of
0.045 against 0.428 for the correctly answered. But an AUC computed against two
negatives is an observation, not a result, and we report it as one. What can be
said firmly is that on this corpus the generator's own abstention already does
the work the entailment check was added to do, and that a verification step has
value in proportion to how much the generator lets through.

