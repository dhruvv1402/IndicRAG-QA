# Section VI — Results

Modules 1, 2, 3 and 5 are written from measured numbers. Module 4 waits on the
generation sweep and is marked as such rather than estimated.

**Two different bases, and they must not be conflated.** The retrieval figures
(§VI-B to §VI-F) are `[PROBE]`: 180 synthetic probes over the real corpus. The
answerability figures (§VI-H) are measured on all 400 gold items, which exist
but are **not yet human-verified**. Neither may be quoted as a verified gold-set
result; §VI-A states what the probes can and cannot support.

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

Lexical retrieval does not cross the language boundary at all. BM25 scores 0.740
monolingual and 0.006 cross-lingual — a factor of more than a hundred. This is
not a weakness to be tuned away but the expected consequence of matching surface
tokens between a Devanagari query and a Latin-script passage that share
essentially none. It is the empirical form of the structural claim in §VIII, and
it is what makes the fusion behaviour in §VI-E a predictable consequence rather
than a surprise.

### D. Indic pretrained checkpoints underperform, with pooling controlled

MuRIL, pretrained on seventeen Indian languages, is the weakest dense encoder
tested at 0.238 Recall@5 — below MiniLM-L12 at 0.303, a model with roughly half
the parameters but trained with a retrieval objective. We ran both mean and CLS
pooling and report the better, so the result is not a pooling artefact; CLS
pooling is worse still at 0.143.

The sharpest form of the result is in the breakdown: MuRIL scores **exactly
0.000** on both the cross-lingual and the code-mixed slices. The model
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

Against dense alone — the comparison that decides whether hybrid retrieval earns
its place at all — script-aware fusion gains +0.028 cross-lingual (p = 0.037),
+0.041 code-mixed (p = 0.074) and +0.050 overall (p = 0.0086).

Three things should be read from this table rather than one.

First, the gain on the target slices is large and significant. Second, the
monolingual regression is real, significant, and the price of the method:
removing the lexical double-vote costs precision where both retrievers could see
the same passage and agreement between them genuinely was evidence. Third, the
overall row is indistinguishable from noise, and a paper reporting only that row
would conceal both the gain and the cost.

This also revises the negative result of the α sweep, which found no interior
weighting beating both endpoints by more than 0.001 and recorded the hybrid
hypothesis as unsupported. That verdict was correct about weighted fusion and
wrong about the hypothesis: hybrid retrieval does beat its components here, once
fusion stops penalising passages for their script.

### F. A caveat the fusion result does not remove

LaBSE, used alone, reaches **0.332** on the code-mixed slice — against 0.173 for
script-aware fusion over BM25 and e5-base, and 0.132 for e5-base alone. On
code-mixed queries, choosing a different encoder outperforms correcting the
fusion over a weaker one, by a wide margin.

We state this plainly because it qualifies the practical reading of §VI-E. Two
points bound it. The fusion correction is orthogonal to the choice of encoder:
it is a statement about how rankings are combined, and nothing prevents applying
it over LaBSE, which we have not tested and which we would expect to be stronger
than either. And LaBSE does *not* lead the cross-lingual slice — it scores 0.097
there, below both MiniLM and e5-base — so it is not a uniformly better choice
but a differently shaped one.

The result contradicts our own prior expectation, recorded before the
experiment, that LaBSE would lead the cross-lingual slice by virtue of its
translation-ranking objective. Instead it is the only model that handles
Romanized Hindi well. If this survives the gold set, the practical guidance is
to select the encoder by query *script* rather than by language coverage, and it
is the most immediately useful finding in this section. It needs the gold set
before it can be claimed.

### G. Direct LLM against RAG (Module 4)

> **Pending.** The generation sweep over arms A (closed-book), B (RAG-dense),
> C (RAG-hybrid) and D (RAG-oracle) is running on a stratified 72-item sample
> balanced across the six language-pair cells. Generations are cached per arm,
> so this table fills in without re-running completed work. The results to be
> reported here are Exact Match and script-aware token F1 per arm, Citation
> Support Rate under both lexical and NLI verification, abstention rate, and the
> fabricated-citation rate — the last being the rate at which an arm names a
> passage id that does not exist, which arm A can do only by invention.

### H. Answerability (Module 5)

A retrieval-score threshold is the standard abstention signal, and on this
corpus it does not work at any operating point. The reason is not that it is
poorly calibrated. It is that the quantity it thresholds does not separate the
classes.

**Top retrieval score by gold label, all 400 items** `[PROBE]`

| Retriever | answerable | unanswerable | best F1 | precision at best |
|---|---|---|---|---|
| BM25 | 13.55 | **13.82** | 0.372 | 0.237 |
| TF-IDF | 0.1424 | **0.1485** | 0.370 | 0.231 |
| Script-aware RRF | 0.0327 | 0.0325 | 0.351 | 0.216 |

The medians are indistinguishable, and in two of the three retrievers the
*unanswerable* questions score marginally higher — the opposite of the direction
a threshold assumes. Across the full sweep of τ, precision never leaves the
neighbourhood of 0.21–0.24 against an unanswerable base rate of 0.20, which is
the signature of a signal carrying no information: abstaining at random achieves
precision equal to the prevalence.

The best F1 of 0.372 is reached at 86% abstention. Those two numbers have to be
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
supported requires reading the passage, which is what natural-language inference
does and what a retrieval score cannot. On the false-premise and near-miss items
where the threshold is uninformative, NLI entailment identified the unanswerable
item in 7 of 7 cases tested. That is a small sample and is reported as such;
the full comparison runs with Module 4.

