# Abstract and Section I — full draft

Completes the prose draft. With `section-II-related-work.md`,
`sections-III-IV-V.md` and `sections-VII-X.md`, every section but §VI (Results)
now exists as paper text; §VI waits on Module 4 and Module 5 numbers.

Retrieval figures are `[PROBE]` — measured on 180 synthetic probes, not the
verified gold set.

---

## Abstract

*(178 words. Figures marked `[PROBE]`; to be re-measured on the gold set.)*

> Retrieval-augmented generation over multilingual corpora is commonly built by
> fusing a lexical retriever with a dense one, on the assumption that their
> errors are complementary. We show that this assumption fails in a specific and
> correctable way when a query and its evidence are written in different
> scripts. A lexical retriever cannot return a passage whose script differs from
> the query's, because the two share almost no tokens; under rank fusion such a
> passage collects one vote where a same-script passage collects two, and is
> penalised for the lexical retriever's *blindness* rather than for its own
> irrelevance. On a bilingual Hindi–English corpus of Indian government welfare
> schemes, Reciprocal Rank Fusion reaches 0.022 Recall@5 on cross-lingual
> queries against 0.125 for its own dense component. We propose **script-aware
> fusion**, which scores each candidate over the retrievers *eligible* to return
> it, recovering 0.153 cross-lingual and 0.173 code-mixed Recall@5
> (p = 0.0001, p = 0.0002) at a cost of 0.044 monolingual Recall@5. We release a
> 400-item bilingual QA set with an explicit query-language × evidence-language
> matrix and a four-class unanswerable taxonomy.

---

## I. Introduction

### A. The access gap

Public-interest information in India is routinely published in one language and
asked about in another. A citizen checking whether their household qualifies for
a post-matric scholarship is reading a circular issued in English, or its Hindi
counterpart published separately and not as a translation, and is asking the
question in Hindi — or, far more commonly in practice, in Romanized
Hindi–English code-mix: *"Scholarship ke liye minimum eligibility kya hai?"*

Keyword search fails on this query three times over. The language differs from
the document's. The script differs from the document's. And the surface words
differ from both, because Romanized Hindi has no standardised orthography, so
the same word reaches the index as *eligibility*, *yogyata* and *योग्यता*
depending on who typed it. Each of these is individually well studied. Their
conjunction is the ordinary case for a large population of users, and it is the
setting this paper measures.

The stakes are not those of open-domain question answering. Documents that
govern who receives public money are consulted by people deciding whether to
apply, and a confident wrong answer about an eligibility ceiling or a deadline
is materially worse than no answer at all. A system in this setting must be able
to say that it does not know, and must be evaluated on whether it says so in the
right places rather than only on the quality of the answers it does give.

### B. The standard remedy, and where it breaks

The accepted response to the language and script mismatch is multilingual dense
retrieval, which embeds queries and passages into a shared space and so is not
bound by surface overlap. The accepted response to dense retrieval's weakness on
rare terms — scheme names, amounts, section numbers — is hybrid retrieval:
combine the dense retriever with a lexical one and fuse their rankings, most
commonly with Reciprocal Rank Fusion, which requires no score normalisation and
is strong enough to be a standard baseline.

We find that hybrid retrieval assembled in this standard way does not merely
fail to help the cross-lingual case. It *actively harms* it, scoring below the
dense component it contains. On our corpus, plain RRF reaches 0.022 Recall@5 on
cross-lingual queries while its own dense retriever, used alone, reaches 0.125.

The cause is structural rather than a matter of tuning, which is why an α sweep
over weighted fusion found nothing and reported a clean negative result. Rank
fusion treats agreement between retrievers as evidence and therefore discounts a
candidate returned by only one of them. Across a script boundary the lexical
retriever has zero recall *by construction* — it cannot return a Devanagari
passage for a Latin-script query at any rank, for any query, however relevant —
so its silence carries no information. Fusion nonetheless charges the candidate
for that silence, and does so precisely in the cases where the dense retriever
is the only informative signal available.

The correction that follows from this diagnosis is small: score each candidate
over the retrievers that were *eligible* to return it rather than over those
that did. It requires no training, no tuning and one lookup per candidate, and
it recovers cross-lingual Recall@5 from 0.022 to 0.153 and code-mixed from 0.028
to 0.173, at a measured and reported cost of 0.044 on monolingual queries.

### C. How the result was found

We note the path deliberately, because it bears on how such systems should be
evaluated. The aggregate retrieval tables showed hybrid fusion underperforming
and gave no indication why; every component behaved as specified, and the
hyperparameter sweep returned a negative result about a question that was not
the relevant one. The mechanism became visible only on a structured reading of
twenty failure cases, in which the dense retriever alone had found the gold
passage in eight and plain RRF had lost all eight. A fusion method that discards
every case its stronger component got right is not a method with a disappointing
coefficient. Aggregate metrics did not merely fail to show this; they pointed
away from it.

### D. Contributions

1. **Script-aware fusion.** A correction to rank fusion for retrievers with
   asymmetric coverage of a collection, with its structural cause, a paired
   significance test, and its cost reported alongside its benefit. We state the
   scope precisely: RRF's discount on single-retriever evidence is intended
   behaviour rather than an oversight, and prior multilingual retrieval work
   already avoids the problem by restricting sparse retrieval to the query's
   language. Our contribution is to show that the fusion *arithmetic* still
   treats a retriever's silence as evidence, to quantify what that costs, and to
   correct it in a form that generalises beyond language to any
   asymmetric-coverage setting.

2. **A bilingual evidence-grounded QA set.** 400 items over Indian government
   welfare schemes, with an explicit query-language × evidence-language matrix
   covering six cells, a four-class unanswerable taxonomy, per-item gold passage
   identifiers, and a documented annotation protocol with blind second-pass
   agreement.

3. **A decomposition of answerability.** We show that retrieval-score
   thresholds, the standard abstention signal, reach 0.000 recall on
   false-premise questions, and give the mechanism: a false-premise question
   retrieves confidently because the scheme it names is real, so no threshold
   over retrieval scores can separate it from an answerable one.

4. **A controlled negative result on Indic MLM checkpoints.** Off-the-shelf
   masked-language-model encoders for Indic languages underperform
   sentence-trained multilingual encoders at retrieval, with the mean-pooling
   confound controlled rather than left as an explanation.

### E. Organisation

Section II surveys cross-lingual retrieval, code-mixed IR and rank fusion.
Section III describes the corpus and the QA set. Section IV describes the
system, Section V the experimental protocol, and Section VI the results.
Section VII presents the error analysis from which the main result came,
Section VIII discusses the mechanism and where it generalises, and Sections IX
and X state limitations and conclusions.
