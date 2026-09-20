# Section II — Related Work

**Literature pass done 2026-09-21.** Citations below were checked against
primary sources this session unless marked `[VERIFY]`, which flags a work I am
confident exists and is standard but whose exact venue, year or author list was
not re-checked here and must be confirmed before submission. Do not submit a
`[VERIFY]` entry without checking it.

---

## The framing gate — result

**The framing in `draft.md` §I must be narrowed. "We show" becomes "we
characterise and correct".**

The single-retriever penalty in Reciprocal Rank Fusion is **not an oversight; it
is the documented, intended behaviour**. A document appearing in only one
retriever's list contributes a reciprocal rank from that list and zero from the
other, and this is described in the practitioner literature as exactly what a
hybrid ranker should do — mutual confirmation across retrievers is treated as
evidence of relevance.

That is correct *whenever both retrievers could have retrieved the document*.
Our contribution is to identify the precondition and show where it fails: when
the query and a candidate passage are in different scripts, a lexical retriever
has **zero recall by construction**, so its silence is not a withheld vote but an
impossible one. The penalty then measures the retriever's blind spot rather than
the passage's relevance.

**The closest prior art recognises the asymmetry but corrects it on the retrieval
side rather than the fusion side.** Tu and Padmanabhan's MIA 2022 shared-task
submission searches "the top K passages globally in the dense indices in all
languages, and the top K passages in the sparse index **in the same language as
the question**" — an explicit acknowledgement that sparse retrieval is
same-language-only, handled by restricting what the sparse retriever is asked
for. We are not aware of work that corrects the *fusion arithmetic* so that a
cross-script passage is scored over the retrievers eligible to return it.

**Honest statement of novelty**, to be used in §I:

> The limitation of lexical retrieval across scripts is well known, and prior
> cross-lingual QA systems restrict the sparse index to the query's language in
> response. We show that the standard fusion arithmetic nonetheless encodes the
> assumption that a retriever's silence about a passage is evidence against it,
> quantify the cost of that assumption on Indic and code-mixed queries, and give
> a correction that recovers it.

If a reviewer finds prior art for the fusion-side correction specifically, the
claim degrades gracefully to a quantification on Indic data, which the
per-slice results in §VI-B and §VI-C still support on their own.

---

## Draft text

Our work sits at the intersection of four lines of research.

### A. Multilingual dense retrieval

Dense retrieval over multilingual encoders is the standard remedy for the
vocabulary and language mismatch that defeats lexical search. Sentence-level
encoders trained with a contrastive or retrieval objective — LaBSE `[VERIFY:
Feng et al., ACL 2022]`, multilingual E5 `[VERIFY: Wang et al., 2024]`, and
multilingual Sentence-BERT `[VERIFY: Reimers and Gurevych, EMNLP 2019 and the
multilingual extension, EMNLP 2020]` — project queries and passages into a shared
space in which translation-equivalent texts are near neighbours.

Benchmarks for this setting are predominantly *monolingual-per-language* or
cross-lingual at the language rather than the script level. Mr. TyDi (Zhang et
al., arXiv:2108.08787) established a multilingual dense-retrieval benchmark across
eleven typologically diverse languages; the MIA 2022 shared task (Asai et al.,
arXiv:2207.00758) extended open-retrieval QA to sixteen languages; and AfriQA
(Ogundepo et al., arXiv:2305.06897) covers cross-lingual open-retrieval QA for
African languages. These establish that multilingual dense retrieval works and
that its quality varies sharply by language. What they do not isolate is the
interaction we study: what happens to a *fused* system when the query script and
the evidence script differ.

### B. Indic language representations

Encoders pretrained specifically on Indian languages are the natural first choice
for this corpus. MuRIL (Khanuja et al., arXiv:2103.10730, 2021) is trained on
Indian-language corpora and, notably for our purposes, augments pretraining with
**translated and transliterated document pairs** to supply explicit cross-lingual
signal; it outperforms mBERT across the XTREME benchmark. IndicBERT and the wider
AI4Bharat resources `[VERIFY: Kakwani et al., Findings of EMNLP 2020]` serve a
similar role.

The gap we address is that these models are evaluated on classification and
understanding benchmarks, and their readiness for *retrieval* off the shelf is
generally assumed rather than measured. The distinction is not incidental. Both
MuRIL and IndicBERT are masked-language-model checkpoints with no pooling layer
trained for sentence representation, and MLM representations are known to be
anisotropic — occupying a narrow cone in which the average cosine similarity of
unrelated points is high — which is precisely the geometry that defeats
similarity search (Li et al., EMNLP 2020, arXiv:2011.05864). This anisotropy has
been shown to persist in multilingual models for cross-lingual semantic
similarity specifically (arXiv:2306.00458). Our §VI-B result quantifies the
consequence for Indic retrieval, and is sharpened by MuRIL's transliteration-pair
pretraining: the model given explicit transliteration signal is the one that
retrieves nothing on our code-mixed slice.

### C. Hybrid and fused retrieval

Combining a lexical and a dense retriever is standard practice, on the premise
that their errors are complementary: lexical matching is precise on rare terms,
names and numbers, while dense retrieval generalises over paraphrase and
synonymy. Reciprocal Rank Fusion (Cormack, Clarke and Büttcher, SIGIR 2009,
pp. 758–759) remains the default combiner, chosen for needing no score
normalisation and no tuning; weighted score fusion and learned combiners such as
ColBERT and SPLADE `[VERIFY]` occupy the other end of the complexity range.

RRF rewards documents ranked highly by several retrievers and, by construction,
penalises a document returned by only one. This is the intended semantics and is
well founded **when every retriever is capable of returning every document** — a
condition that holds on the monolingual collections where fusion was developed
and validated. Our §VI-C result concerns what happens when it does not.

Tu and Padmanabhan (MIA 2022 shared task submission, arXiv:2207.01940) come
closest to the problem, restricting the sparse index to the question's language
in a dense-sparse hybrid for cross-lingual QA. That addresses the retrieval step;
the fusion step still treats a dense-only passage as weakly evidenced.

### D. Answerability and grounded abstention

Systems that answer from retrieved evidence must be able to decline. SQuAD 2.0
`[VERIFY: Rajpurkar et al., ACL 2018]` established unanswerable questions as a
first-class evaluation target, and selective prediction and RAG-hallucination
work has since developed confidence-based abstention `[VERIFY — needs a current
survey citation]`.

Unanswerable items in these resources are typically treated as a single
undifferentiated class. Our Module 5 result turns on the distinction we draw in
§III-E: an out-of-scope question is rejected by any threshold, whereas a
**false-premise** question — one presupposing a benefit the scheme does not
provide — retrieves its scheme's passages at high similarity because the topic is
right and only the asserted fact is absent. Reporting a single answerability F1
over a set dominated by the former conceals total failure on the latter.

### E. Code-mixed retrieval

Romanized Hindi-English is not a marginal input mode. Reported usage statistics
put the overwhelming majority of Hindi-speakers' social-media text in Roman
rather than Devanagari script, with code-mixed usage rising over the last decade
`[VERIFY — statistic seen in the deromanization literature; locate and cite the
primary source before use, or drop the specific figures and make the qualitative
claim]`.

Work on code-mixed text has concentrated on language identification,
transliteration and deromanization as preprocessing, typically to normalise input
into a single native script before a downstream monolingual model `[VERIFY]`.
That framing treats script mismatch as a preprocessing problem to be eliminated.
We instead retain it as an experimental variable, because the query script is
exactly what determines whether a lexical retriever can see a passage at all —
and therefore whether fusion helps or harms.

---

## Bibliography checklist

Before submission, verify every `[VERIFY]` entry and fill in full details.

| # | Work | Status |
|---|---|---|
| 1 | Cormack, Clarke, Büttcher. *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR 2009, 758–759. | **verified** |
| 2 | Khanuja et al. *MuRIL: Multilingual Representations for Indian Languages.* arXiv:2103.10730, 2021. | **verified** |
| 3 | Li et al. *On the Sentence Embeddings from Pre-trained Language Models.* EMNLP 2020, arXiv:2011.05864. | **verified** |
| 4 | Tu, Padmanabhan. *MIA 2022 Shared Task Submission: ... Dense-Sparse Hybrids ...* arXiv:2207.01940. | **verified** |
| 5 | Asai et al. *MIA 2022 Shared Task* overview. arXiv:2207.00758. | **verified** |
| 6 | Zhang et al. *Mr. TyDi.* arXiv:2108.08787. | **verified** |
| 7 | Ogundepo et al. *AfriQA.* arXiv:2305.06897. | **verified** |
| 8 | *Exploring Anisotropy and Outliers in Multilingual Language Models...* arXiv:2306.00458. | **verified** (authors to fill) |
| 9 | LaBSE (Feng et al.) | `[VERIFY]` |
| 10 | multilingual-E5 (Wang et al.) | `[VERIFY]` |
| 11 | Sentence-BERT (Reimers, Gurevych) + multilingual extension | `[VERIFY]` |
| 12 | IndicBERT / IndicNLPSuite (Kakwani et al.) | `[VERIFY]` |
| 13 | SQuAD 2.0 (Rajpurkar et al.) | `[VERIFY]` |
| 14 | ColBERT, SPLADE | `[VERIFY]` |
| 15 | Code-mixed script-usage statistics — primary source | `[VERIFY]` |
| 16 | Selective prediction / RAG hallucination survey | `[VERIFY]` |
