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
encoders trained with a contrastive or retrieval objective — LaBSE (Feng, Yang,
Cer, Arivazhagan and Wang, ACL 2022, 878–891), multilingual E5 (Wang et al.,
arXiv:2402.05672, 2024), and multilingual Sentence-BERT (Reimers and Gurevych,
EMNLP-IJCNLP 2019, with the multilingual extension by knowledge distillation,
EMNLP 2020) — project queries and passages into a shared space in which
translation-equivalent texts are near neighbours.

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
AI4Bharat resources (Kakwani et al., IndicNLPSuite, Findings of EMNLP 2020) serve
a similar role.

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
normalisation and no tuning. Neural retrieval has since produced alternatives to
the sparse half of the pair — ColBERT's late interaction over contextualised
token embeddings (Khattab and Zaharia, SIGIR 2020, 39–48) and SPLADE's learned
sparse representations with term expansion (Formal, Piwowarski and Clinchant,
SIGIR 2021) — but neither removes the constraint that concerns us. SPLADE in
particular expands terms within its model vocabulary, which mitigates vocabulary
mismatch but not *script* mismatch: expansion cannot bridge a Devanagari passage
and a Latin query that share no surface form.

RRF rewards documents ranked highly by several retrievers and, by construction,
penalises a document returned by only one. This is the intended semantics and is
well founded **when every retriever is capable of returning every document** — a
condition that holds on the monolingual collections where fusion was developed
and validated. Our §VI-E result concerns what happens when it does not.

Tu and Padmanabhan (MIA 2022 shared task submission, arXiv:2207.01940) come
closest to the problem, restricting the sparse index to the question's language
in a dense-sparse hybrid for cross-lingual QA. That addresses the retrieval step;
the fusion step still treats a dense-only passage as weakly evidenced.

### D. Answerability and grounded abstention

Systems that answer from retrieved evidence must be able to decline. SQuAD 2.0
(Rajpurkar, Jia and Liang, ACL 2018, 784–789) established unanswerable questions
as a first-class evaluation target, and did so adversarially: its 50,000+
unanswerable questions were written by crowdworkers *to resemble answerable
ones*. Confidence-based abstention for retrieval-augmented systems is an active
area, including risk control for RAG (Findings of EMNLP 2024) and
uncertainty-aware evidence verification `[VERIFY — pick one or two current
citations rather than gesturing at a literature]`.

**Our claim here must be stated carefully, because SQuAD 2.0 partly anticipates
it.** Adversarially authored unanswerable questions are already near-misses in
spirit. What the existing resources do not do is *stratify* them, so results are
reported over an undifferentiated unanswerable class. Our §III-E taxonomy
separates four kinds, and §VI-H shows why the distinction is not cosmetic: 55 of
our 80 unanswerable items concern schemes that *are* in the corpus. A BM25
threshold rejects every out-of-scope question on our test split and only 0.522 of
near-misses, whose passages are genuinely about the right scheme; a single F1 over
an undifferentiated unanswerable class hides that. The contribution is the stratified
measurement, not the observation that hard unanswerable questions exist.

### E. Code-mixed retrieval

Romanized Hindi-English is not a marginal input mode. **The published
platform-level statistics conflict, and the paper should say so rather than pick
the most favourable.** Figures in the deromanization literature put 93.17% of
native Hindi speakers' social-media posts in Roman script against 2.93% in
Devanagari, and a 2015 YouTube analysis reports 52% against 1%; yet another
measurement reports Devanagari on Twitter *rising* from 35% in 2014 to 82% in
2022. These are not reconcilable as stated and likely reflect different
platforms, sampling frames and dates.

The claim we can make safely is the code-mixing trend rather than the script
split: a peer-reviewed study of Indian Twitter users reports the proportion
preferring Hinglish rising from 44.9% in 2014 to 56.3% after 2020 (*Humanities
and Social Sciences Communications*, 2024, DOI 10.1057/s41599-024-03058-6)
`[VERIFY — title and authors confirmed via search; the article is behind an
authentication redirect, so confirm the figures against the full text before
quoting them]`.

Work on code-mixed text has concentrated on language identification,
transliteration and deromanization as preprocessing, typically to normalise input
into a single native script before a downstream monolingual model.
That framing treats script mismatch as a preprocessing problem to be eliminated.
We instead retain it as an experimental variable, because the query script is
exactly what determines whether a lexical retriever can see a passage at all —
and therefore whether fusion helps or harms.

---

## Bibliography

Verified against primary sources on 2026-09-21 unless marked otherwise.

| # | Work | Venue |
|---|---|---|
| 1 | Cormack, Clarke, Büttcher. *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* | SIGIR 2009, 758–759 |
| 2 | Khanuja, Bansal, Mehtani, Khosla, Dey, Gopalan, Margam, Aggarwal, Nagipogu, Dave, Gupta, Gali, Subramanian, Talukdar. *MuRIL: Multilingual Representations for Indian Languages.* | arXiv:2103.10730, 2021 |
| 3 | Li, Zhou, He, Wang, Yang, Li. *On the Sentence Embeddings from Pre-trained Language Models.* | EMNLP 2020, arXiv:2011.05864 |
| 4 | Tu, Padmanabhan. *MIA 2022 Shared Task Submission: Leveraging Entity Representations, Dense-Sparse Hybrids, and Fusion-in-Decoder for Cross-Lingual QA.* | arXiv:2207.01940, 2022 |
| 5 | Asai et al. *MIA 2022 Shared Task: Evaluating Cross-lingual Open-Retrieval QA for 16 Diverse Languages.* | arXiv:2207.00758 |
| 6 | Zhang, Ma, Lin et al. *Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval.* | arXiv:2108.08787 |
| 7 | Ogundepo et al. *AfriQA: Cross-lingual Open-Retrieval QA for African Languages.* | arXiv:2305.06897 |
| 8 | *Exploring Anisotropy and Outliers in Multilingual Language Models for Cross-Lingual Semantic Sentence Similarity.* | arXiv:2306.00458 |
| 9 | Feng, Yang, Cer, Arivazhagan, Wang. *Language-agnostic BERT Sentence Embedding.* | ACL 2022, 878–891 |
| 10 | Wang et al. *Multilingual E5 Text Embeddings: A Technical Report.* | arXiv:2402.05672, 2024 |
| 11 | Reimers, Gurevych. *Sentence-BERT.* | EMNLP-IJCNLP 2019 |
| 12 | Reimers, Gurevych. *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.* | EMNLP 2020, ACL Anthology 2020.emnlp-main.365 |
| 13 | Kakwani et al. *IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Models for Indian Languages.* | Findings of EMNLP 2020, 2020.findings-emnlp.445 |
| 14 | Rajpurkar, Jia, Liang. *Know What You Don't Know: Unanswerable Questions for SQuAD.* | ACL 2018, 784–789 |
| 15 | Khattab, Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* | SIGIR 2020, 39–48 |
| 16 | Formal, Piwowarski, Clinchant. *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.* | SIGIR 2021 (short), arXiv:2107.05720 |

### Still open

| Item | What is needed |
|---|---|
| Author list for arXiv:2306.00458 | fill from the paper |
| Hinglish trend statistic | *Humanities and Social Sciences Communications* 2024, DOI 10.1057/s41599-024-03058-6 — title and venue confirmed, but the article sits behind an authentication redirect. Confirm the 44.9%→56.3% figures against the full text before quoting, or make the claim qualitatively. |
| Roman-vs-Devanagari script split | **Do not cite.** The published figures conflict irreconcilably: 93.17%/2.93% (deromanization literature), 52%/1% (2015 YouTube), and Devanagari on Twitter *rising* 35%→82% (2014–2022). Different platforms and sampling frames. Use the code-mixing trend instead. |
| RAG abstention citation | Pick one or two concrete works rather than gesturing at a literature. Candidates: *Controlling Risk of Retrieval-augmented Generation* (Findings of EMNLP 2024); RAGTruth (Niu et al., 2024). |
