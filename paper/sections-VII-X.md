# Sections VII, VIII, IX, X — full draft

Written as paper prose rather than notes, continuing `sections-III-IV-V.md`.

Every retrieval figure quoted here is `[PROBE]`: measured on 180 synthetic
probes, not on the human-verified gold set. The probe set is built from real
article titles and real passages, so the *structure* of the finding is sound, but
no number below may be reported as a gold-set result until the 400-item set has
been through verification. Where a number would change the argument if it moved,
that is said explicitly rather than left to the reader.

---

## VII. Error Analysis (Module 6)

### A. Protocol

We analyse twenty failures drawn against a fixed taxonomy of ten categories —
code-mixing, transliteration variation, named entities, amounts, dates,
ambiguity, near-duplicate schemes across schemes, the cross-lingual gap,
long-tail schemes, and short queries — with two cases per category.

Selection is by rule rather than by hand. Within each category we take the
failures on which the system was most confident and most wrong, ranked by
retrieval score against a failed gold match. This matters more than it may
appear. Choosing illustrative cases after seeing which ones look interesting is
how an error analysis becomes a collection of anecdotes: the examples then
demonstrate whatever the author already believed. Fixing the taxonomy first, the
count per category second, and the selection rule third means the cases are
answerable to the taxonomy rather than to the argument.

Each case is recorded as a structured record — query, gold passages, the ranked
output of each system under comparison, the generated answer, a diagnosis, and
the component judged to have failed — and is committed to the repository as
`evals/errors.jsonl` alongside the rendered report. The diagnosis field names one
of retrieval, generation, answerability or data, so that the distribution of
failures across components is itself a result rather than an impression.

### B. What the cases showed

The dominant failure mode is not subtle. In the majority of the twenty cases the
gold passage was never retrieved at all, and the top-ranked hit was the correct
scheme in the *wrong language*: a Hinglish query about the National Health
Mission's untied grants returns the English article when the gold passage is the
Hindi one, and an English query about Sukanya Samriddhi eligibility does the
same in reverse. The system was not confused about the topic. It was confused
about nothing at all — it had simply never been able to see the passage that
would have answered the question.

This concentrates the error in one component. Categories that appear
linguistically distinct in the taxonomy — code-mixing, transliteration
variation, the cross-lingual gap — turn out to share a single mechanism once the
retrieved lists are read side by side, and that mechanism is a retriever that
cannot cross a script boundary. In 16 of the 20 cases the gold passage is not
returned at any rank, and all 20 are diagnosed as retrieval failures.

**That unanimity is a property of the selection, not a finding about the
generator.** The cases are drawn from retrieval failures against a taxonomy of
retrieval phenomena, so a generation error has no route into this sample. It
would be a mistake to read "all 20 are retrieval" as evidence that the generator
is sound, and §VI-G.1 shows the opposite: given the gold passage and nothing to
find, the generator still loses 0.597 token-F1, three times what retrieval
loses. The error analysis says where *retrieval* fails and why; the oracle arm
is what says how the two components compare.

### C. The case that produced the paper's main result

One case (E02 in the committed report) is worth stating in full, because the
central contribution of this paper came out of it rather than out of a metric.

The aggregate tables showed hybrid fusion performing *worse* than dense
retrieval alone on exactly the queries hybrid retrieval was supposed to help
with, and gave no indication why. Both components were behaving as documented;
the fusion was implemented correctly against its own specification; the α sweep
found no interior setting that beat either endpoint by more than 0.001, which
reads as a clean negative result. Read as a table, the finding was that hybrid
retrieval does not help here.

Reading the twenty cases gave the actual explanation. Of the twenty failures,
the dense retriever alone had returned the gold passage in eight. Plain
Reciprocal Rank Fusion lost all eight. A fusion method that discards every case
its stronger component got right is not a method with a disappointing
coefficient; it is a method whose arithmetic is wrong for this setting, and no
amount of sweeping α would have revealed that, because α was never the problem.

The mechanism is developed in §VIII. The methodological point belongs here: the
aggregate metric was not merely uninformative about the cause, it pointed away
from it. Twenty cases, selected by rule and read in full, were what turned a
negative result into a positive one.

---

## VIII. Discussion

### A. Why fusion fails across a script boundary

Rank fusion rests on an assumption that is usually harmless and is false here.
Combining two retrievers is worthwhile when their errors are complementary —
when each recovers documents the other misses — and rank-based fusion in
particular treats agreement between retrievers as evidence. A document returned
by both systems accumulates two contributions; one returned by a single system
accumulates one. Under Reciprocal Rank Fusion the fused score is
`Σᵢ 1/(k + rankᵢ)`, summed over the systems that returned the document, and the
document found by one system alone is therefore scored at roughly half the
document found by both.

Across a script boundary this arithmetic silently changes meaning. A lexical
retriever matching surface tokens has, by construction, *zero* recall over
passages written in a script the query does not use: a Devanagari query and a
Latin-script passage share essentially no tokens, so BM25 cannot return that
passage at any rank, for any query, however relevant it is. The lexical
retriever's silence about a cross-script passage is not evidence against that
passage. It is not evidence at all.

Plain RRF cannot tell the difference between these two situations, because the
sum is taken over the systems that *did* return the document rather than over
the systems that *could*. A cross-script passage is docked a full contribution
for the lexical retriever's blindness, and it is docked it relative to
same-script passages whose lexical votes were possible. The penalty is therefore
applied precisely where it is least deserved: the cross-lingual case is the one
in which the dense retriever is the only informative signal, and it is the case
in which fusion most aggressively discounts it.

It is worth being precise about what is and is not new. This behaviour is not a
bug in RRF, which was designed for monolingual settings in which every retriever
can reach every document, and under which discounting single-system evidence is
sensible. Nor is the underlying difficulty unknown: work on multilingual
retrieval has previously restricted sparse retrieval to same-language candidates
when combining it with dense retrieval, which avoids the problem by not creating
it. Our contribution is not the observation that sparse retrievers struggle
across languages. It is the characterisation of *how the fusion arithmetic
converts that limitation into an active penalty*, and a minimal correction
stated in those terms.

### B. Script-aware fusion

The correction follows directly from the diagnosis. We score each passage over
the retrievers that were *eligible* to return it, rather than over those that
did, treating a lexical retriever as ineligible for a passage whose script
differs from the query's. A cross-script candidate is then judged on the dense
contribution alone — the only evidence that ever existed about it — instead of
being penalised for a vote that was never possible.

The change is small, requires no training and no tuning, and adds one lookup per
candidate. On the probe set it takes cross-lingual Recall@5 from 0.022 to 0.153
(paired randomization test, p = 0.0001) and code-mixed Recall@5 from 0.028 to
0.173 (p = 0.0002), against a dense-only baseline of 0.125 and 0.132
respectively. The comparison against dense alone is the one that decides whether
hybrid retrieval earns its place at all, and script-aware fusion beats it
overall by 0.050 (p = 0.0086).

This also revises a negative result reported earlier in the same work. The α
sweep had found no interior weighting that beat both endpoints, and we had
recorded the hypothesis that hybrid retrieval outperforms its components as
unsupported. That verdict was correct about the method it tested and wrong about
the hypothesis: hybrid retrieval does beat its own components on this corpus,
but only once fusion stops penalising passages for being written in the language
the query was not.

### C. The cost, stated plainly

Script-aware fusion is a trade and not a free win. Monolingual Recall@5 falls
from 0.712 to 0.668, a regression of 0.044 at p = 0.032. Removing the lexical
double-vote costs precision exactly where it was legitimate — where both
retrievers could see the same passage, agreement between them really was
evidence, and we are now discarding some of it.

On this corpus the trade is clearly favourable, because monolingual retrieval
was already strong and cross-lingual retrieval was near zero, and a method that
moves a slice from 0.022 to 0.153 at the cost of 0.044 elsewhere is buying a
great deal for very little.

**The cost may not exist at all.** The 0.044 is a probe-set figure. Measured on
the 320 gold questions (§VI-A.1) the monolingual regression is −0.008 at
p = 1.00, while the cross-lingual gain rises to +0.389. Neither set is verified,
so we report the probe figure as the conservative one and note that the only
measurement taken on real questions found no price worth naming. On a deployment whose queries are overwhelmingly
monolingual the same trade would be a poor one. We report the regression rather
than the aggregate because the aggregate hides it: overall Recall@5 moves from
0.483 to 0.500, a difference indistinguishable from noise (p = 0.32), and a
paper reporting only that number would be concealing both the gain and the cost.

### D. Retrieval is not the binding constraint here

The oracle arm of §VI-G puts a number on what this correction can be worth
end to end, and it is smaller than the retrieval figures alone suggest. Given
the gold passage and nothing to find, our generator reaches 0.403 token-F1;
the full system reaches 0.210. Generation error is therefore 0.597 and
retrieval error 0.193, a factor of three apart.

We report this because it constrains the practical claim rather than the
scientific one. Script-aware fusion is a large improvement to retrieval,
measured as retrieval. It does **not** produce a measurable improvement in
answer quality over dense retrieval alone at this sample size: paired over the
questions both arms answered, arm C exceeds arm B by 0.003 token-F1 (p = 0.89)
and 0.026 citation support (p = 1.00).

The two facts fit together rather than conflicting. A generator losing 0.597 on
its own is a noise floor that a retrieval gain has to clear to become visible
downstream, and on 72 questions a gain of this size does not clear it. A reader
who cares about end-to-end answer quality should know that fixing retrieval
entirely would buy 0.193, while the generator is leaving 0.597 on the table.

That ratio is a property of this configuration, not a general law. A 3B model at
4-bit quantization is a deliberately small generator, chosen so every number
here is reproducible on commodity hardware, and a larger one would move the
0.597 without touching the 0.193. The point is that the two error sources have
to be measured separately before either is optimised, which is precisely what
the oracle arm is for and why the brief asks for it.

### E. Where this generalises

The mechanism is not about Hindi, and it is not about script. It appears
wherever rank fusion is applied over retrievers with *asymmetric coverage* of
the collection — where at least one retriever cannot, in principle, return part
of what is being searched. Cross-script and cross-language pairs are the case
studied here. The same structure arises when fusing a text retriever with one
operating over images or tables, where each is blind to the other's modality;
when fusing over access-controlled shards, where one retriever is permitted a
subset of documents; and when fusing a domain-specialised index with a general
one over a collection only partly covered by the specialist.

The practical guidance is correspondingly simple, and is the form in which we
would want this result used: before fusing retrievers, check their *coverage*,
not only their quality. Two retrievers of equal measured accuracy can combine
well or destructively depending on whether each could have retrieved what the
other did. Where coverage is asymmetric, fusion should be taken over the
retrievers eligible for a candidate rather than those that returned it.

---

## IX. Limitations

We state these at length, because the value of a result of this kind depends on
knowing precisely how far it reaches.

**The corpus is encyclopedic, not regulatory.** Our documents describe Indian
government schemes; they are not the scheme guidelines themselves. Sustained
attempts to build the corpus from official portals failed on soft 404s and
image-only scans — one eighteen-page document yielded seventeen characters of
extractable text — and the deviation is documented in the product requirements
rather than quietly absorbed. Consequently we make no claim about retrieval over
statutory or regulatory Hindi, which differs from encyclopedic Hindi in register,
sentence length and terminology. The cross-script retrieval mechanism we
identify is a property of the retrievers and the writing systems rather than of
the genre, so we expect it to transfer; we have not shown that it does.

**One Indic language.** We evaluate Hindi and Hindi–English code-mixing. India
has twenty-two scheduled languages, and several of the most widely spoken use
scripts other than Devanagari. Script-aware fusion is formulated over an
eligibility relation between queries and passages and is not specific to
Devanagari, but a two-script corpus cannot distinguish a method that handles
script asymmetry in general from one that happens to handle this pair. Behaviour
with three or more scripts, and with partially overlapping scripts such as those
sharing a Perso-Arabic base, is untested.

**Scale, and a single primary annotator.** The evaluation set is 400 items, of
which 320 are answerable, spread over six query-language-by-passage-language
cells. Cells therefore hold 45 to 70 items each, and per-cell differences of a
few points are not resolvable. All items are authored by a single primary
annotator working from model-generated candidates; a stratified 15% second pass
re-labels the answerability decision blind, and Cohen's κ is reported, with
κ ≥ 0.70 treated as a gate on reporting any answerability result. Inter-annotator
agreement on a 15% sample is a weaker control than independent double annotation
throughout, which was not feasible here.

**Preliminary numbers.** Every retrieval figure in this paper is measured on
synthetic probes rather than on the verified gold set, and is marked `[PROBE]`
throughout. The probes are constructed from real passages and real article
titles, and the mechanism they expose is structural rather than statistical, so
we do not expect the direction of the fusion result to change. The magnitudes
may. Any figure still marked `[PROBE]` at submission is labelled as such in the
text, and no such figure is reported as a gold-set result.

**A small quantized generator on CPU.** Generation uses Qwen2.5-3B-Instruct at
4-bit quantization on a four-core CPU. This is a deliberate constraint — the
system is intended to be reproducible on commodity hardware, and every figure we
report is one a reader can regenerate without a GPU — but it bounds what the
question-answering results say. A larger or unquantized model would plausibly
change the absolute answer quality figures. It would not change the retrieval
results, which are model-independent, and the retrieval results are where our
contribution lies.

**The fusion correction is demonstrated over three encoders, not all of them.**
§VI-A.2 applies it over multilingual-e5-base, MiniLM-L12 and LaBSE on the gold
set, and it improves each by a similar margin (+0.081, +0.103, +0.072), which is
what a claim about the arithmetic rather than the representation predicts. That
is evidence for the generalisation in §VIII-E but not proof of it: three
sentence encoders on one corpus is a narrow base, and we have not tested the
mechanism on the non-language asymmetries §VIII-E claims it reaches.

**Most of our own pre-registered targets were missed.** We fixed eight success
criteria before collecting any number — retrieval recall by language group,
answerability F1, citation support, annotator agreement — and met one of them:
that the closed-book arm would sit far below the retrieval arms on citation
support. We report this rather than quietly dropping the targets, and two of the
misses are informative. The retrieval targets (0.85 monolingual, 0.70
cross-lingual) were set against an imagined corpus of official scheme
guidelines, where a question and its answer share vocabulary; they were never
recalibrated when that corpus proved unobtainable and an encyclopedic one was
substituted, which is an error in our planning rather than in the system. The
answerability target of 0.75 was unreachable by the signal it assumed, for the
reason §VI-H gives. A target missed because the question changed under it is
worth separating from a target missed because the method failed, and these are
mostly the former.

**No component is fine-tuned.** All encoders and the generator are used
off-the-shelf. Fine-tuning any of them on in-domain data would likely improve
absolute numbers, and would confound the comparison we are making, which is
between fusion methods holding the retrievers fixed.

**The answerability threshold is a retrieval-side signal, and our negative
result about it is bounded by our own taxonomy.** We show it carries no
information here, but the reason is that 55 of our 80 unanswerable items are
written about schemes present in the corpus. A benchmark whose unanswerable
questions were predominantly out-of-scope would find the same threshold useful,
and the 0.688 recall we measure on that one class shows why. The claim is
therefore conditional: retrieval-score thresholds fail on unanswerable questions
about *present* topics, which we argue is the deployment-relevant case for a
system fielding questions about schemes it documents, but we have not shown they
fail in general. We evaluate a natural-language-inference signal alongside,
which addresses exactly the classes the threshold cannot see.

---

## X. Conclusion

We have presented IndicRAG-QA, an evidence-grounded question answering system
for Hindi, English and Hindi–English code-mixed queries over a parallel corpus
of Indian government welfare schemes, together with a 400-item evaluation set
annotated across six query-language-by-passage-language cells and a taxonomy of
four unanswerable question types.

The central finding is about rank fusion rather than about any individual
retriever. Reciprocal Rank Fusion, applied unmodified to a lexical and a dense
retriever over a bilingual collection, penalises cross-script passages for the
lexical retriever's structural inability to return them, and in doing so
discards the only informative evidence available in exactly the cross-lingual
and code-mixed cases such a system exists to serve. Scoring each candidate over
the retrievers *eligible* to return it, rather than those that did, is sufficient
to correct this: cross-lingual Recall@5 improves roughly sevenfold and code-mixed
roughly sixfold on our probe set, at a measured cost of 0.044 monolingual
Recall@5, with no training and one additional lookup per candidate.

Two things about how the result was obtained seem worth carrying forward. The
first is that it was invisible in the aggregate metrics, which showed only that
hybrid retrieval underperformed, and was recovered from a structured reading of
twenty failure cases; the α sweep that preceded it produced a clean negative
result about the wrong question. The second is that the correction is a change
of arithmetic rather than of model — it required no additional parameters, no
training data, and no additional compute at query time — which suggests that the
gap between the cross-lingual and monolingual performance of hybrid retrieval
systems may be, in part, an artefact of how their components are combined rather
than a limitation of the components themselves.

One measurement bounds all of this and belongs in the summary rather than only
in the results. The oracle arm shows generation error of 0.597 against retrieval
error of 0.193 on our configuration: correcting retrieval entirely would buy
less than a third of what the generator is currently losing. The fusion result
is a claim about retrieval, and it is not a claim that retrieval is what most
limits answer quality on a corpus like this one. Separating the two is what the
oracle arm exists to do, and doing it changed what we would advise a
practitioner to fix first.

Immediate future work is to complete human verification of the evaluation set
and replace every `[PROBE]` figure with a gold-set measurement; to extend the
corpus beyond Hindi to at least one non-Devanagari Indic script, which would
distinguish a general treatment of script asymmetry from a Hindi-specific one;
and to test the eligibility formulation on asymmetric-coverage settings outside
language entirely, of which multimodal and access-controlled retrieval are the
most natural candidates.
