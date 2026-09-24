# Sections III, IV, V — full draft

Written out as paper prose rather than notes. These three sections are fully
determined by work already completed, so nothing here waits on the gold set.
Figures and tables are referenced by the numbering in `draft.md`.

Numbers in this file are measured, not estimated, unless explicitly marked.

---

## III. Dataset

### A. Corpus construction

We assemble a parallel Hindi–English corpus of Indian government welfare and
education schemes. The corpus is bilingual *by construction* rather than by
translation: each scheme is represented by two independently authored documents,
one in each language, so that a query in one language may legitimately be
answered by evidence in the other. This is what makes cross-lingual retrieval a
measurement here rather than a simulation.

Scheme pairs are resolved through the MediaWiki `langlinks` API. We seed a list
of English article titles covering education, skills, financial inclusion,
pensions, social welfare, housing, health, agriculture and governance, and retain
only those for which Wikipedia itself asserts a Hindi counterpart. Parallelism is
therefore an editorial claim by the source, not an assumption by us. Of 49 seeded
schemes, 40 yielded a usable pair; the remainder either had no Hindi article or
had one shorter than 500 characters, and stub articles were rejected rather than
padded. The final corpus is **40 schemes × 2 languages = 80 documents**, spanning
14 scheme categories.

**A deviation from our original design, and the reason for it.** We first
attempted to build the corpus from official ministry guideline documents, which
would have given regulatory rather than encyclopedic text. Two obstacles made
this infeasible. First, the Hindi paths on `socialjustice.gov.in` return HTTP 200
with a "Page not found" body — a soft 404 — so the bilingual premise fails at the
source. Second, and decisively, the scheme guideline PDFs are scans: the
18-page, 2.8 MB *Post-Matric Scholarship for SC Students* guidelines document
yields **17 characters** of extractable text, and no OCR engine was available in
our environment. A corpus of English-only text-layer documents would have
eliminated the cross-lingual question the work exists to study. We consider the
trade worthwhile and state its cost explicitly in §IX: our results describe
retrieval over *descriptive* text about government schemes, and support no claim
about retrieval over statutory or regulatory prose.

### B. Devanagari integrity validation

Text extraction is the highest-risk stage of an Indic corpus pipeline, and its
characteristic failure is silent. Legacy or subset-embedded fonts produce output
that renders as plausible Devanagari while being nonsense at the character level:
dependent vowel signs reordered ahead of their consonant, conjuncts broken at the
virama, or glyphs mapped into the Latin private-use area. None of this raises an
error. It produces text that a reader unfamiliar with the script will accept, and
once indexed it degrades retrieval in a way that resembles poor model quality.

We therefore validate every extracted Hindi document against five independent
checks, and report them because we believe the procedure is reusable:

1. **Misplaced dependent vowel signs.** In a well-formed orthographic syllable a
   matra (U+093A–U+094C) must follow a consonant or nukta. A matra in any other
   position is the signature of the reordering bug.
2. **Dangling word-final viramas.** A virama joins two consonants; word-final it
   indicates the second half of a conjunct was lost.
3. **Devanagari codepoint ratio**, which catches a font mapped outside the block.
4. **Hindi function-word rate.** Text can be entirely within the Devanagari block
   and still not be Hindi; a stopword-rate floor catches that mojibake.
5. **Unicode replacement characters**, indicating an encoding failure.

All 80 documents pass. One finding from developing these checks is worth
reporting for other Indic pipelines. Three clean documents initially failed the
dangling-virama check, and the cause was not corruption: formal Hindi prose
writes a **zero-width non-joiner after a virama** to force an explicit halant
rather than a conjunct ligature, as in उद्देश्‍य. ZWNJ lies outside the
Devanagari block, so a word-tokenizer keyed on that block splits the token and
leaves its first half apparently terminating in a bare virama. Stripping
zero-width characters before validation resolves it. We retain a tolerance rather
than requiring zero final viramas, because Hindi does write some genuinely, such
as the Sanskrit-derived adverb अर्थात्.

### C. Passage segmentation

Segmentation bounds the achievable retrieval quality: a fact split across a chunk
boundary cannot be retrieved intact regardless of encoder. We therefore split on
document structure first and length second. Source articles carry explicit
section markers, which we use both as split points and as retained metadata —
a passage's section path ("Eligibility > Income criteria") is genuine context for
a generator, and it distinguishes the many near-identical eligibility passages
that different schemes contribute.

Sections comprising navigation or bibliography (references, external links, see
also, and their Hindi equivalents) are discarded; retaining them would place link
lists in the retrieval pool, where they match many queries lexically and answer
none. Within a section, sentences are packed to a target of 170 tokens, bounded
at 240, with 25% overlap between adjacent chunks where a split was length-driven
rather than structural. Overlap is not applied across section boundaries, since
those are real semantic breaks and bleeding across one would place a clause from
one scheme inside another scheme's passage. **Two corrections belong here, because both were found by measuring rather
than by reading.** The committed corpus contains 29 passages above the 240
bound, the largest 415 tokens. The packer respects the bound; a guard that
folds an undersized chunk into the previous passage did not. It tested only
that the previous passage came from the same *document*, so a short chunk from
any later section merged into whatever passage happened to be last — which is
precisely the bleeding across section boundaries that overlap is forbidden from
doing, and it left the merged passage carrying the earlier section's path. One
passage is labelled `Introduction` and ends in text about the Unified Payment
Interface, three sections later. The guard now requires the same section and a
result within the bound.

The corpus has **not** been re-segmented, and that is the second correction.
Passage identifiers are positional — `scheme-lang#p0007` is the eighth passage
of that document — so they are stable only while segmentation is. Applying the
fix produces 803 passages instead of 694, and although every existing
identifier still resolves, **312 of them resolve to different text**, and 123 of
the 320 answerable gold items cite one. A re-segmentation would therefore
repoint more than a third of the gold set at passages that may no longer contain
the answer, without a single broken reference to signal it. Content-derived
identifiers would fail loudly instead, and are the right fix; changing the
scheme now would invalidate the same gold set for the same reason. The corpus is
frozen until verification completes, and the measurements in this paper are
taken on the corpus as committed.

**How much this costs the results.** 86 passages of 694 draw sentences from more
than one source section; 5 of those pair a section with its own subsection and
are harmless, leaving 81 that join unrelated material, and 46 of the 320
answerable gold items cite one. A merged passage is larger than it should be,
so it presents more surface for a query to match and might be expected to
inflate recall. It does, slightly and not significantly: those 46 items reach
Recall@5 0.674 [0.543, 0.804] under script-aware fusion against 0.591
[0.526, 0.650] for the other 274. The intervals overlap heavily, and at 14% of
the set an effect of that size moves the overall figure by about a point.

We report the defect rather than the reassurance, because the reassurance is
narrow. The retrieval numbers survive it. What does not survive is the claim
that a passage's `section_path` is reliable context for a generator: for those
81 passages it names one of the sections present and not the others, which is
the input §IV-A feeds the model as provenance.

**Three further metadata defects, and why they went unseen.** Auditing the
remaining per-passage invariants found that `section_path` is not the only
provenance field that does not mean what it says. 29 passages exceed the stated
240-token bound, one reaching 321. `text_raw`, documented as the source spelling
kept alongside the normalized `text` so that a cited amount is displayed as the
circular wrote it, is itself normalized: 691 of 694 passages have a `text_raw`
that normalization leaves unchanged, and no Hindi passage retains the Devanagari
digits its source uses. `char_span` is worse still -- the segmenter advances its
cursor by the normalized chunk length while searching raw source text, so the
offsets drift monotonically within a document, and only 22.8% name a slice that
recovers their own passage while 56 have drifted past the end and name nothing.

All three share the root cause the section-spanning defect has: segmentation
splits sentences after normalizing a section and never keeps a mapping back to
the source. They share something more instructive too. Every one of these fields
is written, serialized, and documented, and *read by nothing* -- so none of them
could fail loudly, and a downstream citation view built on `char_span` would
have pointed a reader at the wrong text roughly four times in five. This is the
same failure mode as §VI-D's uninstantiated NLI scorer and the unread
`cited_found` flag: not incorrect code that broke, but correct-looking code that
no execution path ever reached. We therefore added `corpus audit`, which reads
every one of these invariants and reports a per-check rate.

**Repairing two of them without unfreezing the corpus.** The obvious fix --
re-running segmentation with a source-offset mapping -- is unavailable, because
it would also apply the merge-guard correction and so change how many passages
each document yields, shifting every identifier after the first merge. But the
freeze protects `passage_id` and `text`, and neither is what these two fields
need. `corpus repair-spans` therefore re-derives them in place: each passage's
normalized text is aligned back to its source document and `char_span` is set to
the located extent, with `text_raw` read from it. The alignment is sentence-wise
rather than a single forward walk, because a passage is not always contiguous in
its source -- overlap re-emits the tail of the previous chunk, so the same
sentences can appear twice with the second copy sitting behind where the first
ended. A single-run aligner scored 0.03 coverage on 39 passages that are in fact
entirely present in their documents.

The repair refuses to write if any `passage_id`, `text` or `token_count` moved;
none did, so no index, cache, gold reference or published number in this paper
is affected. Span fidelity rises from 25.4% to 98.8% of the passages that carry
a span, 99.1% carry one, and `text_raw` now preserves source spelling for every
passage -- 21 Hindi passages recover the Devanagari digits their sources use,
which is exactly the fidelity the field was introduced to provide. The six
passages the aligner cannot place have their span *cleared* rather than left
drifted, on the principle that an absent span is honest where a wrong one
misleads; `corpus audit` counts the two states separately so a repair cannot
hide its misses among its refusals.

That separation earned its keep immediately. The first version of the aligner
declined 23 passages, and inspecting them showed every sentence of 20 of them
aligning perfectly when searched from the start of the document: the search
retried from the top only when a forward pass returned *nothing*, so a
half-matching anchor found ahead of the cursor was committed to instead of the
true match behind it. Retrying whenever the forward pass does poorly, rather
than only when it fails outright, took the declines from 23 to 6. What remains
unfixed is the third defect: the 29 oversized passages, and the welded ones
whose span must swallow a heading to cover them, both require re-chunking and so
wait on the same re-anchoring as the merge fix.

The gold set, the embedding caches and the error analysis all key off these
identifiers, which is what makes their apparent stability worth stating
precisely rather than assuming.

This yields **694 passages** (491 English, 203 Hindi) across 40 schemes. Passage
length has median 170 tokens (mean 157, interquartile range 111–193, range
20–415), and 82.7% carry a section path beyond the article lead.

**The language imbalance is inherent and should be read as a property of the
source.** Hindi Wikipedia articles run roughly 40% the length of their English
counterparts, giving a 0.41 Hindi-to-English passage ratio. Scheme coverage is
also uneven, from 3 to 62 passages per scheme. Both facts constrain the dataset
design in §III-D: Hindi-evidence questions are drawn from a smaller pool, and
passage selection must be stratified across schemes to prevent a single long
article dominating a matrix cell.

### D. Question-answer set

*(Verified 2026-09-23 in two language-model passes; test split reviewed by a person 2026-09-24. 393 items survive, 7 were rejected.)*

The set was built as 400 instances, 320 answerable and 80 unanswerable; after
verification it holds 393, 313 answerable and 80 unanswerable. The
answerable items follow an explicit **query-language × evidence-language matrix**
(Table I), which is what permits the monolingual / cross-lingual / code-mixed
comparison in §VI-B to be made on balanced strata rather than on whatever
distribution happened to arise.

**TABLE I — Answerable items by query and evidence language (verified; design target in parentheses)**

| Query ↓ / Evidence → | English | Hindi | Total |
|---|---|---|---|
| English | 69 (70) | 45 (45) | 114 |
| Hindi (Devanagari) | 44 (45) | 59 (60) | 103 |
| Hinglish (Romanized) | 51 (55) | 45 (45) | 96 |
| **Total** | 164 | 149 | **313** |

Three groupings follow directly: **monolingual** (128), where query and evidence
share a language and script; **cross-lingual** (89), different languages each in
native script; and **code-mixed** (96), Romanized Hindi-English queries whose
evidence is never in the query's script. The Hinglish row is never monolingual by
construction, since Romanized Hindi is not the script of any corpus document —
which is precisely why code-mixed queries are expected to be hardest.

Candidates are bootstrapped with a locally hosted quantized model
(Qwen2.5-3B-Instruct, Q4_K_M). The protocol specified that a person then verify
each one. **Verification instead began with a language model (Claude Opus
5.5), in two independent passes, and a person then reviewed the test split.**
The first pass read every item against
its cited passage and proposed a correction or a rejection; it judged only 27 of
the 320 bootstrapped answerable items correct as generated, found 63 gold
answers the passage contradicts or does not state, and found 112 questions that
were unreadable machine translation, almost all in the cross-lingual and
Hinglish cells. The duplicated unanswerable stems (§III-E) and 7 rejected
answerable items were replaced by newly written ones. A second pass, run by
fresh model instances with no access to the first, then checked every item in
its final form against its evidence and the passages a retriever returns for it,
confirming 353 and making small corrections to 42. It rejected 5 — three
because another passage in the corpus gives a conflicting figure, so the
question has no single answer — and 2 more were dropped as duplicate questions. It also added, for 73 items, passages that state the same fact,
most of them the other language's article (see below).

A person — the author — then reviewed every one of the 273 test-split items
against its evidence, with the model passes' notes shown beside it, and
accepted all 273 as they stood: no question or answer was edited and none was
rejected. Two qualifications apply. The review was not blind, since the
reviewer saw the model's verdict, so its 100% acceptance is a confirmation of
the model passes rather than an independent labelling of the items; and it was
one person. The 120 dev-split items, which are used only for fitting
thresholds, were verified by the model passes alone. Each item records who
checked it (`annotator`), and every report states the mix in its banner. The
blind second pass (§IX) was run by a model instance, and its κ = 1.000 over 59
items measures one model's consistency, not agreement between annotators.

Two construction details materially affect the measurements. First,
**cross-lingual items translate the question only** and leave the gold passage
where it is. Translating the passage as well would convert a cross-lingual item
into a monolingual one and inflate every cross-lingual figure. Second, **gold
evidence is a set, not a single passage**: where a fact appears in both language
versions of a scheme, either passage counts as correct. Scoring against one
designated passage would penalise a cross-lingual retriever for exactly the
behaviour under study.

**Semantic validation of generated candidates.** Bootstrapping with a small model
introduces a failure mode that parse-rate monitoring cannot see. In development,
a 0.5B-parameter model asked to translate `इस योजना के लिए आय सीमा क्या है?` into
English returned *"What is the salary for this job?"* — well-formed JSON with a
non-empty question field, and entirely invented. Such an output scores as a
success under any validity check while corrupting the dataset. We therefore apply
structural checks to every generated item: a translation into Hindi must contain
Devanagari and must differ from its input; a romanization must be Latin script
*and* retain Hindi function words, or the model translated rather than
transliterated; and a generated question must match its passage's language, not
echo prompt scaffolding, and share at least one token with its own evidence.
These checks cannot distinguish a good translation from a mediocre one. They
catch the model not performing the task at all, which is the failure a small
model actually produces.

### E. Unanswerable items and their taxonomy

The 80 unanswerable items are stratified into four classes, and the
stratification is the point. A question about a topic absent from the corpus is
rejected by any threshold and demonstrates nothing; reporting a single
answerability F1 over a set dominated by such questions would conceal the system's
behaviour on the cases that matter.

**TABLE II — Unanswerable taxonomy (verified; design target in parentheses)**

| Class | n | Definition |
|---|---|---|
| Out-of-scope | 25 (25) | Topic absent from the corpus entirely |
| **Near-miss** | 29 (30) | Topic and scheme present; the specific fact is not |
| **False premise** | 16 (15) | Presupposes a benefit or clause that does not exist |
| Under-specified | 10 (10) | Unanswerable without naming a scheme |

The two emphasised classes are the hallucination test. For both, retrieval
returns a confidently scored, topically correct passage, and only a properly
calibrated answerability signal abstains. These items are hand-authored rather
than model-generated: a language model asked for a question its passage cannot
answer reliably produces the trivial out-of-scope kind. Near-miss and
false-premise items are additionally paired with the **distractor passage** they
are designed to nearly match, so that an annotator can confirm the fact is truly
absent rather than merely unfound.

### F. Leakage control

Model-bootstrapped questions tend to reuse the distinctive wording of their
source passage, which advantages lexical retrieval and would make any
dense-versus-lexical comparison an artefact of annotation rather than a property
of retrieval. The generation prompt instructs paraphrase rather than extraction,
and we measure what that achieved.

**Content-token Jaccard overlap, question against gold passage** (verified set, n = 313):

| Cell | n | median | max | leakage possible? |
|---|---|---|---|---|
| en→en | 69 | 0.088 | 0.239 | yes |
| hi→hi | 59 | 0.094 | 0.245 | yes |
| hinglish→en | 51 | 0.036 | 0.129 | yes |
| en→hi | 45 | 0.000 | 0.107 | no — cross-script |
| hi→en | 44 | 0.015 | 0.088 | no — cross-script |
| hinglish→hi | 45 | 0.000 | 0.038 | no — cross-script |

The cross-script cells sit near zero because the two scripts share almost no
tokens. That is the script boundary rather than evidence of annotation hygiene,
so the figure that bears on leakage is the same-script one: **median 0.077, max
0.245** over 179 items, against 0.066 and 0.340 before verification. The model
rewrites of §III-D did not make questions copy their passages; the one cell
whose overlap rose, hinglish→en (median 0.009 to 0.036), rose because rewritten
questions name their scheme in English, which is how such questions are
actually typed.

**Testing it requires holding the cell fixed.** Our pre-registered plan was to
repeat the retrieval comparison on the low-overlap tertile of the whole set, but
the low tertile is then mostly one language pair, so the split measures the pair
rather than the overlap. Within each same-script cell, splitting at the median
overlap instead:

| Cell | BM25 − dense, low half | high half | difference |
|---|---|---|---|
| en→en | +0.015 | +0.000 | −0.015 |
| hi→hi | −0.017 | +0.017 | +0.034 |
| hinglish→en | +0.060 | +0.096 | +0.036 |

No cell shows a lexical advantage that grows with overlap by more than a few
points, and none of the within-half differences is significant (all p ≥ 0.25).
Leakage is not detectable in the verified set. Full output is in
`evals/report-leakage.txt`.

---

## IV. System

### A. Pipeline

The system is a conventional retrieval-augmented pipeline with one non-standard
component. Documents are fetched and validated (§III-A, §III-B), normalized,
segmented (§III-C), and indexed both lexically and densely. At query time the
input is classified by language and script, retrieved against both indices,
fused, passed to a generator constrained to the retrieved evidence, and gated by
an answerability decision. Figure 1 gives the full diagram.

![Fig. 1. System pipeline. Offline, once: Devanagari validation, segmentation, and lexical and dense indexing. Per query: language and script identification, script-aware fusion, generation constrained to the evidence, and an answerability decision. One path serves all three query types.](figures/fig1-pipeline.png)

Crucially, **the pipeline does not branch on detected query language**. One path
serves English, Hindi and code-mixed queries alike. This is deliberate: if
retrieval varied by query type, the per-language comparison in §VI-B would
measure the routing logic rather than the retriever.

### B. Normalization

Normalization runs identically on passages and queries. This symmetry is the
property that matters — a corpus normalized one way at index time and queries
another at search time cease to match on exactly the cases of interest, and the
resulting failure resembles poor model quality rather than a preprocessing bug.

Beyond Unicode NFC composition and whitespace repair, three corpus-specific
equivalences are applied. Devanagari digits are mapped to ASCII, since a query
written with one and a passage with the other otherwise never match. Monetary
amounts are canonicalised, collapsing `Rs. 3,50,000`, `₹3.5 lakh`, `3,50,000/-`
and `३,५०,०००` to a single key; amounts are the most frequent answer type in this
corpus and their surface form differs systematically between the English and
Hindi versions of the same scheme. Sentence segmentation treats danda (।) and
double danda (॥) as terminators alongside the full stop, without which a Hindi
paragraph is returned as a single sentence — and normalization preserves both
terminators, which is a precondition for that split rather than an incidental
property, since segmentation normalizes a section before splitting it.

**Where each equivalence actually applies.** Two of the three act on the
retrieval path; amount canonicalisation does not. It is called by answer scoring
and error analysis, not by the index tokenizer, so at retrieval time `₹3.5 lakh`
and `3,50,000` share only the token `3`. We state this because the paragraph
above would otherwise imply a matching benefit that the system does not deliver,
and because 87 of the 313 answerable verified items carry an amount in the
question. Supplying the canonical key to the tokenizer as well moves BM25
Recall@5 on those 87 items from 0.640 [0.534, 0.736] to 0.674 [0.567, 0.766], a
paired difference of +0.034 at *p* = 0.25, and +0.011 over all 313 (p = 0.13). The reason the gap costs so
little is worth recording: the tokenizer splits on commas, so `3,50,000` already
fragments into `3`, `50` and `000`, and the Devanagari form yields the same
three subtokens once digits are mapped. Fragmentation supplies most of the
equivalence incidentally, and the forms it does not cover — `3.5 lakh` against
`3,50,000` — are the rarer ones.

The symmetry claim itself does hold, and not only by construction: the indexing
path normalizes a passage a second time, with soft-wrap joining disabled, while
a query is normalized once. Applying the second pass changes the token sequence
of 0 of 694 passages, so the two paths agree in practice as well as in intent.

Original surface text is retained alongside the normalized form throughout, so
that citations display what the document actually said — a property §III-C
records as having been broken in the committed corpus until it was repaired.

### C. Query language identification

Queries are classified into the three classes the task requires — English, Indic,
Code-Mixed — by a rule-based procedure. We prefer rules to a learned classifier
here because the decision is defined by script and by a small closed set of
function words, and because a misclassification changes the language the answer
is generated in, making an explainable decision worth more than a marginal
accuracy gain.

A query with a Devanagari character ratio above 0.60 is Indic. A query containing
both Devanagari and Latin characters is Code-Mixed, tested by presence rather
than ratio: a query such as *"PM-YASASVI के लिए eligibility criteria kya hai?"*
is only 8% Devanagari by character count and would slip below any reasonable
floor, yet is plainly code-mixed. A Latin-only query is Code-Mixed if it contains
Hindi function words, and English otherwise.

The function-word lexicon carries the classification and requires one refinement.
Several valid Romanizations of Hindi function words are also among the most
frequent English words — *the* (थे), *is* (इस), *to* (तो), *us* (उस), *me* (में).
Counting these as Hindi evidence classified plain English questions as code-mixed
in development. We therefore partition the lexicon into unambiguous markers,
which can trigger the code-mixed label, and ambiguous markers, which corroborate
but never trigger it.

On the 400 gold questions the procedure agrees with the annotated language on
every one: 143 English, 131 Indic, 126 Code-Mixed, no errors in either
direction. We are careful about what that shows. These questions were generated
per language-pair cell, so each is a clean instance of its class — a Hindi
question is fully Devanagari, a Hinglish one is consistently Romanized. Genuine
user input is messier: transliteration is inconsistent, English and Hindi
alternate within a clause, and typos cross the function-word lexicon. The result
establishes that the rules implement the definition without a gap, not that the
definition covers everything a user will type.

It matters mainly because everything downstream depends on it. The script the
classifier reports is what decides which passages the lexical retriever is
treated as eligible for (§IV-E), so a misclassification would disable the
correction silently rather than raise anything.

### D. Retrieval

**Lexical.** We implement TF-IDF with cosine scoring and BM25-Okapi
(k₁ = 1.2, b = 0.75) over a script-aware tokenizer that dispatches per token
rather than per document — necessary for code-mixed input, where English content
words sit beside Romanized or Devanagari function words within one query. English
tokens are casefolded and lightly stemmed; Hindi tokens receive light suffix
stripping for common case markers. We note that a full Hindi morphological
analyser is out of scope and that its absence costs some lexical recall on Hindi
queries.

**Dense.** Five multilingual encoders are registered (Table III), of which four
were evaluated; `ai4bharat/indic-bert` is a gated repository and could not be
obtained, and we report its exclusion rather than omitting it silently.

**TABLE III — Encoder registry**

| Model | Params | Type | Role |
|---|---|---|---|
| `intfloat/multilingual-e5-base` | 278M | sentence | primary |
| `paraphrase-multilingual-MiniLM-L12-v2` | 118M | sentence | speed baseline |
| `LaBSE` | 471M | sentence | cross-lingual specialist |
| `google/muril-base-cased` | 236M | **MLM** | Indic-pretrained |
| `ai4bharat/indic-bert` | 33M | MLM | *gated, not evaluated* |

MuRIL and IndicBERT are masked-language-model checkpoints, not sentence encoders:
they were pretrained with an MLM objective and have no pooling layer trained for
sentence representation. We include MuRIL because it is the model practitioners
reach for by name on Indic tasks, and because the comparison isolates whether
pretraining-language coverage substitutes for retrieval training. To prevent the
result being dismissed as a pooling artefact we encode it under both mean and CLS
pooling and report the better. Pooling is performed explicitly rather than
delegated, which would otherwise silently attach an untrained mean-pooling head
and present the result as a sentence encoder.

At 694 passages we perform **exact** search over an L2-normalized float32 matrix
rather than using an approximate index. An ANN index at this scale would
introduce recall error indistinguishable from model error, which this study
cannot afford, since separating the two is its purpose. Every embedding cache
stores a fingerprint of the passages it was built from and refuses to load when
they differ — a silently misaligned embedding matrix produces plausible, ordered
and meaningless results.

### E. Fusion

We evaluate weighted score fusion, `α·lex + (1−α)·dense` over min-max normalized
scores, and Reciprocal Rank Fusion at k = 60. Both min-max and z-score
normalization are implemented, and we report the ablation rather than asserting
the preference: the two are indistinguishable on the gold set, −0.003 Recall@5
for z-score at p = 1.00, and no point of the α sweep separates them by more than
0.013.

The reason usually given for preferring min-max — that BM25 scores are strongly
right-skewed, so z-score leaves the fused ranking dominated by lexical outliers
— holds in its premise and not in its consequence. BM25 candidate pools are
right-skewed on 97.7% of queries (mean skewness 2.38), and the top candidate
scores 2.86× the pool mean against 1.06× for the dense retriever. The fused
ranking does not follow: z-score raises the lexical share of the top-1 fused
score from 29.0% to 34.4%, and the share of top-5 slots held by passages the
dense retriever never returned from 3.2% to 6.5%. The predicted direction is
there; the magnitude never reaches the metric.

What separates the two normalizers is imputation, not skew. A passage returned
by only one component contributes 0 from the other, and 0 is the floor of a
min-max range but the mean of a z-scored one — an absent dense component ranks
above 63% of the candidates it is compared against under z-score, and above none
of them under min-max. We keep min-max because silence from a retriever should
not be scored as an average opinion, which is a claim about what the fusion
means rather than about what it measures. The ablation is
`evals/report-fusion-normalisation.txt`.

RRF is left untuned to keep it an honest baseline against a tuned α.

We additionally propose **script-aware fusion**, which is the subject of §VI-E
and is described there in full. In brief: a passage is scored over the retrievers
*eligible* to return it, where a lexical retriever is ineligible for a passage
whose script differs from the query's.

### F. Generation and answerability

Generation is performed by a locally hosted quantized model behind a provider
interface, with an extractive provider available as a non-neural baseline and as
a model-free fast path for testing.

Answerability is decided by one of four signals, evaluated independently so the
paper can say which carries the decision. Two are **retrieval-side**: a
threshold on the top score, and a calibrated logistic combination over the top
score, the top-1-to-top-2 margin, mean top-k similarity, score spread and scheme
agreement. Two read the **passage text**: the generator's own `answerable`
report with its confidence, and a natural-language-inference check asking
whether the cited passage entails the produced answer. All four are fitted on
the development split by maximising F1 on the unanswerable class rather than
accuracy, since the classes are deliberately imbalanced and accuracy is
maximised by never abstaining.

The margin feature deserves note, because it is the one we expected to carry the
retrieval-side signals. Maximum similarity indicates how good the best passage
looks; it does not indicate whether anything else looked equally good. A
near-miss unanswerable question should retrieve several passages at similar
scores, because it matches a scheme's subject matter without matching any
particular statement, whereas an answerable question usually has one clear
winner — so a high maximum score with a small margin should be the signature of
the hardest unanswerable class.

We state that as the design hypothesis it was, rather than as a property of the
data. §VI-H measures it: the direction holds, and the magnitude does not. The
split between the two kinds of signal is what the results turn on.

---

## V. Experimental Setup

### A. Hardware, and why it is reported

All experiments run on an Intel Core i5-11320H (4 physical cores, 8 threads),
15.7 GB RAM, **with no GPU**. We report this not as a caveat but as a property of
the work: every component was selected to run on a commodity laptop, and the full
pipeline — corpus construction, five encoder sweeps, retrieval evaluation and
generation — is reproducible without accelerator hardware. For work concerning
access to public-information systems in India, reproducibility on the hardware
students and civic technologists actually own is a relevant property.

The constraint shapes several choices: exact rather than approximate search,
4-bit quantized generation, and CPU-only PyTorch wheels.

### B. Measured throughput

**TABLE IV — Encoding time, 694 passages, 4 CPU threads**

| Encoder | Params | Time | Rate |
|---|---|---|---|
| MiniLM-L12 | 118M | 149.6 s | 4.6/s |
| multilingual-e5-base | 278M | ≈380 s | ≈1.8/s |
| LaBSE | 471M | ≈600 s | ≈1.2/s |
| MuRIL (mean pooling) | 236M | 632.3 s | 1.1/s |
| MuRIL (CLS pooling) | 236M | 622.2 s | 1.1/s |

Two observations. First, the full registry encodes in under 40 minutes on CPU,
which is the figure that makes the study reproducible. Second, **throughput does
not track parameter count**: MuRIL at 236M is slower than LaBSE at 471M, because
the MLM arm runs a hand-rolled encoding loop while the sentence-transformer
models use an optimised batching implementation. Studies budgeting CPU inference
from parameter counts alone will misestimate.

Retrieval latency is negligible by comparison: exact dense search over 694
passages is a single matrix-vector product, and BM25 scoring completes in a few
milliseconds.

### C. Splits and tuning protocol

The question set is partitioned into a **120-item development split** and a
**test split** — 280 items as designed, 273 after verification removed 7 —
stratified over query language, answerability and scheme, with seed 20260922. Stratification matters because an unstratified split
leaves some cell of Table I with a handful of test items, and a per-language
result computed on a handful of items reports noise with the appearance of
authority.

Every threshold, fusion weight, retrieval depth and prompt is selected on the
development split. The test split is scored once. We state this because the
alternative is not recoverable after the fact: once a test figure has been seen,
it cannot be unseen, and a protocol that permits iteration against it produces
numbers that are optimistic by an unmeasurable margin.

The split was sealed on 2026-09-23 over the 393 verified items: **dev 120,
test 273**, stratified by query language, answerability and scheme with
largest-remainder allocation. The answerability threshold is fitted on dev and
reported on test. The retrieval configuration — α = 0.4, RRF k = 60, 50
candidates per component, the primary encoder — was fixed before the gold set
existed and was not tuned on it, so retrieval is reported on test (Table in
§VI-A) and, where the analysis needs the power, on dev and test together; the
section says which. Figures from the earlier unverified set and from synthetic
probes are marked as such where they are kept for comparison.

### D. Metrics

Retrieval is scored with Recall@k, Precision@k and Hit Rate@k for
k ∈ {1, 3, 5, 10}, Mean Reciprocal Rank, and nDCG@10, with reciprocal rank taken
as zero where no gold passage is returned. Question answering is scored with
Exact Match and token-level F1 under a script-aware normalizer, plus semantic
similarity against the gold answer. Answerability is scored with accuracy,
precision, recall and F1 treating UNANSWERABLE as the positive class, with a
confusion matrix and per-class recall.

The normalizer requires comment, because a normalizer written for English is
wrong for Hindi in a way that does not announce itself. The widely used SQuAD
normalization strips English articles, casefolds, and removes ASCII punctuation.
Applied to Devanagari it strips articles Hindi does not have, casefolds to no
effect, and leaves the danda — Hindi's sentence terminator, which is not ASCII
punctuation — in place. The result is not an error but an Exact Match score a few
points low on every Hindi row. Our normalizer dispatches on script and
additionally unifies digit systems, monetary surface forms, and the joined and
separated spellings of common Hindi postpositions.

### E. Statistical protocol

Every headline figure carries a 95% confidence interval from a percentile
bootstrap over queries with 1000 resamples. Comparisons between systems use a
two-sided paired randomization test with 10000 trials, pairing on question
identifier rather than position.

We use a permutation test rather than a t-test because per-query metric values
are bounded, discrete and strongly non-normal — Recall@5 on a single question
with one gold passage takes values in {0, 1} — so a test assuming normality is
inappropriate. The permutation test assumes only exchangeability under the null,
which is precisely the statement that two systems are equivalent.

With a test split in the low hundreds, differences of two or three points are
routinely within noise. We report intervals alongside every comparison because
the alternative overstates the evidence at no saving.
