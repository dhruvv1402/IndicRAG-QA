# Demo

The deliverable demo is the CLI. `indicrag ask` runs the full pipeline —
language identification, retrieval, answerability, grounded generation — and
prints the complete response object from PRD §9, so every field the brief
requires is visible in one place rather than summarised.

```bash
indicrag ask "Sukanya Samriddhi account kholne ke liye kya chahiye?"
indicrag ask "मनरेगा के तहत कितने दिन का रोजगार मिलता है?" --k 5
indicrag ask "What is the eligibility for MGNREGA?" --method bm25
```

`--method` selects the retriever, which is what makes the demo an argument
rather than a display: the same query can be run through `hybrid` (script-aware,
the default), `hybrid-rrf` (plain RRF), `bm25` or `dense`, and the citations
change in the way §VI of the paper predicts.

Transcripts below are real output, abridged to the fields under discussion.

---

## 1. Code-mixed query, monolingual evidence

    $ indicrag ask "Sukanya Samriddhi account kholne ke liye kya chahiye?"

    query_type : Code-Mixed | lang hi-en
    answerable : ANSWERABLE | conf 0.6227
    method     : hybrid | top_score 0.0328
    answer     : Sukanya Samriddhi Yojana (Girl Child Prosperity Account) is a
                 Government of India backed saving scheme targeted at the
                 parents of girl children.
    citations  :
       sukanya-en#p0000             en  score=0.0328
       sukanya-en#p0006             en  score=0.032
       sukanya-en#p0009             en  score=0.0318
       sukanya-en#p0005             en  score=0.0313

Romanized Hindi with English nouns, correctly classified as Code-Mixed, with
every citation on the right scheme.

The `top_score` of 0.0328 is worth noticing. Reciprocal Rank Fusion sums
`1/(60+rank)`, so a fused score sits near 0.03, while BM25 is unbounded and runs
around 10 on this corpus. The scale is the quickest way to confirm which
retriever actually ran — and it is how a silent regression was caught in which
`ask` built only a lexical index, so the default `hybrid` degraded to plain BM25
and nothing in the output said so.

## 2. Devanagari query, cross-script retrieval

    $ indicrag ask "मनरेगा के तहत कितने दिन का रोजगार मिलता है?"

    query_type : Indic | lang hi | answer_lang hi
    method     : hybrid | top_score 0.0325
    answer     : 2008-09 के दौरान 4,49,40,870 ग्रामीण परिवारों को मनरेगा के तहत
                 रोजगार उपलब्ध कराया गया, जहां प्रत्येक परिवार में 48 कार्य दिवस
                 का राष्ट्रीय औसत था।
    citations  :
       mgnrega-hi#p0004             hi  score=0.0325
       mgnrega-hi#p0002             hi  score=0.0323
       mgnrega-hi#p0000             hi  score=0.032
       mgnrega-hi#p0005             hi  score=0.031
       mgnrega-en#p0001             en  score=0.0303

This is the case the project exists for. The query is Devanagari, the answer
comes back in Devanagari, and the fifth citation is an **English** passage —
retrieved for a Devanagari query, which BM25 cannot do at any rank, because the
two share no tokens. Under plain RRF that passage would have been docked a full
contribution for the lexical retriever's silence; under script-aware fusion it
is scored on the evidence that actually existed about it.

Numerals are preserved in the form the source writes them
(4,49,40,870 — the Indian grouping), which the script-aware normalizer handles
rather than mangling into Western thousands separators.

## 3. A query the corpus cannot answer

    $ indicrag ask "Scholarship ke liye minimum eligibility kya hai?"

    query_type : Code-Mixed | lang hi-en
    answerable : ANSWERABLE | conf 0.4591
    answer     : The Pension Parishad ... demanding ... old age pension ...
    citations  :
       nsap-en#p0010                en  score=10.2878

**This one is wrong, and it is included because it is wrong.**

The corpus holds 40 schemes and none of them is a scholarship scheme; the
nearest are education schemes (`rte-act`, `sarva-shiksha`, `kgbv`,
`midday-meal`). The right response is a refusal. Instead the system answered
confidently about pensions.

Two separate things produced that, and only one is a defect:

- The score of 10.2878 is BM25-scale, so this transcript predates the fix in
  §1 and was retrieved lexically. That was the defect, and it is fixed.
- The system still does not abstain, because `ask` runs with `tau = 0` and
  abstains only when retrieval returns nothing at all. That default is
  deliberate — an uncalibrated threshold silently suppresses answers and makes
  retrieval look worse than it is — but it means the interactive path does not
  yet carry the answerability signal that `eval answerability` measures.

Wiring the calibrated threshold into `ask` needs the dev split, which needs
human verification of the gold set. Until then the honest statement is that the
demo demonstrates retrieval and grounding, not abstention, and §IX of the paper
says so.

---

## What the demo does not do

- **No web interface.** The CLI is the deliverable; a Streamlit surface was
  listed as optional and is not built.
- **No abstention on the interactive path**, for the reason above.
- **First call is slow.** The dense index and encoder load per invocation
  (~30 s). There is no server mode; the evaluation harness amortises this by
  loading once per run, which is why `eval` exists as a separate command rather
  than a loop over `ask`.
