---
title: IndicRAG-QA
emoji: 🔎
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Cross-lingual QA for Hindi, English and Hinglish queries
---

# IndicRAG-QA

Evidence-grounded question answering for Hindi, English and Romanized
Hindi–English (Hinglish) questions over Indian government scheme documents,
and the retrieval finding behind it: standard hybrid rank fusion penalises
passages written in another script, and *script-aware fusion* removes that
penalty (cross-lingual Recall@5 0.274 → 0.685 on the sealed test split).

Ask a question on the page in any of the three languages. Answers are either
the best-supporting sentence from the retrieved passages or, when the Space
has a `GROQ_API_KEY` secret, a hosted model's answer grounded in the same
passages. Each answer shows its citations; the system refuses when the
passages do not contain the answer.

**What is sent where.** Extracted answers are computed inside this Space. If
you choose the hosted model, your question and the retrieved passages are
sent to Groq. Questions are not logged.

**Sources.** Passages come from English and Hindi Wikipedia articles, by
Wikipedia contributors, under CC BY-SA 4.0; source URLs and hashes are in
`data/corpus_manifest.jsonl`.

Built with `scripts/build-space.py` from the IndicRAG-QA repository
(https://github.com/dhruvv1402/IndicRAG-QA), which holds the code, the
evaluation reports and the paper.
