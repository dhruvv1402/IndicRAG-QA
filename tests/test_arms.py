"""Module 4 generation arms and grounding metrics.

Run against a scripted provider, so the arm logic -- which context each arm
receives, what the cache keys on, how abstention is recorded -- is tested without
model inference.

The arm D guard matters most. `docs/ARCHITECTURE.md` §12 makes the oracle arm the
thing that separates retrieval error from generation error, and an oracle item
silently falling back to closed-book would corrupt exactly that decomposition.
"""

from __future__ import annotations

import json

import pytest

from indicrag.evaluation.grounding import (
    GroundingReport,
    rouge_l_precision,
    score_generation,
)
from indicrag.models import Passage, QAItem
from indicrag.rag.arms import ARMS, GenerationCache, prompt_hash, run_arm
from indicrag.rag.prompts import REFUSAL, build_answer_prompt, format_passages


def _p(pid, text="The scheme provides Rs. 12,000 per annum to eligible students.", lang="en"):
    return Passage(
        passage_id=pid,
        doc_id="d1",
        scheme="s1",
        lang=lang,
        text=text,
        section_path="Eligibility",
        token_count=40,
    )


def _item(qid="q1", gold=("g1",)):
    return QAItem(
        id=qid,
        question="How much does the scheme provide?",
        query_lang="en",
        passage_lang="en",
        answerable=True,
        gold_passage_ids=list(gold),
        scheme="s1",
    )


class _Provider:
    """Answers with a fixed object, recording every prompt."""

    def __init__(self, answer="Rs. 12,000 per annum", answerable=True):
        self.prompts: list[str] = []
        self._answer, self._answerable = answer, answerable

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "answerable": self._answerable,
                "answer": self._answer,
                "citation": "g1",
                "confidence": 0.8,
            }
        )


# --- arm context -----------------------------------------------------------------


def test_closed_book_arm_receives_no_passage_text():
    """Arm A measures what the model asserts with no evidence; leaking the
    passage into its prompt would destroy the control."""
    prov = _Provider()
    run_arm(ARMS["A"], [_item()], [_p("g1")], prov)
    assert "Rs. 12,000" not in prov.prompts[0]
    assert "id=g1" not in prov.prompts[0]


def test_oracle_arm_receives_the_gold_passage():
    prov = _Provider()
    run_arm(ARMS["D"], [_item()], [_p("g1")], prov)
    assert "id=g1" in prov.prompts[0]


def test_retrieval_arm_receives_what_the_retriever_returned():
    prov = _Provider()
    passages = [_p("g1"), _p("x1", "Unrelated text about something else entirely.")]
    run_arm(ARMS["C"], [_item()], passages, prov, retrieve=lambda i: ["x1"])
    assert "id=x1" in prov.prompts[0]
    assert "id=g1" not in prov.prompts[0]


def test_oracle_item_with_no_resolvable_gold_is_skipped_not_downgraded():
    """Falling back to closed-book here would inflate measured retrieval error,
    which is the quantity arm D exists to isolate."""
    prov = _Provider()
    out = run_arm(ARMS["D"], [_item(gold=["missing"])], [_p("g1")], prov)
    assert out == []
    assert prov.prompts == []


def test_a_retrieval_arm_without_a_retriever_raises():
    with pytest.raises(ValueError):
        run_arm(ARMS["B"], [_item()], [_p("g1")], _Provider())


# --- output handling -------------------------------------------------------------


def test_an_unanswerable_reply_becomes_the_exact_refusal_string():
    prov = _Provider(answer="", answerable=False)
    out = run_arm(ARMS["D"], [_item()], [_p("g1")], prov)
    assert out[0].answerable is False
    assert out[0].answer == REFUSAL


def test_an_empty_answer_counts_as_abstention_even_if_flagged_answerable():
    prov = _Provider(answer="   ", answerable=True)
    out = run_arm(ARMS["D"], [_item()], [_p("g1")], prov)
    assert out[0].answerable is False


def test_unparseable_output_is_recorded_not_dropped():
    class Bad:
        def __call__(self, prompt):
            return "I cannot answer that."

    out = run_arm(ARMS["D"], [_item()], [_p("g1")], Bad())
    assert len(out) == 1 and out[0].answerable is False


# --- cache -----------------------------------------------------------------------


def test_cache_prevents_a_second_model_call(tmp_path):
    cache = GenerationCache(tmp_path / "gen.jsonl")
    prov = _Provider()
    run_arm(ARMS["D"], [_item()], [_p("g1")], prov, cache=cache, model="m")
    run_arm(ARMS["D"], [_item()], [_p("g1")], prov, cache=cache, model="m")
    assert len(prov.prompts) == 1


def test_cache_survives_reload(tmp_path):
    path = tmp_path / "gen.jsonl"
    prov = _Provider()
    run_arm(ARMS["D"], [_item()], [_p("g1")], prov, cache=GenerationCache(path), model="m")
    reloaded = GenerationCache(path)
    assert len(reloaded) == 1
    run_arm(ARMS["D"], [_item()], [_p("g1")], prov, cache=reloaded, model="m")
    assert len(prov.prompts) == 1


def test_cache_key_separates_models_and_arms():
    a = prompt_hash("p", "model-a", "A")
    assert a != prompt_hash("p", "model-b", "A")
    assert a != prompt_hash("p", "model-a", "D")


# --- prompt formatting -----------------------------------------------------------


def test_passages_are_numbered_in_retrieval_order_with_ids():
    text = format_passages([_p("a1"), _p("b2")])
    assert text.index("id=a1") < text.index("id=b2")
    assert "[1]" in text and "[2]" in text


def test_answer_prompt_carries_the_question_and_the_abstain_instruction():
    prompt = build_answer_prompt("How much?", [_p("g1")])
    assert "How much?" in prompt
    assert "answerable" in prompt


# --- grounding -------------------------------------------------------------------


def test_rouge_l_precision_endpoints():
    assert rouge_l_precision("Rs. 12,000", "The scheme gives Rs. 12,000 a year.") == 1.0
    assert rouge_l_precision("completely unrelated wording", "Rs. 12,000 per annum") == 0.0


def test_rouge_l_respects_order():
    """A subsequence must appear in order, not merely as a bag of words."""
    forward = rouge_l_precision("a b c", "x a y b z c")
    scrambled = rouge_l_precision("c b a", "x a y b z c")
    assert forward == 1.0
    assert scrambled < forward


def test_an_extracted_answer_is_fully_supported():
    o = score_generation(
        item_id="q1",
        arm="D",
        answer="Rs. 12,000 per annum",
        abstained=False,
        evidence="The scheme provides Rs. 12,000 per annum to eligible students.",
    )
    assert o.lexically_supported


def test_an_invented_answer_is_not_supported():
    o = score_generation(
        item_id="q1",
        arm="A",
        answer="The allowance is fifty thousand rupees monthly",
        abstained=False,
        evidence="The scheme provides Rs. 12,000 per annum.",
    )
    assert not o.lexically_supported


def test_abstentions_are_excluded_from_the_support_denominator():
    """Refusing to answer is not a grounding failure; counting it as one would
    reward a system that answers nothing."""
    report = GroundingReport(
        "s",
        [
            score_generation(
                item_id="1", arm="C", answer="Rs. 12,000", abstained=False,
                evidence="gives Rs. 12,000 yearly",
            ),
            score_generation(item_id="2", arm="C", answer="", abstained=True, evidence="x"),
        ],
    )
    assert report.n_answered("C") == 1
    assert report.citation_support_rate("C") == 1.0
    assert report.abstention_rate("C") == 0.5


def test_disagreements_between_lexical_and_entailment_are_surfaced():
    report = GroundingReport(
        "s",
        [
            score_generation(
                item_id="1", arm="C", answer="Rs. 12,000", abstained=False,
                evidence="gives Rs. 12,000 yearly", entailment=0.1,
            ),
        ],
    )
    assert len(report.disagreements("C")) == 1
