"""The pre-verification review: rule checks, merging, and how verify shows it.

The property that matters most is negative: nothing in this path may mark an
item verified. PRD §6.5 requires a person to confirm every item, and advice
that quietly became a verification would make every downstream number claim a
provenance it does not have.
"""

from __future__ import annotations

import pytest

from indicrag.dataset.review import format_review, merge, review
from indicrag.dataset.verify import render_item, take_suggestion, verify_loop
from indicrag.models import Passage, QAItem, read_jsonl, write_jsonl


def _p(pid: str, text: str, lang: str = "en") -> Passage:
    return Passage(
        passage_id=pid, doc_id=pid.split("#")[0], scheme="s", lang=lang,
        text=text, text_raw=text, section_path="", page=0, char_span=(0, 0),
        token_count=len(text.split()),
    )


def _item(qid: str, question: str, *, answer: str = "", pid: str = "d#p0", **kw) -> QAItem:
    base = dict(
        id=qid, question=question, query_lang="en", passage_lang="en",
        answerable=True, gold_passage_ids=[pid], answer_gold=answer,
    )
    base.update(kw)
    return QAItem(**base)


PASSAGE = _p(
    "d#p0",
    "The Stand Up India scheme offers loans with a repayment period of up to 7 years "
    "and a moratorium of up to 18 months, at a rate of MCLR plus 3 percent.",
)


def _codes(items, passages=(PASSAGE,)):
    return {r.item_id: set(r.codes) for r in review(items, list(passages))}


def test_a_clean_item_raises_nothing():
    codes = _codes([_item("ok", "What is the repayment period of a Stand Up India loan?", answer="7 years")])
    assert codes["ok"] == set()


def test_a_question_that_leans_on_its_passage_is_not_standalone():
    codes = _codes([_item("q", "According to the passage, what is the moratorium?", answer="18 months")])
    assert "not-standalone" in codes["q"]


def test_a_placeholder_question_is_caught():
    codes = _codes([_item("q", "अंग्रेज़ी प्रश्न", answer="7 years")])
    assert "placeholder" in codes["q"]


def test_an_answer_the_passage_does_not_contain_is_unsupported():
    codes = _codes([_item("q", "What is the repayment period of Stand Up India?", answer="five to ten months")])
    assert "answer-unsupported" in codes["q"]


def test_a_truncated_answer_is_caught():
    codes = _codes([_item("q", "What is the rate on Stand Up India loans?", answer="MCLR plus 3 percent (at")])
    assert "answer-truncated" in codes["q"]


def test_a_bare_number_needs_no_devanagari_but_english_words_do():
    hindi = _p("h#p0", "स्टैंड अप इंडिया योजना में भुगतान अवधि अधिकतम 7 वर्ष है।", lang="hi")
    items = [
        _item("num", "Stand Up India bhugtan avadhi kitne varsh?", answer="7", pid="h#p0",
              query_lang="hinglish", passage_lang="hi"),
        _item("eng", "Stand Up India bhugtan avadhi kitni hai?", answer="seven years", pid="h#p0",
              query_lang="hinglish", passage_lang="hi"),
    ]
    codes = _codes(items, [hindi])
    assert "missing-hindi" not in codes["num"]
    assert "missing-hindi" in codes["eng"]


def test_repeated_questions_are_flagged_on_every_copy():
    """The unanswerable scaffold cycles its stems; each copy must say so."""
    items = [
        _item(f"u{i}", "What is the income limit?", answerable=False, gold_passage_ids=[],
              passage_lang="", unanswerable_class="under-specified")
        for i in range(3)
    ]
    codes = _codes(items)
    assert all("duplicate" in codes[f"u{i}"] for i in range(3))


def test_an_unanswerable_item_that_cites_evidence_is_inconsistent():
    codes = _codes([_item("u", "What is the income limit?", answerable=False)])
    assert "inconsistent" in codes["u"]


def test_merge_attaches_notes_and_says_whose_they_are():
    items = [_item("a", "What is the repayment period of Stand Up India?", answer="7 years")]
    recs = merge(review(items, [PASSAGE]), {"a": {"id": "a", "verdict": "fix", "issues": ["answer-verbose"],
                                                    "suggested_answer": "up to 7 years", "comment": ""}},
                 reviewed_by="model pass")
    assert recs[0]["reviewed_by"] == "model pass"
    assert recs[0]["review"] == {"verdict": "fix", "issues": ["answer-verbose"], "suggested_answer": "up to 7 years"}
    assert "not a verification" in "\n".join(format_review(recs, items))


def test_merge_refuses_an_unknown_verdict():
    items = [_item("a", "What is the repayment period of Stand Up India?", answer="7 years")]
    with pytest.raises(ValueError):
        merge(review(items, [PASSAGE]), {"a": {"id": "a", "verdict": "verified"}}, reviewed_by="x")


def test_verify_shows_advice_under_the_evidence_and_hides_it_in_the_second_pass():
    item = _item("a", "What is the repayment period of Stand Up India?", answer="7 years")
    rec = {"id": "a", "rules": [], "reviewed_by": "model pass",
           "review": {"verdict": "fix", "comment": "trim it", "suggested_answer": "up to 7 years"}}
    shown = render_item(item, {"d#p0": PASSAGE}, index=1, total=1, assist=rec)
    text = "\n".join(shown)
    assert "pre-review (advice only)" in text and "trim it" in text
    assert text.index("evidence [d#p0]") < text.index("pre-review")
    hidden = "\n".join(render_item(item, {"d#p0": PASSAGE}, index=1, total=1, hide_label=True, assist=rec))
    assert "pre-review" not in hidden


def test_taking_a_suggestion_edits_but_never_verifies(tmp_path):
    item = _item("a", "Stand Up India loan?", answer="7 years")
    path = tmp_path / "gold.jsonl"
    write_jsonl(path, [item])
    rec = {"id": "a", "rules": [], "review": {"verdict": "fix",
           "suggested_question": "What is the repayment period of a Stand Up India loan?",
           "suggested_answer": "up to 7 years"}}

    assert take_suggestion(item, rec) == ["question", "answer_gold"]
    assert item.verified is False

    # Through the loop: take, then skip. The edit is saved; the item is not verified.
    answers = iter(["t", "s"])
    verify_loop(path, [PASSAGE], annotator="a1", ask=lambda _p: next(answers),
                say=lambda _m: None, assist={"a": rec})
    saved = list(read_jsonl(path, QAItem))[0]
    assert saved.answer_gold == "up to 7 years"
    assert saved.verified is False
