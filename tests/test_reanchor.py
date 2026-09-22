"""Re-anchoring gold citations across segmentation versions.

The failure this exists to prevent is silent: a cited ID that still resolves but
names different text. So every test here is about what the tool refuses to guess
as much as what it maps.
"""

from __future__ import annotations

from indicrag.dataset.reanchor import apply, format_reanchor, reanchor
from indicrag.models import Passage, QAItem


def _p(pid: str, text: str) -> Passage:
    return Passage(
        passage_id=pid, doc_id=pid.split("#")[0], scheme="s", lang="en",
        text=text, text_raw=text, section_path="", page=0, char_span=(0, 0),
        token_count=len(text.split()),
    )


def _item(qid: str, pid: str, answer: str) -> QAItem:
    return QAItem(
        id=qid, question="q", query_lang="en", passage_lang="en",
        answerable=True, gold_passage_ids=[pid], answer_gold=answer, verified=True,
    )


INTRO = "The scheme was launched in 2015 by the ministry of rural development."
BUDGET = "Its budget allocation for 2021 was forty billion rupees across all states."
UPI = "Payments are routed through the unified payments interface nationwide."


def test_every_status_is_reached_and_only_confident_ones_are_rewritten():
    old = [
        _p("d#p0000", INTRO),
        _p("d#p0001", f"{BUDGET} {UPI}"),  # split in v2
        _p("d#p0002", "Eligibility covers every rural household below the poverty line."),
        _p("e#p0000", "Alpha beta gamma delta epsilon zeta eta theta iota kappa."),
    ]
    new = [
        _p("d#p0000", INTRO),                    # unchanged
        _p("d#p0001", BUDGET),                   # piece carrying the answer
        _p("d#p0002", UPI),                      # the other piece
        _p("d#p0003", "Eligibility covers every rural household below the poverty line."),  # moved
        _p("e#p0000", "Alpha beta gamma delta."),
        _p("e#p0001", "Epsilon zeta eta theta iota kappa."),
    ]
    items = [
        _item("i-unchanged", "d#p0000", "2015"),
        _item("i-split", "d#p0001", "forty billion rupees"),
        _item("i-moved", "d#p0002", "rural household"),
        _item("i-review", "e#p0000", "lambda"),  # answer in no piece
    ]
    res = reanchor(items, old, new)
    status = {r.item_id: r.status for r in res.items}
    assert status == {
        "i-unchanged": "unchanged",
        "i-split": "split-answer",
        "i-moved": "moved",
        "i-review": "review",
    }
    assert res.repointed_ids == 3  # d#p0001, d#p0002, e#p0000 all name new text

    out = {i.id: i for i in apply(items, res, version=2)}
    assert out["i-unchanged"].gold_passage_ids == ["d#p0000"]
    assert out["i-unchanged"].verified is True
    assert out["i-split"].gold_passage_ids == ["d#p0001"]
    assert out["i-moved"].gold_passage_ids == ["d#p0003"]
    # Not guessed: the old ID is kept, and the item is no longer verified.
    assert out["i-review"].gold_passage_ids == ["e#p0000"]
    assert out["i-review"].verified is False
    assert "review" in out["i-review"].notes


def test_a_passage_absorbed_into_a_larger_one_is_contained():
    old = [_p("d#p0000", INTRO), _p("d#p0001", BUDGET)]
    new = [_p("d#p0000", f"{INTRO} {BUDGET}")]
    res = reanchor([_item("i", "d#p0001", "forty billion")], old, new)
    assert res.items[0].status == "contained"
    assert res.items[0].new_ids == ["d#p0000"]


def test_when_two_pieces_both_carry_the_answer_it_is_left_for_review():
    old = [_p("d#p0000", "The fund grew to forty billion. Later the fund held forty billion.")]
    new = [
        _p("d#p0000", "The fund grew to forty billion."),
        _p("d#p0001", "Later the fund held forty billion."),
    ]
    res = reanchor([_item("i", "d#p0000", "forty billion")], old, new)
    assert res.items[0].status == "review"
    assert "2 pieces carry the answer" in res.items[0].citations[0].detail


def test_the_report_lists_what_was_not_mapped():
    old = [_p("e#p0000", "Alpha beta gamma delta epsilon zeta eta theta iota kappa.")]
    new = [_p("e#p0000", "Alpha beta gamma delta."), _p("e#p0001", "Epsilon zeta eta theta iota kappa.")]
    lines = format_reanchor(reanchor([_item("i", "e#p0000", "lambda")], old, new),
                            old_version=1, new_version=2)
    text = "\n".join(lines)
    assert "NEEDS REVIEW" in text and "e#p0000" in text
