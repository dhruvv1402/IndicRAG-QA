"""Second-pass re-labelling and Cohen's kappa.

PRD §10.2 gates Module 5 on kappa >= 0.70, so these tests pin the two things
that would quietly invalidate that gate: showing the re-labeller the first pass's
verdict (which measures compliance, not agreement) and sampling in a way that
misses the hard unanswerable classes (which measures the easy boundary).
"""

from __future__ import annotations

import json

from indicrag.dataset.second_pass import (
    KAPPA_GATE,
    blind,
    compare,
    draw_sample,
    format_agreement,
    write_blind_sample,
)
from indicrag.models import QAItem


def _item(qid, *, answerable=True, klass=None, verified=True):
    return QAItem(
        id=qid,
        question=f"question {qid}",
        query_lang="en",
        passage_lang="en" if answerable else "",
        answerable=answerable,
        unanswerable_class=klass,
        answer_gold="Rs. 12,000" if answerable else "",
        gold_passage_ids=["p1"] if answerable else [],
        scheme="s1",
        verified=verified,
    )


def _corpus():
    items = [_item(f"a{i:03d}") for i in range(80)]
    for klass, n in [
        ("out-of-scope", 25),
        ("near-miss", 30),
        ("false-premise", 15),
        ("under-specified", 10),
    ]:
        items += [_item(f"{klass}-{i:03d}", answerable=False, klass=klass) for i in range(n)]
    return items


# --- blinding --------------------------------------------------------------------


def test_the_blind_view_hides_every_trace_of_the_first_verdict():
    """A re-labeller shown `answerable: false` will agree, and kappa then
    measures compliance rather than agreement."""
    view = blind(_item("u1", answerable=False, klass="near-miss"))
    assert view["answerable"] is None
    assert "unanswerable_class" not in view
    assert "answer_gold" not in view
    assert "verified" not in view


def test_the_blind_view_keeps_what_is_needed_to_judge():
    view = blind(_item("a1"), ["p7", "p9"])
    assert view["question"] and sorted(view["evidence_ids"]) == ["p1", "p7", "p9"]


def test_the_evidence_does_not_give_the_label_away():
    """Unanswerable items cite nothing, so passing gold IDs through made an
    empty evidence list mean 'unanswerable'. Both kinds must look alike."""
    ans = blind(_item("a1"), ["p7", "p9"])
    una = blind(_item("u1", answerable=False, klass="near-miss"), ["p7", "p9", "p4"])
    assert "gold_passage_ids" not in ans and "gold_passage_ids" not in una
    assert set(ans) == set(una)
    assert len(ans["evidence_ids"]) == len(una["evidence_ids"]) == 3


def test_the_evidence_count_is_fixed_whatever_the_label():
    """Gold passages were added on top of retrieval, so an answerable item whose
    gold was not retrieved showed one passage more than any unanswerable one."""
    retrieved = ["r1", "r2", "r3", "r4", "r5"]
    ans = blind(_item("a1"), retrieved)  # gold p1 not among the retrieved
    una = blind(_item("u1", answerable=False, klass="near-miss"), retrieved)
    assert len(ans["evidence_ids"]) == len(una["evidence_ids"]) == 5
    assert "p1" in ans["evidence_ids"]  # the gold passage is never the one dropped


def test_written_sample_is_valid_jsonl_with_null_labels(tmp_path):
    path = tmp_path / "sample.jsonl"
    n = write_blind_sample(draw_sample(_corpus()), path)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == n
    assert all(rec["answerable"] is None for rec in lines)


# --- sampling --------------------------------------------------------------------


def test_sample_covers_every_unanswerable_class():
    """A flat 15% draw routinely misses false-premise -- 15 items of 400 -- and
    kappa is only interesting where the boundary is hard."""
    sample = draw_sample(_corpus())
    classes = {i.unanswerable_class for i in sample if not i.answerable}
    assert classes == {"out-of-scope", "near-miss", "false-premise", "under-specified"}


def test_sample_is_about_the_requested_fraction():
    sample = draw_sample(_corpus(), fraction=0.15)
    assert 20 <= len(sample) <= 35  # 15% of 160, plus per-stratum rounding


def test_sample_is_deterministic_under_the_same_seed():
    a = [i.id for i in draw_sample(_corpus(), seed=7)]
    b = [i.id for i in draw_sample(_corpus(), seed=7)]
    assert a == b


def test_unverified_items_are_never_sampled():
    """Agreement on labels nobody has checked would measure nothing."""
    items = [_item(f"a{i}", verified=False) for i in range(50)]
    assert draw_sample(items) == []


def test_rejected_items_are_never_sampled():
    items = _corpus()
    for i in items:
        i.notes = "REJECTED unusable"
    assert draw_sample(items) == []


# --- agreement -------------------------------------------------------------------


def _write_labels(path, pairs):
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item_id, label in pairs:
            fh.write(json.dumps({"id": item_id, "answerable": label}) + "\n")


def test_perfect_agreement_scores_one(tmp_path):
    items = _corpus()
    sample = draw_sample(items)
    path = tmp_path / "second.jsonl"
    _write_labels(path, [(i.id, i.answerable) for i in sample])
    result = compare(items, path)
    assert result.kappa == 1.0
    assert result.passes_gate
    assert result.disagreements == []


def test_inverted_labels_score_below_chance(tmp_path):
    items = _corpus()
    sample = draw_sample(items)
    path = tmp_path / "second.jsonl"
    _write_labels(path, [(i.id, not i.answerable) for i in sample])
    result = compare(items, path)
    assert result.kappa < 0
    assert not result.passes_gate


def test_disagreements_are_listed_for_review(tmp_path):
    items = _corpus()
    sample = draw_sample(items)
    flipped = {sample[0].id, sample[1].id}
    path = tmp_path / "second.jsonl"
    _write_labels(
        path, [(i.id, (not i.answerable) if i.id in flipped else i.answerable) for i in sample]
    )
    result = compare(items, path)
    assert len(result.disagreements) == 2
    assert {d[0] for d in result.disagreements} == flipped


def test_unlabelled_rows_are_ignored_rather_than_counted(tmp_path):
    items = _corpus()
    sample = draw_sample(items)
    path = tmp_path / "second.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for n, item in enumerate(sample):
            label = item.answerable if n < 5 else None
            fh.write(json.dumps({"id": item.id, "answerable": label}) + "\n")
    result = compare(items, path)
    assert result.n == 5


def test_an_empty_second_pass_does_not_raise(tmp_path):
    path = tmp_path / "second.jsonl"
    path.write_text("", encoding="utf-8")
    result = compare(_corpus(), path)
    assert result.n == 0 and result.kappa == 0.0
    assert "No labels found" in "\n".join(format_agreement(result))


def test_the_report_states_the_gate_verdict(tmp_path):
    items = _corpus()
    sample = draw_sample(items)
    path = tmp_path / "second.jsonl"
    _write_labels(path, [(i.id, i.answerable) for i in sample])
    text = "\n".join(format_agreement(compare(items, path)))
    assert f"{KAPPA_GATE:.2f}" in text
    assert "PASSES" in text


def test_agreement_between_model_passes_is_not_reported_as_passing_the_gate(tmp_path):
    items = _corpus()
    sample = draw_sample(items)
    path = tmp_path / "second.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for i in sample:
            row = {"id": i.id, "answerable": i.answerable, "labelled_by": "model:x"}
            fh.write(json.dumps(row) + "\n")
    text = "\n".join(format_agreement(compare(items, path)))
    assert "PASSES" not in text
    assert "not between" in text and "model:x" in text


def test_the_blind_labeller_shows_no_verdict_and_records_the_labeller(tmp_path):
    import json

    from indicrag.dataset.second_pass import label_blind
    from indicrag.models import Passage

    passage = Passage(passage_id="d#p0", doc_id="d", scheme="s", lang="en",
                      text="Loans are repaid over up to 7 years.", section_path="",
                      token_count=7)
    rows = [
        {"id": "a", "question": "Repayment period?", "evidence_ids": ["d#p0"], "answerable": None},
        {"id": "b", "question": "Interest in Nagaland?", "evidence_ids": ["d#p0"], "answerable": None},
    ]
    path = tmp_path / "blind.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    shown: list[str] = []
    answers = iter(["y", "w"])
    done, total = label_blind(path, [passage], labeller="Second Person",
                              ask=lambda _p: next(answers), say=shown.append)
    assert (done, total) == (1, 2)
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert saved[0]["answerable"] is True and saved[0]["labelled_by"] == "Second Person"
    assert saved[1]["answerable"] is None  # quit before it; resumable
    text = "\n".join(shown)
    assert "Loans are repaid" in text and "gold" not in text.lower()
