"""Tolerant JSONL loading, which the whole annotation workflow sits on.

`models.py` drops unknown keys rather than raising, so that adding a field
mid-project does not invalidate the file an annotator is halfway through. That
tolerance is deliberate and it has a cost the module names: a misspelled key
looks exactly like a new one. The stash in `unknown_fields` is the stated
mitigation, and nothing read it until `dataset stats` did -- so the mitigation
existed on paper only.

287 lines of serialization had no dedicated tests.
"""

from __future__ import annotations

import json

from indicrag.models import Passage, QAItem, read_jsonl, write_jsonl

_ROW = {"id": "qa-1", "question": "q", "query_lang": "en", "answerable": True}


def _write(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_an_unknown_key_is_kept_rather_than_dropped(tmp_path):
    path = tmp_path / "gold.jsonl"
    _write(path, [{"id": "qa-1", "question": "q", "query_lang": "en",
                   "answerable": True, "answer_gold_hin": "typo'd key"}])
    item = next(iter(read_jsonl(path, QAItem)))
    assert item.unknown_fields == {"answer_gold_hin": "typo'd key"}


def test_optional_fields_take_their_defaults(tmp_path):
    """Tolerance covers optional fields. Required ones are still required --
    `query_lang` and `answerable` have no sensible default, and inventing one
    would let a malformed row into the evaluation looking complete."""
    path = tmp_path / "gold.jsonl"
    _write(path, [{"id": "qa-1", "question": "q", "query_lang": "en",
                   "answerable": True}])
    item = next(iter(read_jsonl(path, QAItem)))
    assert item.gold_passage_ids == []
    assert item.answer_gold == ""
    assert item.verified is False


def test_blank_lines_are_skipped_so_hand_edits_still_load(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps(_ROW) + "\n\n\n" + json.dumps({**_ROW, "id": "qa-2"}) + "\n",
        encoding="utf-8",
    )
    assert len(list(read_jsonl(path, QAItem))) == 2


def test_a_round_trip_preserves_devanagari(tmp_path):
    path = tmp_path / "p.jsonl"
    original = Passage(
        passage_id="p1", doc_id="d", scheme="s", lang="hi",
        text="मनरेगा के तहत रोजगार", section_path="परिचय",
    )
    write_jsonl(path, [original])
    back = next(iter(read_jsonl(path, Passage)))
    assert back.text == original.text
    assert back.section_path == original.section_path


def test_devanagari_is_not_escaped_on_disk(tmp_path):
    """Escaping non-ASCII makes the file unreadable in exactly the situation
    where someone needs to read it: inspecting an annotation by eye."""
    path = tmp_path / "p.jsonl"
    write_jsonl(path, [Passage(passage_id="p1", doc_id="d", scheme="s",
                               lang="hi", text="मनरेगा")])
    assert "मनरेगा" in path.read_text(encoding="utf-8")


def test_reading_a_missing_file_yields_nothing_rather_than_raising(tmp_path):
    assert list(read_jsonl(tmp_path / "absent.jsonl", QAItem)) == []


def test_unknown_fields_are_reported_by_dataset_stats():
    """The mitigation only works if something surfaces it."""
    from indicrag.dataset.split import format_unknown_fields

    item = QAItem(id="qa-1", question="q", query_lang="en",
                  passage_lang="en", answerable=True)
    item.unknown_fields = {"answer_gold_hin": "x"}
    text = "\n".join(format_unknown_fields([item, item]))
    assert "answer_gold_hin" in text
    assert "reaching the evaluation" in text


def test_nothing_is_reported_when_every_key_is_recognised():
    from indicrag.dataset.split import format_unknown_fields

    item = QAItem(id="qa-1", question="q", query_lang="en",
                  passage_lang="en", answerable=True)
    assert format_unknown_fields([item]) == []
