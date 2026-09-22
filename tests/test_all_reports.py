"""The `eval all` harness.

PRD NFR-6 requires every reported table to regenerate from one command. These
tests exercise the parts that make that safe without loading any encoder: the
provenance banner, and the guarantee that one missing input cannot take the whole
run down.

The banner test is the important one. A preliminary number reaching a paper
because nothing on the page said it was preliminary is the specific failure this
harness exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

from indicrag.evaluation.all_reports import RunSummary, StageResult, provenance, run_all
from indicrag.models import QAItem


def _item(qid, *, verified=True, answerable=True):
    return QAItem(
        id=qid,
        question="q",
        query_lang="en",
        passage_lang="en" if answerable else "",
        answerable=answerable,
        gold_passage_ids=["p1"] if answerable else [],
        verified=verified,
    )


# --- provenance ------------------------------------------------------------------


def test_banner_warns_loudly_when_any_item_is_unverified():
    text = "\n".join(provenance([_item("a"), _item("b", verified=False)], "evals/gold.jsonl"))
    assert "PRELIMINARY" in text
    assert "Verified:  1/2" in text
    assert "may be quoted" in text or "gold-set figure" in text


def test_banner_is_quiet_when_everything_is_verified():
    text = "\n".join(provenance([_item("a"), _item("b")], "evals/gold.jsonl"))
    assert "PRELIMINARY" not in text
    assert "Verified:  2/2" in text


def test_banner_reports_the_answerable_split():
    items = [_item("a"), _item("b"), _item("u", answerable=False)]
    text = "\n".join(provenance(items, "src"))
    assert "3 (2 answerable, 1 unanswerable)" in text


def test_banner_names_its_source():
    assert "evals/probes.jsonl" in "\n".join(provenance([_item("a")], "evals/probes.jsonl"))


# --- degradation -----------------------------------------------------------------


def test_a_missing_corpus_stops_the_run_with_a_reason(tmp_path, monkeypatch):
    """Nothing can be reported without passages, so this one is fatal -- but it
    must say why rather than raise."""
    monkeypatch.setenv("INDICRAG_DATA_DIR", str(tmp_path / "empty"))
    from indicrag.config import get_settings

    get_settings.cache_clear()
    summary = run_all(tmp_path / "out", gold_path=tmp_path / "none.jsonl")
    get_settings.cache_clear()
    assert len(summary.stages) == 1
    assert not summary.stages[0].ok
    assert "corpus segment" in summary.stages[0].skipped


def test_summary_distinguishes_produced_from_skipped():
    summary = RunSummary()
    summary.add(StageResult("report-corpus", path=Path("evals/report-corpus.txt"), lines=40))
    summary.add(StageResult("report-errors", skipped="no errors.jsonl"))
    text = "\n".join(summary.format())
    assert "ok       report-corpus" in text
    assert "skipped  report-errors" in text
    assert "1/2 stages" in text


def test_an_empty_summary_formats_without_dividing_by_zero():
    assert "0/0 stages" in "\n".join(RunSummary().format())


def test_a_model_verification_is_disclosed_on_every_report():
    """PRD §6.5 specifies a person. When a model did it instead, every report
    that rests on those items has to say so, so a copied number keeps it."""
    a = _item("a")
    b = _item("b")
    b.annotator = "model:claude-opus-5.5"
    text = "\n".join(provenance([a, b], "evals/gold.jsonl"))
    assert "1 by a person, 1 by a model" in text
    assert "MODEL-VERIFIED" in text
    assert "MODEL-VERIFIED" not in "\n".join(provenance([a], "evals/gold.jsonl"))
