"""Module 6 case selection.

The selection rule is the thing worth testing. Picking examples after seeing
which look interesting turns an error analysis into a set of anecdotes, so the
taxonomy is fixed and the within-category rule is deterministic: the failure
where the system was most confident and most wrong.
"""

from __future__ import annotations

from indicrag.evaluation.errors import CATEGORIES, ErrorCase, categorise, collect_cases
from indicrag.evaluation.retrieval import Outcome
from indicrag.models import Passage, QAItem


def _p(pid, scheme="s1", lang="en", section="Eligibility"):
    return Passage(
        passage_id=pid,
        doc_id=f"{scheme}-{lang}",
        scheme=scheme,
        lang=lang,
        text="x " * 80,
        section_path=section,
        token_count=80,
    )


def _item(qid, question, *, ql="en", pl="en", scheme="s1", gold=("g1",)):
    return QAItem(
        id=qid,
        question=question,
        query_lang=ql,
        passage_lang=pl,
        answerable=True,
        gold_passage_ids=list(gold),
        scheme=scheme,
    )


def _outcome(item_id, gold, retrieved, scores):
    return Outcome(
        item_id=item_id,
        slice_key="EN->EN",
        language_group="monolingual",
        gold=list(gold),
        retrieved=list(retrieved),
        scores=list(scores),
    )


# --- categorisation --------------------------------------------------------------


def test_code_mixed_queries_are_tagged_as_such():
    tags = categorise(_item("q", "Scholarship ke liye eligibility kya hai?"), {}, {})
    assert "code-mixing" in tags
    assert "transliteration-variation" in tags


def test_cross_lingual_items_are_tagged_from_the_language_pair():
    tags = categorise(_item("q", "What is the limit?", ql="en", pl="hi"), {}, {})
    assert "cross-lingual-gap" in tags


def test_amount_and_date_detectors_fire():
    assert "amount" in categorise(_item("q", "Is the limit Rs 3,50,000 per year?"), {}, {})
    assert "date" in categorise(_item("q", "What changed in 2019 for this scheme?"), {}, {})


def test_short_queries_are_tagged_ambiguous():
    tags = categorise(_item("q", "What is the limit?"), {}, {})
    assert "short-query" in tags and "ambiguity" in tags


def test_long_tail_schemes_are_detected_from_corpus_size():
    tags = categorise(_item("q", "A reasonably long question about the scheme here", scheme="tiny"), {}, {"tiny": 3})
    assert "long-tail-scheme" in tags


def test_every_item_gets_at_least_one_category():
    assert categorise(_item("q", "zzz qqq wwww eeee rrrr tttt"), {}, {})


# --- selection -------------------------------------------------------------------


def test_only_failures_are_collected():
    passages = [_p("g1"), _p("x1")]
    items = [_item("hit", "Rs 500 question here", gold=["g1"]), _item("miss", "Rs 500 other question", gold=["g1"])]
    outcomes = [
        _outcome("hit", ["g1"], ["g1", "x1"], [0.9, 0.1]),
        _outcome("miss", ["g1"], ["x1"], [0.8]),
    ]
    cases = collect_cases(outcomes, items, passages, k=5, per_category=5)
    assert {c.query for c in cases} == {"Rs 500 other question"}


def test_the_most_confident_failure_wins_its_category():
    """A confident miss is a systematic blind spot; a borderline one is noise."""
    passages = [_p("g1"), _p("x1")]
    items = [
        _item("low", "Rs 100 question about amounts", gold=["g1"]),
        _item("high", "Rs 900 question about amounts", gold=["g1"]),
    ]
    outcomes = [
        _outcome("low", ["g1"], ["x1"], [0.10]),
        _outcome("high", ["g1"], ["x1"], [0.95]),
    ]
    cases = collect_cases(outcomes, items, passages, k=5, per_category=1)
    amount_cases = [c for c in cases if c.category == "amount"]
    assert amount_cases and amount_cases[0].top_score == 0.95


def test_an_item_is_not_reported_twice_across_categories():
    passages = [_p("g1"), _p("x1")]
    items = [_item("q", "Rs 3,50,000 in 2019 ke liye kya hai?", gold=["g1"])]
    outcomes = [_outcome("q", ["g1"], ["x1"], [0.5])]
    cases = collect_cases(outcomes, items, passages, k=5, per_category=2)
    assert len({c.query for c in cases}) == len(cases)


def test_gold_rank_is_none_when_the_passage_was_never_returned():
    passages = [_p("g1"), _p("x1")]
    items = [_item("q", "Rs 500 question about amounts", gold=["g1"])]
    cases = collect_cases([_outcome("q", ["g1"], ["x1"], [0.5])], items, passages, k=5)
    assert cases and cases[0].gold_rank is None


def test_gold_rank_is_recorded_when_it_ranked_below_the_cut():
    passages = [_p("g1")] + [_p(f"x{i}") for i in range(6)]
    items = [_item("q", "Rs 500 question about amounts", gold=["g1"])]
    retrieved = [f"x{i}" for i in range(6)] + ["g1"]
    cases = collect_cases(
        [_outcome("q", ["g1"], retrieved, [0.5] * 7)], items, passages, k=5
    )
    assert cases and cases[0].gold_rank == 7


def test_other_runs_are_carried_into_the_case():
    """A miss every system shares points at the corpus; a miss one system makes
    points at that system."""
    passages = [_p("g1"), _p("x1"), _p("y1")]
    items = [_item("q", "Rs 500 question about amounts", gold=["g1"])]
    cases = collect_cases(
        [_outcome("q", ["g1"], ["x1"], [0.5])],
        items,
        passages,
        k=5,
        other_runs={"bm25": {"q": ["y1"]}},
    )
    assert cases[0].retrieved["primary"] == ["x1"]
    assert cases[0].retrieved["bm25"] == ["y1"]


def test_case_round_trips_through_jsonl_form():
    case = ErrorCase(
        case_id="E01",
        category="amount",
        query="q",
        query_type="English",
        gold_passage_ids=["g1"],
    )
    assert ErrorCase.from_dict(case.as_dict()) == case


def test_taxonomy_is_fixed_and_non_empty():
    assert len(CATEGORIES) == len(set(CATEGORIES)) >= 10


# --- the recovery summary --------------------------------------------------------


def test_the_report_computes_what_a_single_component_recovered():
    """The paper's origin story rests on this number -- the dense retriever
    alone found the gold in several cases and fusion lost every one. It used to
    be hand-written prose in the committed report, so regenerating the cases
    could change the figure while the sentence quoting it stayed put."""
    from indicrag.evaluation.errors import ErrorCase, format_errors

    cases = [
        ErrorCase(
            case_id="E1", category="code-mixing", query="q", query_type="Code-Mixed",
            gold_passage_ids=["g1"],
            retrieved={"primary": ["x"], "e5": ["g1"], "bm25": ["y"]},
        ),
        ErrorCase(
            case_id="E2", category="amount", query="q", query_type="English",
            gold_passage_ids=["g2"],
            retrieved={"primary": ["x"], "e5": ["g2"], "bm25": ["y"]},
        ),
    ]
    text = "\n".join(format_errors(cases, []))
    assert "e5         found the gold in  2 of 2; the fused system lost 2 of those" in text
    assert "bm25       found the gold in  0 of 2" in text


def test_a_component_the_fused_system_agreed_with_is_not_counted_as_lost():
    from indicrag.evaluation.errors import ErrorCase, format_errors

    cases = [
        ErrorCase(
            case_id="E1", category="amount", query="q", query_type="English",
            gold_passage_ids=["g1"],
            retrieved={"primary": ["g1"], "e5": ["g1"]},
        )
    ]
    text = "\n".join(format_errors(cases, []))
    assert "found the gold in  1 of 1; the fused system lost 0 of those" in text


def test_no_summary_when_only_the_primary_run_is_recorded():
    from indicrag.evaluation.errors import ErrorCase, format_errors

    cases = [
        ErrorCase(
            case_id="E1", category="amount", query="q", query_type="English",
            gold_passage_ids=["g1"], retrieved={"primary": ["x"]},
        )
    ]
    assert "WOULD HAVE RECOVERED" not in "\n".join(format_errors(cases, []))
