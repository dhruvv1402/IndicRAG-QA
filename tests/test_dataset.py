"""Dataset bootstrapping, coverage and splitting.

These run against a scripted stub rather than a real model, so the generation
logic -- matrix quotas, which language gets translated, passage selection -- is
tested without a 2 GB download and without waiting on CPU inference.

The split tests matter most. `docs/PLAN.md` risk R13 calls test-split
contamination fatal to the paper, and unlike most bugs it cannot be repaired
after the fact: once a test number has been looked at, it cannot be unlooked at.
"""

from __future__ import annotations

import json

from indicrag.dataset.generate import MATRIX, generate_candidates, parse_reply, select_passages
from indicrag.dataset.split import UNANSWERABLE_TARGET, coverage, stratified_split
from indicrag.dataset.verify import cohens_kappa, progress_of
from indicrag.models import Passage, QAItem


def _passages(n_schemes: int = 6, per_scheme: int = 6) -> list[Passage]:
    out: list[Passage] = []
    for s in range(n_schemes):
        for lang in ("en", "hi"):
            for i in range(per_scheme):
                body = (
                    f"Scheme {s} provides Rs. {1000 * (i + 1)} per annum to eligible students "
                    if lang == "en"
                    else f"योजना {s} पात्र छात्रों को प्रति वर्ष {1000 * (i + 1)} रुपये देती है "
                )
                out.append(
                    Passage(
                        passage_id=f"sch{s}-{lang}#p{i:04d}",
                        doc_id=f"sch{s}-{lang}",
                        scheme=f"sch{s}",
                        lang=lang,
                        text=body * 12,
                        section_path="Eligibility",
                        token_count=120,
                    )
                )
    return out


class _Stub:
    """A scripted model that answers each prompt type plausibly.

    It has to be language-aware because the semantic checks in
    `dataset/validate.py` reject a Hindi prompt answered in English, a
    translation that returns its input, and a romanization that is really a
    translation. A stub returning one fixed English string for every call fails
    those checks -- correctly -- so it would test nothing but the rejection path.
    """

    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, prompt: str, grammar: str | None = None) -> str:
        self.prompts.append(prompt)
        if "Rewrite this Hindi question" in prompt:
            reply = {"question": "Yojana kitne rupye deti hai?", "answer": "", "kind": "fact"}
        elif "into Hindi" in prompt:  # translate -> Hindi
            reply = {"question": "योजना कितने रुपये देती है?", "answer": "", "kind": "fact"}
        elif "into English" in prompt:  # translate -> English
            reply = {"question": "How much does the scheme provide?", "answer": "", "kind": "fact"}
        elif "अनुच्छेद" in prompt:  # base generation, Hindi passage
            reply = {"question": "योजना कितने रुपये देती है?", "answer": "1000 रुपये", "kind": "number"}
        else:  # base generation, English passage
            reply = {"question": "What is the annual amount provided?", "answer": "Rs. 1000", "kind": "number"}
        return json.dumps(reply, ensure_ascii=False)


# --- parsing -------------------------------------------------------------------


def test_parse_reply_reads_a_well_formed_object():
    c = parse_reply('{"question": "How much?", "answer": "Rs. 500", "kind": "number"}')
    assert c is not None and c.question == "How much?" and c.kind == "number"


def test_parse_reply_treats_the_empty_reply_as_no_candidate():
    """The prompt explicitly allows this when a passage states no usable fact."""
    assert parse_reply('{"question": "", "answer": "", "kind": "fact"}') is None


def test_parse_reply_survives_malformed_json():
    assert parse_reply("not json at all") is None
    assert parse_reply("") is None


# --- selection -----------------------------------------------------------------


def test_selection_spreads_across_schemes_rather_than_draining_one():
    import random

    picked = select_passages(_passages(), "en", 12, rng=random.Random(0))
    schemes = {p.scheme for p in picked}
    assert len(picked) == 12
    assert len(schemes) >= 5, f"only drew from {schemes}"


def test_selection_respects_the_requested_language():
    import random

    picked = select_passages(_passages(), "hi", 8, rng=random.Random(0))
    assert all(p.lang == "hi" for p in picked)


# --- generation ----------------------------------------------------------------


def test_generation_fills_every_matrix_cell():
    stub = _Stub()
    matrix = dict.fromkeys(MATRIX, 3)
    items = generate_candidates(_passages(), stub, matrix=matrix)
    got = {}
    for i in items:
        got[(i.query_lang, i.passage_lang)] = got.get((i.query_lang, i.passage_lang), 0) + 1
    assert set(got) == set(MATRIX)
    assert all(v == 3 for v in got.values()), got


def test_every_generated_item_is_unverified():
    """Nothing may reach a reported metric without a human pass (PRD §6.5)."""
    items = generate_candidates(_passages(), _Stub(), matrix=dict.fromkeys(MATRIX, 2))
    assert items and all(not i.verified for i in items)
    assert all(i.annotator == "" for i in items)


def test_cross_lingual_items_keep_the_passage_in_the_other_language():
    """The question is translated; the evidence stays put. That is the whole point."""
    items = generate_candidates(_passages(), _Stub(), matrix={("en", "hi"): 3})
    assert len(items) == 3
    for item in items:
        assert item.query_lang == "en" and item.passage_lang == "hi"
        assert all("-hi#" in pid for pid in item.gold_passage_ids)


def test_monolingual_generation_makes_one_model_call_per_item():
    stub = _Stub()
    generate_candidates(_passages(), stub, matrix={("en", "en"): 4})
    assert len(stub.prompts) == 4


def test_hinglish_from_an_english_passage_takes_one_extra_model_call():
    """English question -> Hindi is a model call; Hindi -> Romanized is not.

    Romanization is deterministic transliteration (query/translit.py), so a
    hinglish-from-English item costs two model calls, not three.
    """
    stub = _Stub()
    generate_candidates(_passages(), stub, matrix={("hinglish", "en"): 2})
    assert len(stub.prompts) == 4


def test_a_passage_is_never_reused_across_items():
    items = generate_candidates(_passages(), _Stub(), matrix=dict.fromkeys(MATRIX, 3))
    ids = [pid for i in items for pid in i.gold_passage_ids]
    assert len(ids) == len(set(ids))


# --- coverage ------------------------------------------------------------------


def _item(qid, ql, pl, answerable=True, verified=True, scheme="s1", klass=None):
    return QAItem(
        id=qid,
        question="q",
        query_lang=ql,
        passage_lang=pl,
        answerable=answerable,
        gold_passage_ids=["p1"] if answerable else [],
        scheme=scheme,
        verified=verified,
        unanswerable_class=klass,
    )


def test_coverage_reports_the_gap_against_the_prd_matrix():
    items = [_item(f"q{i}", "en", "en") for i in range(10)]
    cov = coverage(items)
    assert cov.actual[("en", "en")] == 10
    assert cov.deltas()[("en", "en")] == 10 - MATRIX[("en", "en")]
    assert not cov.within(3)


def test_coverage_counts_unanswerable_classes():
    items = [_item(f"u{i}", "en", "", answerable=False, klass="near-miss") for i in range(5)]
    cov = coverage(items)
    assert cov.unanswerable_actual["near-miss"] == 5
    assert cov.unanswerable_target == UNANSWERABLE_TARGET


def test_rejected_items_are_not_counted_towards_coverage():
    items = [_item("a", "en", "en"), _item("b", "en", "en")]
    items[1].notes = "REJECTED unusable"
    assert coverage(items).actual[("en", "en")] == 1


# --- splitting -----------------------------------------------------------------


def test_split_never_includes_an_unverified_item():
    items = [_item(f"q{i}", "en", "en", verified=(i % 2 == 0)) for i in range(20)]
    dev, test = stratified_split(items, dev_size=6)
    assert len(dev) + len(test) == 10
    assert all(i.verified for i in dev + test)


def test_split_is_disjoint_and_labelled():
    items = [_item(f"q{i}", "en", "en", scheme=f"s{i % 4}") for i in range(40)]
    dev, test = stratified_split(items, dev_size=12)
    assert not ({i.id for i in dev} & {i.id for i in test})
    assert all(i.split == "dev" for i in dev)
    assert all(i.split == "test" for i in test)


def test_split_is_deterministic_under_the_same_seed():
    items = [_item(f"q{i}", "en", "en", scheme=f"s{i % 5}") for i in range(50)]
    a, _ = stratified_split(items, dev_size=15, seed=7)
    b, _ = stratified_split(items, dev_size=15, seed=7)
    assert [i.id for i in a] == [i.id for i in b]


def test_split_keeps_every_query_language_on_both_sides():
    items = [
        _item(f"q{i}", lang, "en", scheme=f"s{i % 5}")
        for lang in ("en", "hi", "hinglish")
        for i in range(20)
    ]
    dev, test = stratified_split(items, dev_size=18)
    assert {i.query_lang for i in dev} == {"en", "hi", "hinglish"}
    assert {i.query_lang for i in test} == {"en", "hi", "hinglish"}


# --- verification ---------------------------------------------------------------


def test_progress_counts_rejected_separately_from_remaining():
    items = [_item("a", "en", "en"), _item("b", "en", "en", verified=False)]
    items[1].notes = "REJECTED bad"
    p = progress_of(items)
    assert (p.verified, p.rejected, p.remaining) == (1, 1, 0)




def test_kappa_is_one_for_perfect_agreement_and_zero_for_chance():
    labels = [True, False, True, False, True, False]
    assert cohens_kappa(labels, labels) == 1.0
    # Complete disagreement on a balanced set is worse than chance.
    assert cohens_kappa(labels, [not x for x in labels]) < 0


def test_kappa_gate_rejects_weak_agreement():
    """PRD §6.5 sets 0.70 as the gate below which Module 5 cannot be trusted."""
    a = [True] * 10 + [False] * 10
    b = [True] * 7 + [False] * 3 + [False] * 7 + [True] * 3
    assert cohens_kappa(a, b) < 0.70


# --- JSON extraction (replaces GBNF grammar; see TransformersProvider) ----------


def test_json_extraction_handles_the_ways_small_models_wrap_output():
    """llama.cpp's grammar made malformed JSON unreachable; this path has no
    grammar, so extraction has to cope with prose and markdown fences."""
    from indicrag.rag.providers import extract_json_object

    obj = '{"question": "How much?", "answer": "Rs. 500", "kind": "number"}'
    assert extract_json_object(obj) == obj
    assert extract_json_object(f"Here you go:\n```json\n{obj}\n```\nHope that helps") == obj
    assert extract_json_object(f"Sure! {obj} Let me know if you need more.") == obj


def test_json_extraction_respects_braces_inside_strings():
    """A regex trips on this; brace counting must ignore braces in string values."""
    from indicrag.rag.providers import extract_json_object

    obj = '{"question": "What is {scheme}?", "answer": "a}b", "kind": "fact"}'
    assert extract_json_object(obj) == obj


def test_json_extraction_returns_none_when_there_is_no_object():
    from indicrag.rag.providers import extract_json_object

    assert extract_json_object("I cannot answer that.") is None
    assert extract_json_object("") is None
    assert extract_json_object('{"unterminated": ') is None


def test_extracted_json_round_trips_through_the_candidate_parser():
    from indicrag.rag.providers import extract_json_object

    raw = 'Here:\n```json\n{"question": "Kitna milega?", "answer": "Rs. 1000", "kind": "number"}\n```'
    cand = parse_reply(extract_json_object(raw))
    assert cand is not None and cand.answer == "Rs. 1000"


# --- unanswerable scaffolds ----------------------------------------------------


def _corpus_for_unanswerable():
    return _passages(n_schemes=8, per_scheme=4)


def test_unanswerable_hits_every_prd_class_quota():
    from indicrag.dataset.unanswerable import build_unanswerable

    items = build_unanswerable(_corpus_for_unanswerable())
    counts = {}
    for i in items:
        counts[i.unanswerable_class] = counts.get(i.unanswerable_class, 0) + 1
    assert counts == UNANSWERABLE_TARGET
    assert len(items) == 80


def test_unanswerable_language_mix_is_balanced():
    """PRD §6.3 wants roughly 36/33/31 so per-language abstention rates stay
    comparable with the answerable set. An English-heavy stem list broke this."""
    from indicrag.dataset.unanswerable import build_unanswerable

    items = build_unanswerable(_corpus_for_unanswerable())
    n = len(items)
    for lang in ("en", "hi", "hinglish"):
        share = sum(1 for i in items if i.query_lang == lang) / n
        assert 0.25 <= share <= 0.45, f"{lang} at {share:.0%}"


def test_every_unanswerable_item_is_marked_unanswerable_and_unverified():
    from indicrag.dataset.unanswerable import build_unanswerable

    items = build_unanswerable(_corpus_for_unanswerable())
    assert all(not i.answerable for i in items)
    assert all(not i.verified for i in items)
    assert all(i.gold_passage_ids == [] for i in items)
    assert all(i.unanswerable_class for i in items)


def test_near_miss_items_carry_a_distractor_for_the_annotator_to_check():
    """Absence cannot be confirmed without the passage it nearly matches."""
    from indicrag.dataset.unanswerable import build_unanswerable

    items = build_unanswerable(_corpus_for_unanswerable())
    near = [i for i in items if i.unanswerable_class == "near-miss"]
    assert near and all("distractor=" in i.notes for i in near)


def test_scheme_named_classes_spread_across_schemes():
    from indicrag.dataset.unanswerable import build_unanswerable

    items = build_unanswerable(_corpus_for_unanswerable())
    schemes = {i.scheme for i in items if i.scheme}
    assert len(schemes) >= 5, schemes


def test_out_of_scope_and_under_specified_name_no_scheme():
    from indicrag.dataset.unanswerable import build_unanswerable

    items = build_unanswerable(_corpus_for_unanswerable())
    for i in items:
        if i.unanswerable_class in ("out-of-scope", "under-specified"):
            assert i.scheme == ""
            assert "{scheme}" not in i.question


# --- script-aware fusion --------------------------------------------------------


def test_script_aware_rrf_does_not_penalise_a_cross_script_passage():
    """Plain RRF docks a passage 2:1 for the lexical retriever's blindness.

    BM25 cannot return a Devanagari passage for a Latin query at all, so its
    silence is no evidence rather than evidence against.
    """
    from indicrag.index.hybrid import rrf_fusion, script_aware_rrf
    from indicrag.models import Retrieved

    # BM25 finds two same-script passages; dense ranks the cross-script gold first.
    lexical = [
        Retrieved(passage_id="en1", score=9.0, rank=1, method="bm25"),
        Retrieved(passage_id="en2", score=8.0, rank=2, method="bm25"),
    ]
    dense = [
        Retrieved(passage_id="hi1", score=0.9, rank=1, method="dense"),
        Retrieved(passage_id="en1", score=0.8, rank=2, method="dense"),
    ]
    script_of = {"en1": "latin", "en2": "latin", "hi1": "deva"}

    plain = rrf_fusion(lexical, dense, k=3)
    assert plain[0].passage_id == "en1", "plain RRF promotes the doubly-voted passage"

    aware = script_aware_rrf(
        lexical, dense, script_of=script_of, query_script="latin", k=3
    )
    assert aware[0].passage_id == "hi1", "the cross-script passage should now win"


def test_script_aware_rrf_matches_plain_when_every_candidate_shares_the_script():
    """With no cross-script candidates the correction must be a no-op on ordering."""
    from indicrag.index.hybrid import rrf_fusion, script_aware_rrf
    from indicrag.models import Retrieved

    lexical = [Retrieved(passage_id=f"en{i}", score=9 - i, rank=i + 1, method="bm25") for i in range(3)]
    dense = [Retrieved(passage_id=f"en{i}", score=0.9 - i / 10, rank=i + 1, method="dense") for i in range(3)]
    script_of = {f"en{i}": "latin" for i in range(3)}

    plain = [r.passage_id for r in rrf_fusion(lexical, dense, k=3)]
    aware = [
        r.passage_id
        for r in script_aware_rrf(lexical, dense, script_of=script_of, query_script="latin", k=3)
    ]
    assert plain == aware


def test_mixed_script_query_treats_both_retrievers_as_eligible():
    """A query containing both scripts can be reached by BM25 either way, so no
    correction applies."""
    from indicrag.index.hybrid import rrf_fusion, script_aware_rrf
    from indicrag.models import Retrieved

    lexical = [Retrieved(passage_id="en1", score=9.0, rank=1, method="bm25")]
    dense = [Retrieved(passage_id="hi1", score=0.9, rank=1, method="dense")]
    script_of = {"en1": "latin", "hi1": "deva"}

    plain = [r.passage_id for r in rrf_fusion(lexical, dense, k=2)]
    aware = [
        r.passage_id
        for r in script_aware_rrf(lexical, dense, script_of=script_of, query_script="mixed", k=2)
    ]
    assert plain == aware


# --- generator dtype selection ---------------------------------------------------


def test_dtype_falls_back_to_bfloat16_when_float32_will_not_fit():
    """If float32 spills to the page file, throughput collapses by one to two
    orders of magnitude -- far worse than bfloat16's ~40% penalty. So the fit is
    checked rather than assumed."""
    from indicrag.rag.providers import TransformersProvider as T

    original = T.free_ram_gb
    try:
        T.free_ram_gb = classmethod(lambda cls: 4.0)
        assert T.choose_dtype() == "bfloat16"
        T.free_ram_gb = classmethod(lambda cls: 12.0)
        assert T.choose_dtype() == "float32"
    finally:
        T.free_ram_gb = original


def test_unknown_free_ram_takes_the_option_that_cannot_thrash():
    from indicrag.rag.providers import TransformersProvider as T

    original = T.free_ram_gb
    try:
        T.free_ram_gb = classmethod(lambda cls: None)
        assert T.choose_dtype() == "bfloat16"
    finally:
        T.free_ram_gb = original


def test_free_ram_probe_returns_a_plausible_value_on_this_machine():
    from indicrag.rag.providers import TransformersProvider as T

    free = T.free_ram_gb()
    assert free is None or 0.1 < free < 2048


def test_exclusion_happens_before_selection_not_after():
    """Regression: en->hi finished at 20/45 while 64 usable passages sat unused.

    Selecting count*3 candidates from the whole corpus and *then* dropping the
    already-used ones looks equivalent to excluding first and is not. By the
    third cell drawing on the same language, most of what gets selected is
    already spent, the over-selection headroom evaporates, and the cell runs out
    of candidates rather than out of corpus.
    """
    passages = _passages(n_schemes=4, per_scheme=6)
    en = [p for p in passages if p.lang == "en"]
    # Spend all but four English passages.
    spent = {p.passage_id for p in en[:-4]}
    items = generate_candidates(
        passages, _Stub(), matrix={("en", "en"): 4}, exclude=spent
    )
    assert len(items) == 4, "the four remaining passages should all be reachable"
    assert not ({pid for i in items for pid in i.gold_passage_ids} & spent)



# --- segmentation merge guard ----------------------------------------------------


def _doc(doc_id="d-en"):
    from indicrag.models import Document

    return Document(doc_id=doc_id, scheme="s", lang="en", title="t")


def test_a_short_chunk_does_not_merge_across_a_section_boundary():
    """§III-C forbids overlap across section boundaries because bleeding across
    one puts a clause from one scheme inside another's passage. The merge guard
    did exactly that: it checked doc_id only, so a short chunk from any later
    section merged into whatever passage was last -- and the result kept the
    *earlier* section's path as its metadata."""
    from indicrag.corpus.segment import segment_document

    long_a = " ".join(f"Alpha sentence {n} about eligibility criteria." for n in range(40))
    # Long enough to clear the 15-token boilerplate floor, short enough to sit
    # under MIN_TOKENS -- exactly the chunk the old guard swallowed.
    short_b = " ".join(f"Budget clause {n} names an allocation." for n in range(6))
    text = f"== Introduction ==\n{long_a}\n\n== Budget ==\n{short_b}\n"
    passages = segment_document(_doc(), text)

    budget = [p for p in passages if "Budget clause" in p.text]
    assert budget, "a short but non-trivial section must stand as its own passage"
    assert all("Alpha sentence" not in p.text for p in budget)
    assert all(p.section_path == "Budget" for p in budget)
    intro = [p for p in passages if "Alpha sentence" in p.text]
    assert all("Budget clause" not in p.text for p in intro)


def test_merging_never_pushes_a_passage_past_the_hard_maximum():
    from indicrag.corpus.segment import MAX_TOKENS, segment_document

    body = " ".join(f"Sentence number {n} carrying scheme detail." for n in range(120))
    passages = segment_document(_doc(), f"== Introduction ==\n{body}\n")
    assert passages
    assert max(p.token_count for p in passages) <= MAX_TOKENS


def test_a_passage_reports_the_section_it_actually_came_from():
    from indicrag.corpus.segment import segment_document

    text = (
        "== Introduction ==\nThe scheme was launched in 2015 to widen access.\n\n"
        "== Unified payment interface ==\nUPI processes billions of transactions.\n"
    )
    passages = segment_document(_doc(), text)
    for p in passages:
        if "UPI processes" in p.text:
            assert "Introduction" not in (p.section_path or "")
