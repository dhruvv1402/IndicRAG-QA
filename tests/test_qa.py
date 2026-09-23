

def test_a_citation_in_the_prompts_own_notation_resolves():
    """The prompt labels passages `[1] id=<pid>`; citing back that way names a
    real passage and must not count as fabricated."""
    from indicrag.evaluation.qa_run import resolve_citation

    ctx = ["a-en#p0001", "b-hi#p0002"]
    assert resolve_citation("id=b-hi#p0002", ctx) == "b-hi#p0002"
    assert resolve_citation("[id=a-en#p0001]", ctx) == "a-en#p0001"
    assert resolve_citation("[2]", ctx) == "b-hi#p0002"
    assert resolve_citation("a-en#p0001 (a, Reception)", ctx) == "a-en#p0001"
    assert resolve_citation("id=b-hi#p0002 (b, lead),", ctx) == "b-hi#p0002"
    assert resolve_citation(" a-en#p0001 ", ctx) == "a-en#p0001"
    assert resolve_citation("[9]", ctx) == "9"  # out of range: left to fail as fabricated
    assert resolve_citation("", ctx) == ""
