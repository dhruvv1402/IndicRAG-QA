"""Grounding metrics: is the answer actually supported by the passage it cites?

H3 claims RAG reduces unsupported answers relative to a closed-book model. That
requires a measurement of "unsupported", not an assumption, and this module is
it.

**Citation Support Rate** is the headline: of the questions a system chose to
answer, the fraction whose answer is supported by the evidence it cited. Two
variants are computed because they fail differently and reporting only the
favourable one would be dishonest.

- **Lexical (ROUGE-L precision).** What share of the answer's content appears, in
  order, in the cited passage. Cheap, language-agnostic, and rewards copying --
  an extractive system scores 1.0 by construction, which is why the extractive
  provider is a useful floor rather than a competitor.
- **Entailment (NLI).** Does the passage entail the answer? Semantic, catches
  correct paraphrase that ROUGE-L misses, and brings its own error rate. Requires
  a model, so it is optional and reported when available.

Where the two disagree is itself worth reporting: high ROUGE-L with low
entailment suggests the answer stitched passage fragments into something the
passage does not actually say, which is a specific and interesting failure.

**Abstentions are excluded from the denominator.** Refusing to answer is not a
grounding failure. Counting it as one would reward a system that answers nothing,
which is precisely the degenerate behaviour the answerability metrics already
guard against -- so over-abstention is reported separately rather than being
allowed to flatter this number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..evaluation.qa import normalize_answer

#: Above this share of the answer appearing in order in the cited passage, we
#: call it lexically supported. Not a tuned value; it is a reporting threshold,
#: and the underlying continuous score is kept so the choice can be varied.
LEXICAL_SUPPORT_THRESHOLD = 0.6
ENTAILMENT_SUPPORT_THRESHOLD = 0.5


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length, O(len(a) * len(b)) space-optimised."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l_precision(answer: str, passage: str) -> float:
    """Share of the answer's tokens covered, in order, by the passage.

    Precision rather than F1: the passage is far longer than the answer, so
    recall of the passage is meaningless here. The question is whether the
    *answer* is contained in the evidence, not the reverse.
    """
    a = normalize_answer(answer).split()
    p = normalize_answer(passage).split()
    if not a:
        return 0.0
    return _lcs_length(a, p) / len(a)


@dataclass
class GroundingOutcome:
    item_id: str
    arm: str
    query_type: str = ""
    abstained: bool = False
    lexical_support: float = 0.0
    entailment: float | None = None
    #: Whether the generation named a citation at all.
    cited: bool = False
    #: Whether the named citation resolved to a real passage. Meaningless, and
    #: left True, when `cited` is False -- the two must stay separate, because
    #: collapsing them scores "declined to cite" as "cited something that does
    #: not exist", and only the second is a hallucination.
    cited_found: bool = True

    @property
    def fabricated_citation(self) -> bool:
        """Named a passage id that is not in the corpus."""
        return self.cited and not self.cited_found

    @property
    def lexically_supported(self) -> bool:
        return self.lexical_support >= LEXICAL_SUPPORT_THRESHOLD

    @property
    def entailed(self) -> bool | None:
        if self.entailment is None:
            return None
        return self.entailment >= ENTAILMENT_SUPPORT_THRESHOLD


@dataclass
class GroundingReport:
    system: str
    outcomes: list[GroundingOutcome] = field(default_factory=list)

    def _answered(self, arm: str | None = None) -> list[GroundingOutcome]:
        return [
            o for o in self.outcomes if not o.abstained and (arm is None or o.arm == arm)
        ]

    def n_answered(self, arm: str | None = None) -> int:
        return len(self._answered(arm))

    def citation_support_rate(self, arm: str | None = None) -> float:
        """Lexical variant. Abstentions excluded from the denominator."""
        sel = self._answered(arm)
        return sum(1 for o in sel if o.lexically_supported) / len(sel) if sel else 0.0

    def entailment_support_rate(self, arm: str | None = None) -> float | None:
        sel = [o for o in self._answered(arm) if o.entailment is not None]
        if not sel:
            return None
        return sum(1 for o in sel if o.entailed) / len(sel)

    def unsupported_rate(self, arm: str | None = None) -> float:
        return 1.0 - self.citation_support_rate(arm)

    def abstention_rate(self, arm: str | None = None) -> float:
        sel = [o for o in self.outcomes if arm is None or o.arm == arm]
        return sum(1 for o in sel if o.abstained) / len(sel) if sel else 0.0

    def citation_rate(self, arm: str | None = None) -> float:
        """Of answered questions, the share that named a citation at all."""
        sel = self._answered(arm)
        return sum(1 for o in sel if o.cited) / len(sel) if sel else 0.0

    def fabricated_citation_rate(self, arm: str | None = None) -> float | None:
        """Of answers that cited something, the share citing a passage that does
        not exist.

        Reported separately from Citation Support Rate because it is a different
        failure. An unsupported answer cites real evidence that does not back it;
        a fabricated citation invents the evidence. The second is worse and is
        invisible in CSR, which falls back to the retrieved context when the
        named id does not resolve and so scores the answer as though the model
        had cited honestly.

        Returns None when nothing cited, since a rate over an empty denominator
        would read as 0.0 -- indistinguishable from a system that cites
        faithfully.
        """
        sel = [o for o in self._answered(arm) if o.cited]
        return sum(1 for o in sel if o.fabricated_citation) / len(sel) if sel else None

    def mean_lexical_support(self, arm: str | None = None) -> float:
        sel = self._answered(arm)
        return sum(o.lexical_support for o in sel) / len(sel) if sel else 0.0

    def disagreements(self, arm: str | None = None) -> list[GroundingOutcome]:
        """Cases where lexical and entailment verdicts differ.

        High lexical with low entailment is the interesting direction: the answer
        reuses the passage's words without saying what the passage says.
        """
        return [
            o
            for o in self._answered(arm)
            if o.entailed is not None and o.lexically_supported != o.entailed
        ]


def score_generation(
    *,
    item_id: str,
    arm: str,
    answer: str,
    abstained: bool,
    evidence: str,
    query_type: str = "",
    entailment: float | None = None,
    cited: bool = False,
    cited_found: bool = True,
) -> GroundingOutcome:
    return GroundingOutcome(
        item_id=item_id,
        arm=arm,
        query_type=query_type,
        abstained=abstained,
        lexical_support=0.0 if abstained else rouge_l_precision(answer, evidence),
        entailment=entailment,
        cited=cited,
        cited_found=cited_found,
    )


def format_grounding(report: GroundingReport, arms: Sequence[str]) -> list[str]:
    rule = "-" * 78
    out = [
        f"GROUNDING -- {report.system}",
        rule,
        "Citation Support Rate: of questions ANSWERED, the share whose answer the",
        "cited evidence supports. Abstentions are excluded from the denominator --",
        "declining to answer is not a grounding failure.",
        "",
        f"  {'arm':<16}{'answered':>10}{'CSR (lex)':>12}{'CSR (NLI)':>12}{'abstained':>12}",
    ]
    for arm in arms:
        nli = report.entailment_support_rate(arm)
        out.append(
            f"  {arm:<16}{report.n_answered(arm):>10}"
            f"{report.citation_support_rate(arm):>12.3f}"
            f"{(f'{nli:.3f}' if nli is not None else '-'):>12}"
            f"{report.abstention_rate(arm):>12.3f}"
        )
    out += [
        "",
        "H3 is supported only if the RAG arms exceed the closed-book arm here.",
        "Note that an extractive system scores 1.0 by construction, so it is a",
        "floor for reading the generative arms rather than a competitor.",
        "",
        "CITATIONS -- did the system name its evidence, and did that id exist?",
        rule,
        "A fabricated citation is a different failure from an unsupported answer:",
        "the answer does not merely fail to follow from real evidence, it points",
        "at evidence that does not exist. CSR cannot show this -- it falls back to",
        "the retrieved context when the named id does not resolve, and so scores",
        "such an answer as though the model had cited honestly.",
        "",
        f"  {'arm':<16}{'cited':>12}{'fabricated':>14}",
    ]
    for arm in arms:
        fab = report.fabricated_citation_rate(arm)
        out.append(
            f"  {arm:<16}{report.citation_rate(arm):>12.3f}"
            f"{(f'{fab:.3f}' if fab is not None else '-'):>14}"
        )
    out += [
        "",
        "Arm A is the one to read here. It is shown no passages at all, so any id",
        "it produces is invented outright, and the rate is a direct measure of the",
        "behaviour retrieval is meant to suppress.",
    ]

    out += format_support_tests(report, arms)
    return out


def format_support_tests(report: GroundingReport, arms: Sequence[str]) -> list[str]:
    """Paired tests on citation support between adjacent arms.

    The claim that the retrieval improvement carries through to generation is
    made on *citation support*, not on token-F1 -- arm C beats arm B by 0.064
    there and by 0.003 on F1. So this is the comparison that needs the test, and
    testing only F1 would leave the sentence the results section actually writes
    unsupported.

    Paired on item id over the questions both arms answered. An abstention has
    no answer to ground, so including it would score a refusal as an
    ungrounded answer.
    """
    from .stats import paired_randomization_test

    rule = "-" * 78
    pairs = [(b, a) for a, b in zip(arms, arms[1:], strict=False)]
    if not pairs:
        return []

    out = ["", "PAIRED TESTS -- citation support, two-sided randomization", rule]
    for better, worse in pairs:
        a_out = [o for o in report.outcomes if o.arm == worse and not o.abstained]
        b_out = [o for o in report.outcomes if o.arm == better and not o.abstained]
        res = paired_randomization_test(
            b_out, a_out, lambda o: float(o.lexically_supported)
        )
        out.append(f"  arm {better} vs arm {worse}   {res}")
    out += [
        "",
        "  n is the questions BOTH arms answered: an abstention has no answer to",
        "  ground, and counting it as unsupported would penalise refusing.",
    ]
    return out
