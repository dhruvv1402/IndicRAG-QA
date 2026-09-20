"""Report formatting. Every function returns `list[str]`, never prints.

That single rule is what lets the console output and the committed
`evals/report-*.txt` files be byte-identical, and lets the formatters be tested
without capturing stdout.

Tables are fixed-width ASCII rather than markdown, because these files are read in
a terminal as often as on GitHub, and because a committed report that reflows
depending on the renderer is harder to diff between runs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ..models import Passage
from .retrieval import RetrievalReport
from .stats import Interval, bootstrap_ci, paired_randomization_test

RULE = "-" * 78


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(str(c).ljust(w)[:w] for c, w in zip(cells, widths, strict=False))


def format_corpus_stats(passages: Sequence[Passage]) -> list[str]:
    out = ["CORPUS", RULE]
    langs = Counter(p.lang for p in passages)
    schemes = Counter(p.scheme for p in passages)
    toks = sorted(p.token_count for p in passages)

    out.append(f"passages        {len(passages)}")
    out.append(f"schemes         {len(schemes)}")
    out.append("languages       " + ", ".join(f"{k}={v}" for k, v in sorted(langs.items())))
    if toks:
        n = len(toks)
        out.append(
            f"tokens          min={toks[0]} p25={toks[n // 4]} median={toks[n // 2]} "
            f"p75={toks[3 * n // 4]} max={toks[-1]}"
        )
    with_section = sum(1 for p in passages if p.section_path and p.section_path != "Introduction")
    out.append(f"section paths   {with_section}/{len(passages)} ({with_section / max(1, len(passages)):.0%})")

    out += ["", "PER SCHEME", RULE, _row(["scheme", "en", "hi", "total"], [24, 6, 6, 6])]
    en = Counter(p.scheme for p in passages if p.lang == "en")
    hi = Counter(p.scheme for p in passages if p.lang == "hi")
    for scheme in sorted(schemes):
        out.append(_row([scheme, str(en[scheme]), str(hi[scheme]), str(schemes[scheme])], [24, 6, 6, 6]))
    return out


def format_retrieval(
    reports: Sequence[RetrievalReport],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    resamples: int = 1000,
) -> list[str]:
    """The Module 1 table: every system against every k, with intervals on R@5."""
    widths = [46, 8, 8, 8, 8, 8, 8]
    out = [
        "RETRIEVAL -- overall",
        RULE,
        _row(["system", *[f"R@{k}" for k in ks], "MRR", "nDCG@10"], widths),
        RULE,
    ]
    for r in reports:
        out.append(
            _row(
                [
                    r.system,
                    *[f"{r.recall_at(k):.3f}" for k in ks],
                    f"{r.mrr():.3f}",
                    f"{r.ndcg_at(10):.3f}",
                ],
                widths,
            )
        )

    out += ["", "RETRIEVAL -- Recall@5 with 95% bootstrap CI", RULE]
    for r in reports:
        ci: Interval = bootstrap_ci(r.outcomes, lambda o: o.recall_at(5), resamples=resamples)
        out.append(f"  {r.system:44s} {ci}")

    out += [
        "",
        f"n = {reports[0].n() if reports else 0} answerable queries."
        " Intervals are percentile bootstrap over queries (1000 resamples, seed 20260922).",
        "Differences smaller than the interval width are not distinguishable from noise.",
    ]
    return out


def format_by_slice(
    reports: Sequence[RetrievalReport], *, k: int = 5, groups: Sequence[str] | None = None
) -> list[str]:
    """The Module 2 table: Recall@k broken out by language group."""
    groups = list(groups or ["monolingual", "cross-lingual", "code-mixed"])
    widths = [46] + [14] * len(groups)
    out = [
        f"RETRIEVAL -- Recall@{k} by language group",
        RULE,
        _row(["system", *groups], widths),
        RULE,
    ]
    for r in reports:
        cells = []
        for g in groups:
            n = r.n(g)
            cells.append(f"{r.recall_at(k, g):.3f} (n={n})" if n else "-")
        out.append(_row([r.system, *cells], widths))
    return out


def format_paired(
    reports: Sequence[RetrievalReport], *, k: int = 5, baseline: str | None = None
) -> list[str]:
    """Paired permutation tests against a baseline system."""
    if len(reports) < 2:
        return []
    base = next((r for r in reports if r.system == baseline), reports[0])
    out = [
        f"PAIRED TESTS -- Recall@{k} vs {base.system}",
        RULE,
        "Two-sided paired randomization test, 10000 trials. * marks p < 0.05.",
        "",
    ]
    for r in reports:
        if r is base:
            continue
        res = paired_randomization_test(r.outcomes, base.outcomes, lambda o: o.recall_at(k))
        out.append(f"  {r.system:44s} {res}")
    return out


def format_misses(report: RetrievalReport, *, k: int = 5, limit: int = 10) -> list[str]:
    out = [f"MISSES -- {report.system} (no gold passage in top {k})", RULE]
    misses = report.misses(k, limit)
    if not misses:
        out.append("  none")
        return out
    for o in misses:
        out.append(f"  [{o.item_id}] {o.slice_key:12s} gold={o.gold[:2]}")
        out.append(f"      got: {o.retrieved[:3]}")
    out.append("")
    out.append(f"  {len(report.misses(k, 10**6))} misses of {report.n()} queries")
    return out
