"""Bootstrap confidence intervals and paired significance tests.

This module is the difference between "analysis" and "a demo", and it is the part
most student projects skip.

With a test set in the low hundreds, differences of two or three points are
routinely inside the noise. Reporting "dense beats BM25 by 2.1 points" without an
interval overstates the evidence, and reporting it *with* one costs nothing
because the outcomes are already in memory. Every headline number in
docs/PLAN.md §3 therefore carries a 95% interval, and every system-versus-system
claim carries a paired test.

The paired test is a randomization (permutation) test rather than a t-test: the
per-query metric values are bounded, discrete and badly non-normal -- recall@5 on
a single query with one gold passage takes values in {0, 1} -- so a test that
assumes normality is not appropriate. The permutation test assumes only
exchangeability under the null, which is exactly what "the two systems are
equivalent" means.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .retrieval import Outcome


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def width(self) -> float:
        return self.high - self.low


def bootstrap_ci(
    outcomes: Sequence[Outcome],
    metric: Callable[[Outcome], float],
    *,
    resamples: int = 1000,
    seed: int = 20260922,
    confidence: float = 0.95,
) -> Interval:
    """Percentile bootstrap over queries.

    Resampling is over *queries*, not over gold passages, because the query is the
    independent unit: two questions about the same passage are not two independent
    observations of retrieval quality.
    """
    values = [metric(o) for o in outcomes]
    if not values:
        return Interval(0.0, 0.0, 0.0)

    point = sum(values) / len(values)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * resamples) - 1)]
    hi = means[min(resamples - 1, int((1 - alpha) * resamples))]
    return Interval(point, lo, hi)


@dataclass
class PairedResult:
    delta: float
    p_value: float
    n: int

    def significant(self, level: float = 0.05) -> bool:
        return self.p_value < level

    def __str__(self) -> str:
        star = "*" if self.significant() else " "
        return f"delta={self.delta:+.3f} p={self.p_value:.4f}{star} (n={self.n})"


def paired_randomization_test(
    a: Sequence[Outcome],
    b: Sequence[Outcome],
    metric: Callable[[Outcome], float],
    *,
    trials: int = 10000,
    seed: int = 20260922,
) -> PairedResult:
    """Two-sided paired permutation test on per-query metric differences.

    Under the null the two systems are interchangeable, so for each query the
    observed pair (a_i, b_i) is as likely as (b_i, a_i). Flipping each pair
    independently at random builds the null distribution of the mean difference
    directly, with no distributional assumption.

    Pairs are matched by `item_id`, not by position: two systems may return their
    outcomes in different orders, and silently pairing by index would compare
    unrelated queries and produce a confidently wrong p-value.
    """
    by_a = {o.item_id: o for o in a}
    by_b = {o.item_id: o for o in b}
    shared = [k for k in by_a if k in by_b]
    if not shared:
        return PairedResult(0.0, 1.0, 0)

    diffs = [metric(by_a[k]) - metric(by_b[k]) for k in shared]
    observed = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    extreme = 0
    for _ in range(trials):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / len(diffs)) >= abs(observed) - 1e-12:
            extreme += 1

    # Add-one smoothing: with `trials` samples a p-value of exactly 0 is not
    # supported by the evidence, only "below 1/trials".
    p = (extreme + 1) / (trials + 1)
    return PairedResult(delta=observed, p_value=p, n=len(shared))
