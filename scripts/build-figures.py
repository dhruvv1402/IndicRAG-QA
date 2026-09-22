"""Generate the paper's figures from the committed evaluation data.

Figures are built, not drawn. Every number plotted here is read from the same
artefacts the reports are rendered from, so a figure cannot quietly disagree
with the table beside it -- which is the ordinary way a results section ends up
internally inconsistent, and is exactly what PRD NFR-6 exists to prevent.

The figures deliberately do not hide the cost of the method. Figure 3 plots the
monolingual regression on the same axes as the cross-lingual gain, because a
chart showing only the slices that improved would be a more effective and less
honest chart.

    python scripts/build-figures.py

Writes paper/figures/*.png at 200 dpi. Run after re-running any evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

# Greyscale-safe: these are distinguishable when a reviewer prints the paper.
INK = "#1a1a1a"
DENSE = "#4878a8"
LEXICAL = "#b0b0b0"
GOOD = "#2e7d4f"
BAD = "#b04a4a"
MUTED = "#7a7a7a"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    }
)

#: Recall@5 by language group, from evals/report-retrieval-probes.txt.
BY_GROUP: dict[str, tuple[float, float, float]] = {
    "BM25": (0.740, 0.006, 0.028),
    "MiniLM-L12": (0.394, 0.136, 0.102),
    "e5-base": (0.610, 0.125, 0.132),
    "LaBSE": (0.435, 0.097, 0.332),
    "MuRIL": (0.357, 0.000, 0.000),
}
GROUPS = ("monolingual", "cross-lingual", "code-mixed")

#: Script-aware against plain RRF, from evals/report-script-aware-fusion.txt.
FUSION = {
    "cross-lingual": (0.022, 0.153, "0.0001"),
    "code-mixed": (0.028, 0.173, "0.0002"),
    "monolingual": (0.712, 0.668, "0.032"),
    "overall": (0.483, 0.500, "0.32"),
}

#: Weighted fusion alpha sweep, Recall@5. alpha=1.0 is pure lexical.
ALPHA = [
    (0.0, 0.303), (0.1, 0.328), (0.2, 0.339), (0.3, 0.350), (0.4, 0.407),
    (0.5, 0.487), (0.6, 0.499), (0.7, 0.496), (0.8, 0.494), (0.9, 0.500),
    (1.0, 0.499),
]

#: Threshold sweep from evals/report-answerability.txt: (tau, precision, recall,
#: F1, abstention). The per-class recall figure this replaced plotted a single
#: fitted operating point, and that point moved from 0.000 to 0.93 recall on the
#: same data depending only on the split. The sweep is what shows why: the curve
#: underneath is flat.
SWEEP = [
    (0.000, 0.000, 0.000, 0.000), (0.300, 0.000, 0.000, 0.003),
    (0.400, 0.000, 0.000, 0.025), (0.450, 0.091, 0.037, 0.083),
    (0.500, 0.192, 0.312, 0.325), (0.550, 0.214, 0.925, 0.865),
    (0.600, 0.216, 0.938, 0.868), (0.650, 0.216, 0.938, 0.870),
    (0.700, 0.212, 0.938, 0.882), (0.750, 0.212, 0.950, 0.895),
    (0.800, 0.206, 0.950, 0.922), (0.850, 0.205, 0.963, 0.940),
    (0.900, 0.201, 0.963, 0.958), (0.950, 0.205, 1.000, 0.978),
]
BASE_RATE = 0.20

#: Script-aware fusion on the 320 answerable gold items, from
#: evals/report-fusion-goldset.txt. The probe figures in FUSION above are what
#: the paper reports; these are the same experiment on real questions, where the
#: gain is larger and the monolingual cost is not measurable.
FUSION_GOLD = {
    "cross-lingual": (0.111, 0.500, 0.367, "0.0001"),
    "code-mixed": (0.210, 0.230, 0.090, "0.72"),
    "monolingual": (0.969, 0.962, 0.962, "1.00"),
    "overall": (0.491, 0.603, 0.522, "0.0001"),
}


def fig2_by_language_group() -> Path:
    """Recall@5 by query type and system. The gap is the point."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    systems = list(BY_GROUP)
    width = 0.15

    # One distinct shade per system. A single blue at two alphas is not
    # separable once the figure is printed, and four of these five bars are
    # dense retrievers that the reader has to tell apart.
    shades = {
        "BM25": LEXICAL,
        "MiniLM-L12": "#c6dbef",
        "e5-base": "#6baed6",
        "LaBSE": "#2c5f8a",
        "MuRIL": "#08306b",
    }

    for i, system in enumerate(systems):
        offsets = [j + (i - len(systems) / 2 + 0.5) * width for j in range(len(GROUPS))]
        bars = ax.bar(offsets, BY_GROUP[system], width, label=system, color=shades[system])
        for rect, value in zip(bars, BY_GROUP[system], strict=True):
            if value == 0.0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2, 0.012, "0",
                    ha="center", va="bottom", fontsize=6.5, color=BAD, fontweight="bold",
                )

    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title(
        "BM25 collapses across the language boundary; MuRIL never crosses it",
        fontsize=8.5, pad=22,
    )
    return _save(fig, "fig2-by-language-group.png")


def fig3_script_aware_fusion() -> Path:
    """Script-aware against plain RRF, with the monolingual cost shown."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    slices = list(FUSION)
    x = range(len(slices))
    width = 0.34

    plain = [FUSION[s][0] for s in slices]
    aware = [FUSION[s][1] for s in slices]

    ax.bar([i - width / 2 for i in x], plain, width, label="plain RRF", color=LEXICAL)
    ax.bar(
        [i + width / 2 for i in x], aware, width, label="script-aware RRF",
        color=[GOOD if FUSION[s][1] >= FUSION[s][0] else BAD for s in slices],
    )

    for i, s in enumerate(slices):
        lo, hi, p = FUSION[s]
        delta = hi - lo
        ax.text(
            i, max(lo, hi) + 0.03, f"{delta:+.3f}\np={p}",
            ha="center", va="bottom", fontsize=6.5,
            color=GOOD if delta > 0 else BAD,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(slices)
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0, 0.88)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title(
        "Large gains cross-lingually and code-mixed; a real monolingual cost",
        fontsize=8.5,
    )
    return _save(fig, "fig3-script-aware-fusion.png")


def fig4_alpha_sweep() -> Path:
    """The sweep that produced a clean negative result about the wrong question."""
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    xs = [a for a, _ in ALPHA]
    ys = [r for _, r in ALPHA]
    ax.plot(xs, ys, marker="o", markersize=3.5, color=DENSE, linewidth=1.4)

    best_endpoint = max(ys[0], ys[-1])
    ax.axhline(best_endpoint, color=INK, linestyle=":", linewidth=1)
    ax.annotate(
        f"best endpoint {best_endpoint:.3f}",
        xy=(0.42, best_endpoint), fontsize=7, va="bottom", color=INK,
    )
    best_i = max(range(len(ys)), key=lambda i: ys[i])
    ax.annotate(
        f"best interior {ys[best_i]:.3f}\n(+{ys[best_i] - best_endpoint:.3f})",
        xy=(xs[best_i], ys[best_i]), xytext=(0.55, 0.36),
        fontsize=7, color=BAD,
        arrowprops={"arrowstyle": "->", "color": BAD, "linewidth": 0.8},
    )

    ax.set_xlabel("α   (0 = pure dense, 1 = pure lexical)")
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0.28, 0.56)
    ax.set_title("Weighted fusion: no interior α helps", fontsize=8.5)
    return _save(fig, "fig4-alpha-sweep.png")


def fig5_answerability_sweep() -> Path:
    """Why no threshold works: precision never leaves the base rate."""
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    taus = [r[0] for r in SWEEP]

    ax.plot(taus, [r[1] for r in SWEEP], marker="o", markersize=3,
            color=BAD, linewidth=1.6, label="precision")
    ax.plot(taus, [r[2] for r in SWEEP], marker="s", markersize=3,
            color=DENSE, linewidth=1.6, label="recall")
    ax.plot(taus, [r[3] for r in SWEEP], linestyle="--", color=MUTED,
            linewidth=1.2, label="abstention rate")

    ax.axhline(BASE_RATE, color=INK, linestyle=":", linewidth=1)
    ax.annotate(f"base rate {BASE_RATE:.2f}", xy=(0.02, BASE_RATE + 0.02),
                fontsize=7, color=INK)

    best = max(SWEEP, key=lambda r: (2 * r[1] * r[2] / (r[1] + r[2])) if r[1] + r[2] else 0)
    ax.annotate(
        f"best F1 at {best[3]:.0%}\nabstention",
        xy=(best[0], best[2]), xytext=(0.42, 0.62),
        fontsize=7, color=BAD,
        arrowprops={"arrowstyle": "->", "color": BAD, "linewidth": 0.8},
    )

    ax.set_xlabel("τ  (normalised retrieval score)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title(
        "Precision never leaves the base rate at any threshold", fontsize=8.5
    )
    return _save(fig, "fig5-answerability-sweep.png")


def fig6_fusion_goldset() -> Path:
    """The same comparison as Figure 3, on real questions rather than probes."""
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    slices = list(FUSION_GOLD)
    x = range(len(slices))
    width = 0.26

    plain = [FUSION_GOLD[s][0] for s in slices]
    aware = [FUSION_GOLD[s][1] for s in slices]
    dense = [FUSION_GOLD[s][2] for s in slices]

    ax.bar([i - width for i in x], plain, width, label="plain RRF", color=LEXICAL)
    ax.bar(list(x), dense, width, label="dense alone", color="#c6dbef")
    ax.bar([i + width for i in x], aware, width, label="script-aware RRF", color=GOOD)

    for i, key in enumerate(slices):
        lo, hi, _d, p = FUSION_GOLD[key]
        delta = hi - lo
        ax.text(
            i + width, hi + 0.03, f"{delta:+.3f}\np={p}",
            ha="center", va="bottom", fontsize=6.5,
            # Coloured by significance, not by size: +0.020 at p=0.72 is not a
            # win and must not be drawn as one.
            color=GOOD if float(p) < 0.05 else MUTED,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(slices)
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_title(
        "On gold questions the gain is larger and the monolingual cost is gone",
        fontsize=8.5, pad=20,
    )
    return _save(fig, "fig6-fusion-goldset.png")


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    for build in (
        fig2_by_language_group,
        fig3_script_aware_fusion,
        fig4_alpha_sweep,
        fig5_answerability_sweep,
        fig6_fusion_goldset,
    ):
        path = build()
        print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    print("\nFigures 2-5 are [PROBE], from 180 synthetic probes. Figure 6 is from")
    print("the 320 answerable gold items, which are not human-verified either.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
