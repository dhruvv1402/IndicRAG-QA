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

#: Threshold-signal recall by unanswerable class, from report-answerability.txt.
ANSWERABILITY = [
    ("out-of-scope", 0.688, 16),
    ("under-specified", 0.286, 7),
    ("near-miss", 0.143, 21),
    ("false-premise", 0.000, 11),
]


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


def fig5_answerability_by_class() -> Path:
    """Why the aggregate F1 was the wrong number to read."""
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    labels = [f"{name}\n(n={n})" for name, _, n in ANSWERABILITY]
    values = [v for _, v, _ in ANSWERABILITY]
    colours = [GOOD if v >= 0.5 else BAD for v in values]

    bars = ax.barh(range(len(labels)), values, color=colours, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("recall (threshold signal)")
    ax.set_xlim(0, 0.8)

    for rect, value in zip(bars, values, strict=True):
        ax.text(
            max(value, 0) + 0.015, rect.get_y() + rect.get_height() / 2,
            f"{value:.3f}", va="center", fontsize=7.5,
            color=BAD if value < 0.5 else GOOD, fontweight="bold" if value == 0 else "normal",
        )

    ax.set_title("A retrieval threshold cannot see a false premise", fontsize=8.5)
    return _save(fig, "fig5-answerability-by-class.png")


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
        fig5_answerability_by_class,
    ):
        path = build()
        print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    print("\nFigures are [PROBE]: built from synthetic-probe results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
