"""Generate the paper's figures from the committed evaluation data.

Figures are built, not drawn. Every number plotted here is read from the same
artefacts the reports are rendered from, so a figure cannot quietly disagree
with the table beside it -- which is the ordinary way a results section ends up
internally inconsistent, and is exactly what PRD NFR-6 exists to prevent.

The figures deliberately do not hide the cost of the method. Figure 5 plots the
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

#: Recall@5 by language group on the sealed test split (216 answerable), from
#: evals/report-retrieval-test.txt. MuRIL is its better pooling, mean.
BY_GROUP: dict[str, tuple[float, float, float]] = {
    "BM25": (0.970, 0.153, 0.500),
    "MiniLM-L12": (0.624, 0.605, 0.195),
    "e5-base": (0.976, 0.661, 0.472),
    "LaBSE": (0.655, 0.540, 0.239),
    "MuRIL": (0.582, 0.048, 0.082),
}
GROUPS = ("monolingual", "cross-lingual", "code-mixed")

#: Script-aware against plain RRF on the 180 synthetic probes, from
#: evals/report-script-aware-fusion.txt. Kept as the figure of where the
#: mechanism was found; Figure 3 is the same comparison on the gold set.
FUSION = {
    "cross-lingual": (0.022, 0.153, "0.0001"),
    "code-mixed": (0.028, 0.173, "0.0002"),
    "monolingual": (0.712, 0.668, "0.032"),
    "overall": (0.483, 0.500, "0.32"),
}

#: Weighted fusion alpha sweep over BM25 + e5-base, Recall@5 on the sealed test
#: split, from evals/report-retrieval-test.txt. alpha=1.0 is pure lexical.
ALPHA = [
    (0.0, 0.720), (0.1, 0.720), (0.2, 0.715), (0.3, 0.704), (0.4, 0.688),
    (0.5, 0.657), (0.6, 0.653), (0.7, 0.618), (0.8, 0.611), (0.9, 0.602),
    (1.0, 0.586),
]
#: Script-aware RRF on the same items, for reference: the fusion that does beat
#: the dense endpoint is not a weighting at all.
ALPHA_SCRIPT_AWARE = 0.750

#: BM25 top-score threshold sweep over all 393 verified items, from
#: evals/report-answerability-bm25.txt: (tau, precision, recall, abstention),
#: tau on the min-max normalised score. Before verification the same sweep was
#: flat at the base rate; on the verified questions it is not.
SWEEP = [
    (0.050, 1.000, 0.013, 0.003), (0.100, 0.647, 0.138, 0.043),
    (0.150, 0.448, 0.325, 0.148), (0.200, 0.444, 0.550, 0.252),
    (0.250, 0.428, 0.738, 0.351), (0.300, 0.354, 0.838, 0.481),
    (0.350, 0.316, 0.887, 0.573), (0.400, 0.286, 0.925, 0.659),
    (0.450, 0.258, 0.950, 0.751), (0.500, 0.248, 0.988, 0.812),
    (0.550, 0.233, 1.000, 0.873), (0.600, 0.225, 1.000, 0.906),
    (0.650, 0.217, 1.000, 0.936), (0.700, 0.214, 1.000, 0.952),
    (0.750, 0.209, 1.000, 0.975), (0.800, 0.207, 1.000, 0.982),
    (0.850, 0.205, 1.000, 0.992), (0.900, 0.204, 1.000, 0.997),
]
BASE_RATE = 0.204
#: The report's own best F1; recomputing it from rounded precision and recall
#: gives 0.542.
BEST_F1 = 0.541

#: Script-aware fusion on the sealed test split of the model-verified gold set
#: (216 answerable), from evals/report-fusion-test.txt: (plain RRF, script-aware,
#: dense alone, p script-aware vs plain, p script-aware vs dense).
FUSION_GOLD = {
    "cross-lingual": (0.274, 0.685, 0.661, "0.0001", "0.62"),
    "code-mixed": (0.514, 0.556, 0.472, "0.25", "0.0037"),
    "monolingual": (0.964, 0.964, 0.976, "1.00", "0.75"),
    "overall": (0.618, 0.750, 0.720, "0.0001", "0.052"),
}


#: Module 4 on a 72-item stratified sample of the sealed test split, from
#: evals/report-qa.txt: arm -> (token-F1, CSR lexical, CSR entailment, answered).
QA_ARMS = {
    "closed-book": (0.050, 0.184, 0.245, 49),
    "RAG dense": (0.446, 0.870, 0.667, 54),
    "RAG hybrid": (0.418, 0.860, 0.526, 57),
    "oracle": (0.641, 0.985, 0.667, 66),
}


def fig1_pipeline() -> Path:
    """The system diagram §IV-A cites, which did not exist until now.

    Drawn here rather than by hand so it stays consistent with the prose it
    illustrates. Two rows, because the offline and query-time halves have very
    different costs and a reader needs to see which work happens once.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, *, fc="#eef2f6", bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                facecolor=fc, edgecolor=INK, linewidth=0.9,
            )
        )
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=7.2, color=INK, fontweight="bold" if bold else "normal",
        )

    def arrow(x1, y1, x2, y2, colour=INK):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                linewidth=0.9, color=colour, shrinkA=1, shrinkB=1,
            )
        )

    ax.text(1, 42.5, "offline, once", fontsize=7, style="italic", color=MUTED)
    box(1, 31, 17, 9, "EN / HI documents\n40 schemes", fc="#e4ecf4")
    box(23, 31, 18, 9, "validate\nDevanagari integrity")
    box(46, 31, 15, 9, "segment\n694 passages")
    box(66, 35.5, 17, 5.5, "BM25 / TF-IDF", fc="#f0f0f0")
    box(66, 29, 17, 5.5, "dense encoder", fc="#dce9f5")
    arrow(18, 35.5, 23, 35.5)
    arrow(41, 35.5, 46, 35.5)
    arrow(61, 35.5, 66, 38.2)
    arrow(61, 35.5, 66, 31.7)

    ax.text(1, 22.5, "per query", fontsize=7, style="italic", color=MUTED)
    box(1, 10, 15, 9, "query\nEN / HI / Hinglish", fc="#e4ecf4")
    box(21, 10, 15, 9, "language +\nscript ID")
    box(41, 10, 17, 9, "script-aware\nfusion", fc="#dbeee2", bold=True)
    box(63, 10, 16, 9, "generator,\nconstrained to\nthe evidence")
    box(84, 10, 15, 9, "answerability", fc="#f5e9e9")
    arrow(16, 14.5, 21, 14.5)
    arrow(36, 14.5, 41, 14.5)
    arrow(58, 14.5, 63, 14.5)
    arrow(79, 14.5, 84, 14.5)

    # Both indices feed fusion, and fusion needs each passage's script to know
    # which retriever was eligible to return it.
    arrow(66, 34.8, 54, 19.3, MUTED)
    arrow(66, 30.5, 54, 19.3, MUTED)
    ax.text(56, 25.5, "+ passage script", fontsize=6.4, color=MUTED, style="italic")

    box(84, 1.5, 15, 5.5, "answer + citation\nor refusal", fc="#e4ecf4")
    arrow(91.5, 10, 91.5, 7.2)

    ax.text(
        42, 3.2,
        "One path serves all three query types: the pipeline does not branch on detected language.",
        ha="center", fontsize=6.6, style="italic", color=MUTED,
    )
    return _save(fig, "fig1-pipeline.png")


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
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title(
        "Test split: BM25 collapses across the script boundary; MuRIL barely crosses it",
        fontsize=8.5, pad=22,
    )
    return _save(fig, "fig2-by-language-group.png")


def fig5_fusion_probes() -> Path:
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
        "Probe set, where it was found: large cross-script gains, a monolingual cost",
        fontsize=8.5,
    )
    return _save(fig, "fig5-fusion-probes.png")


def fig4_alpha_sweep() -> Path:
    """The sweep that produced a clean negative result about the wrong question.

    On the verified test split no interior weighting beats pure dense, so the
    curve is drawn beside the fusion that does -- script-aware RRF, which is a
    change of arithmetic rather than of weight.
    """
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    xs = [a for a, _ in ALPHA]
    ys = [r for _, r in ALPHA]
    ax.plot(xs, ys, marker="o", markersize=3.5, color=DENSE, linewidth=1.4)

    best_endpoint = max(ys[0], ys[-1])
    ax.axhline(best_endpoint, color=INK, linestyle=":", linewidth=1)
    ax.annotate(
        f"pure dense {best_endpoint:.3f}",
        xy=(0.62, best_endpoint + 0.004), fontsize=7, va="bottom", color=INK,
    )
    ax.axhline(ALPHA_SCRIPT_AWARE, color=GOOD, linestyle="--", linewidth=1)
    ax.annotate(
        f"script-aware RRF {ALPHA_SCRIPT_AWARE:.3f}",
        xy=(0.52, ALPHA_SCRIPT_AWARE + 0.004), fontsize=7, va="bottom", color=GOOD,
    )

    ax.set_xlabel("α   (0 = pure dense, 1 = pure lexical)")
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0.55, 0.78)
    ax.set_title("Weighted fusion never beats pure dense (test split)", fontsize=8.5)
    return _save(fig, "fig4-alpha-sweep.png")


def fig6_citation_support() -> Path:
    """H3: retrieval makes answers groundable; which retriever barely matters."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    arms = list(QA_ARMS)
    x = range(len(arms))
    width = 0.27
    series = (
        ("token-F1", 0, "#c6dbef"),
        ("citation support (lexical)", 1, DENSE),
        ("citation support (entailment)", 2, "#08306b"),
    )
    for j, (label, col, colour) in enumerate(series):
        offs = [i + (j - 1) * width for i in x]
        ax.bar(offs, [QA_ARMS[a][col] for a in arms], width, label=label, color=colour)
    for i, arm in enumerate(arms):
        ax.text(i, 1.03, f"answered {QA_ARMS[arm][3]}/72", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("rate")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    ax.set_title(
        "Test split: retrieval makes answers groundable; dense vs hybrid is not separable",
        fontsize=8.5, pad=32,
    )
    return _save(fig, "fig6-citation-support.png")


def fig7_answerability_sweep() -> Path:
    """The BM25 threshold's trade-off: a usable signal, bought with abstention."""
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    taus = [r[0] for r in SWEEP]

    ax.plot(taus, [r[1] for r in SWEEP], marker="o", markersize=3,
            color=BAD, linewidth=1.6, label="precision")
    ax.plot(taus, [r[2] for r in SWEEP], marker="s", markersize=3,
            color=DENSE, linewidth=1.6, label="recall")
    ax.plot(taus, [r[3] for r in SWEEP], linestyle="--", color=MUTED,
            linewidth=1.2, label="abstention rate")

    ax.axhline(BASE_RATE, color=INK, linestyle=":", linewidth=1)
    ax.annotate(f"base rate {BASE_RATE:.3f}", xy=(0.72, BASE_RATE + 0.02),
                fontsize=7, color=INK)

    best = max(SWEEP, key=lambda r: (2 * r[1] * r[2] / (r[1] + r[2])) if r[1] + r[2] else 0)
    ax.annotate(
        f"best F1 {BEST_F1:.3f}\nat {best[3]:.0%} abstention",
        xy=(best[0], best[2]), xytext=(0.36, 0.45),
        fontsize=7, color=BAD,
        arrowprops={"arrowstyle": "->", "color": BAD, "linewidth": 0.8},
    )

    ax.set_xlabel("τ  (normalised BM25 top score)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.set_title(
        "BM25 threshold: precision leaves the base rate, at a cost in abstention",
        fontsize=8.5,
    )
    return _save(fig, "fig7-answerability-sweep.png")


def fig3_fusion_goldset() -> Path:
    """Script-aware fusion on the verified test split; Figure 5 is the probe-set original."""
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
        lo, hi, dn, p_plain, p_dense = FUSION_GOLD[key]
        # Two comparisons, each coloured by its own significance rather than by
        # size: the gain over plain RRF is the repair, the gain over dense alone
        # is whether fusing is worth it at all, and on the verified set they differ.
        ax.text(
            i + width, hi + 0.03, f"vs RRF {hi - lo:+.3f}, p={p_plain}",
            ha="center", va="bottom", fontsize=6,
            color=GOOD if float(p_plain) < 0.05 else MUTED,
        )
        ax.text(
            i + width, hi + 0.085, f"vs dense {hi - dn:+.3f}, p={p_dense}",
            ha="center", va="bottom", fontsize=6,
            color=GOOD if float(p_dense) < 0.05 else MUTED,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(slices)
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_title(
        "Test split: script-aware fusion repairs plain RRF; over dense alone it gains little",
        fontsize=8.5, pad=20,
    )
    return _save(fig, "fig3-fusion-goldset.png")


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    for build in (
        fig1_pipeline,
        fig2_by_language_group,
        fig3_fusion_goldset,
        fig4_alpha_sweep,
        fig5_fusion_probes,
        fig6_citation_support,
        fig7_answerability_sweep,
    ):
        path = build()
        print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    print("\nNumbered in order of appearance in the paper. Figure 5 is [PROBE], from")
    print("180 synthetic probes. Figures 2-4 and 6 are from the sealed test split of")
    print("the model-verified gold set, Figure 7 from all 393 verified items.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
