"""One-page figure for the presentation: what each change bought us.
Numbers are copied from reports/results.md (normal `test` split), nothing is recomputed here.
Two panels instead of one dual-axis chart: F1 (higher is better) and private values left visible (lower is better).

Run:  python -m privylens.evaluation.timeline_figure   ->  reports/figures/improvement_timeline.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (stage name, what changed, test F1, % of private values left visible)
STAGES = [
    ("Regex\nonly",                 "patterns for emails,\nIDs, dates...",         0.383, 68.7),
    ("First neural\nmodel",         "had an embedding\ninitialisation bug",         0.211, 36.4),
    ("Bug fixed",                   "embedding start\nvalues corrected",            0.855, 5.7),
    ("4 layers\ninstead of 2",      "depth ablation:\nbiggest single gain",         0.910, 3.8),
    ("+ case\naugmentation",        "ALL-CAPS / lowercase\ntraining copies",        0.921, 3.7),
    ("Hybrid engine",               "regex + model merge,\nword snapping",          0.933, 2.7),
]
DISTILBERT = (0.956, 2.3)            # pretrained reference, 65.2M parameters

INK, MUTED, GRID, SERIES, REF = "#1d2733", "#5d6b7a", "#e3e8ee", "#2a6fdb", "#8b98a8"


def main(path="reports/figures/improvement_timeline.png"):
    xs = list(range(len(STAGES)))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.28})
    fig.patch.set_facecolor("white")

    for ax, idx, title, ref, fmt, ylim in (
        (top, 2, "Test F1  (higher is better)", DISTILBERT[0], "{:.3f}", (0, 1.08)),
        (bottom, 3, "Private values left visible, %  (lower is better)", DISTILBERT[1], "{:.1f}%", (0, 80)),
    ):
        ys = [s[idx] for s in STAGES]
        ax.plot(xs, ys, color=SERIES, lw=2, zorder=3)
        ax.plot(xs, ys, "o", color=SERIES, ms=10, mec="white", mew=2, zorder=4)           # 2px surface ring on the marks
        ax.axhline(ref, color=REF, lw=1.5, ls=(0, (5, 4)), zorder=1)
        ax.text(-0.42, ref, f"DistilBERT (pretrained, 65M params): {fmt.format(ref)}", ha="left",
                va="bottom", color=MUTED, fontsize=10)                                    # left side: the data is far away there
        for x, y in zip(xs, ys):                                                          # value labels in ink, not series colour
            below = idx == 2 and (y > 0.8 or x == 1)       # F1 panel: near the dashed line (or under a callout) -> label below the dot
            ax.annotate(fmt.format(y), (x, y), xytext=(0, -24 if below else 13), textcoords="offset points",
                        ha="center", color=INK, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, len(STAGES) - 0.5)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=10)
        ax.tick_params(axis="x", length=0)

    # stage names + what changed, under the bottom panel
    bottom.set_xticks(xs)
    bottom.set_xticklabels([s[0] for s in STAGES], color=INK, fontsize=11, fontweight="bold")
    for x, s in zip(xs, STAGES):
        bottom.annotate(s[1], (x, 0), xytext=(0, -50), textcoords="offset points", ha="center", va="top",
                        color=MUTED, fontsize=9.5)

    # the all-caps story, where it happened
    bottom.annotate("On ALL-CAPS text this step cut\nleaked values from 40.1% to 6.8%", (4, 9.5), xytext=(3.6, 33),
                    textcoords="data", ha="center", color=INK, fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=REF, lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.4", fc="#f4f6f9", ec=GRID))
    top.annotate("Worse than regex:\nthe model could not tell\nword positions apart", (1, 0.27), xytext=(1.0, 0.62),
                 textcoords="data", ha="center", color=INK, fontsize=10,
                 arrowprops=dict(arrowstyle="-", color=REF, lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f4f6f9", ec=GRID))

    fig.suptitle("PrivyLens: from rules to a 0.93 F1 privacy engine, and what each change bought", x=0.06, ha="left",
                 color=INK, fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.06, 0.925, "Our own Transformer (1.29M to 1.56M parameters), normal test set. "
             "Regex and DistilBERT shown for comparison.", color=MUTED, fontsize=10.5)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.17)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    print("saved", path)


if __name__ == "__main__":
    main()
