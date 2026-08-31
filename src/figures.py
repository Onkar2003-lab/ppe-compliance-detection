"""The S5 figure set: six figures, one visual system (fills ⚑ M2).

Two rubric criteria pay for these directly: *presentation* (10 %) and *confidence in
findings* (10 %). A confidence interval **drawn** is an argument a reader can check at a
glance; the same numbers in a table are an assertion they have to take on trust. So the
forest plot and the CD diagram are not decoration; they are how the statistics are claimed.

**The encoding choice carries the argument.** Colour is reserved for **dataset identity**
(SH17 / CHV, the project's locked categorical slots 1–2), and **architecture is encoded by
position or marker shape, never by hue**. That is deliberate: the finding is that the dataset
governs the score and the architecture barely moves it, so the figures should make the
dataset split the visually loud dimension and the model split the quiet one. It also means
the whole set runs on the two validated colours; no third hue had to be invented, and no
figure cycles a palette.

Everything else follows the project's locked chart contract (`00-Key-Facts` -> "Chart
palette", implemented in `src/eda.py`): single-hue blue ramp for magnitude and never a
rainbow, emphasis colouring with context in grey, selective direct labels rather than a
number on every mark, recessive hairline grid, and **PNG at 200 dpi plus vector PDF** for
every figure, because a figure that pixelates in the LaTeX build costs presentation marks.

Sources are the S5 artefacts, never re-derived here: `X05-accuracy-per-run.json`,
`X05-stats.json`, `X05-efficiency.json`, `X05-violation/`, and each run's `results.csv`.
A figure that recomputed its own numbers could drift from the ledger, so none of them does.

Usage::

    python -m src.figures                 # all six, PNG + PDF
    python -m src.figures --only forest   # one figure while iterating
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eda import (
    AXIS,
    CHV_COLOUR,
    GRID,
    INK,
    INK_SECONDARY,
    MUTED,
    SH17_COLOUR,
    apply_style,
    blue_cmap,
    save,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

RUNS = Path("D:/runs")
ACCURACY = RUNS / "X05-accuracy/X05-accuracy-per-run.json"
STATS = RUNS / "X05-stats/X05-stats.json"
EFFICIENCY = RUNS / "X05-efficiency/X05-efficiency.json"
VIOLATION = RUNS / "X05-violation"
DEFAULT_OUT = RUNS / "X05-figures"

DATASET_COLOUR = {"sh17": SH17_COLOUR, "chv": CHV_COLOUR}
DATASET_LABEL = {"sh17": "SH17", "chv": "CHV"}
# Architecture is encoded by shape, never by hue; see the module docstring.
ARCH_MARKER = {"y8n": "o", "y11n": "s", "y26n": "^"}
ARCH_LABEL = {"y8n": "YOLOv8n", "y11n": "YOLO11n", "y26n": "YOLO26n"}
ARCH_ORDER = ["y8n", "y11n", "y26n"]


def load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy_rows() -> list[dict]:
    rows = load(ACCURACY)
    for r in rows:
        r["architecture"] = r["run_id"].split("-")[1]
    return rows


# ------------------------------------------------------------------- 1 · transfer grid


def fig_transfer_grid(rows: list[dict], out: Path) -> str:
    """The headline finding in one image: rows train, columns test, one panel per model.

    A heatmap rather than a table because the pattern *is* the point: the two columns
    differ far more than the three panels do, which is the dataset-over-model argument made
    visible before a single number is read.
    """
    cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        trained, arch = r["trained_on"], r["architecture"]
        other = "chv" if trained == "sh17" else "sh17"
        cells[(arch, trained, trained)].append(r["in_domain"]["map50"])
        cells[(arch, trained, other)].append(r["cross_domain"]["map50"])

    order = ["sh17", "chv"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), constrained_layout=True)
    grids = []
    for arch in ARCH_ORDER:
        grids.append(np.array([[np.mean(cells[(arch, tr, te)]) for te in order] for tr in order]))
    vmin, vmax = min(g.min() for g in grids), max(g.max() for g in grids)

    for ax, arch, grid in zip(axes, ARCH_ORDER, grids):
        image = ax.imshow(grid, cmap=blue_cmap(), vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(ARCH_LABEL[arch], pad=6)
        ax.set_xticks(range(2), [DATASET_LABEL[d] for d in order])
        ax.set_yticks(range(2), [DATASET_LABEL[d] for d in order])
        ax.set_xlabel("tested on")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                value = grid[i, j]
                # Label ink flips on the dark end of the ramp so it stays legible.
                shade = "white" if (value - vmin) / (vmax - vmin + 1e-9) > 0.55 else INK
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=shade, fontsize=9)
    axes[0].set_ylabel("trained on")
    fig.colorbar(image, ax=axes, shrink=0.82, label="mAP@0.5 (test split)")
    fig.suptitle(
        "Cross-dataset transfer grid: the dataset moves the score, the architecture does not",
        fontsize=10,
        color=INK,
    )
    return save(fig, out, "X05-fig1-transfer-grid")


# ---------------------------------------------------------------------- 2 · forest plot


def fig_forest(stats: dict, out: Path) -> str:
    """Per-model means with 95 % BCa intervals: overlapping bars do the arguing.

    Grouped by dataset because pooling the two would produce a bimodal mean that describes
    the pooling rather than the model (the stats module says so explicitly).
    """
    metric = "map50_in (accuracy, in-domain)"
    cells = stats[metric]["per_cell"]

    entries = []
    for dataset in ("chv", "sh17"):
        for arch in ARCH_ORDER:
            key = f"{arch} | trained on {dataset}"
            if key in cells:
                entries.append((dataset, arch, cells[key]))

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    positions = list(range(len(entries)))[::-1]
    for y, (dataset, arch, interval) in zip(positions, entries):
        colour = DATASET_COLOUR[dataset]
        ax.plot(
            [interval["ci95_low"], interval["ci95_high"]],
            [y, y],
            color=colour,
            lw=2.0,
            solid_capstyle="round",
        )
        ax.plot(
            interval["mean"],
            y,
            marker="o",
            ms=8,
            color="white",
            markeredgecolor=colour,
            markeredgewidth=2.0,
            zorder=3,
        )
        ax.text(
            interval["ci95_high"] + 0.004,
            y,
            f"{interval['mean']:.3f}",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )

    ax.set_yticks(positions, [f"{ARCH_LABEL[a]}  ·  {DATASET_LABEL[d]}" for d, a, _ in entries])
    ax.set_xlabel("in-domain mAP@0.5 (test split), mean with 95 % BCa CI")
    # Deliberately claims the gap, not universal overlap: y11n and y26n on CHV do NOT
    # overlap, so "the intervals overlap" would be false for that pair.
    ax.set_title(
        "The gap between datasets dwarfs every difference between architectures",
        fontsize=10,
    )
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], color=DATASET_COLOUR[d], lw=2.0, label=f"trained on {DATASET_LABEL[d]}")
        for d in ("chv", "sh17")
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    return save(fig, out, "X05-fig2-forest-plot")


# ----------------------------------------------------------------------- 3 · CD diagram


def _holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the original order.

    The five omnibus Friedman tests are one family of hypotheses (does the architecture
    move *any* axis?), so their p-values are corrected for multiple comparisons before a
    single axis is called significant. Without this, running five tests inflates the chance
    that one crosses 0.05 by luck alone, which is exactly what happens here to the in-domain
    axis (raw 0.0302, adjusted 0.151). See Results Table 6.3.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))  # monotone non-decreasing
        adj[idx] = running
    return adj


def fig_cd_diagram(stats: dict, out: Path) -> str:
    """Critical-difference diagram (Friedman/Nemenyi) for every metric tested.

    The standard object for comparing several models over several conditions. Models joined
    by a bar are *not* separated; the reader sees the null directly rather than being told
    about it. The omnibus Friedman p is Holm-corrected across the five metrics, and the
    verdict (and the bars) follow the corrected result: because no omnibus survives the
    correction, no pairwise separation is interpreted and all three models are joined on
    every panel. The raw p is still printed beside the adjusted one for transparency.
    """
    metrics = [
        (label, block) for label, block in stats.items() if "error" not in block["friedman_nemenyi"]
    ]
    raw_p = [block["friedman_nemenyi"]["p_value"] for _, block in metrics]
    adj_p = _holm(raw_p)

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(7.0, 1.35 * len(metrics) + 0.8), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, (label, block), praw, padj in zip(axes, metrics, raw_p, adj_p):
        friedman = block["friedman_nemenyi"]
        ranks = friedman["mean_ranks"]
        cd = friedman.get("critical_difference")
        low, high = 1.0, float(len(ranks))
        corrected_sig = padj < 0.05

        ax.set_xlim(low - 0.25, high + 0.25)
        ax.set_ylim(-1.75, 0.85)
        ax.axis("off")
        ax.plot([low, high], [0, 0], color=AXIS, lw=1.0)
        for tick in np.arange(low, high + 0.01, 0.5):
            ax.plot([tick, tick], [0, 0.09], color=AXIS, lw=0.8)
            ax.text(tick, 0.16, f"{tick:g}", ha="center", fontsize=7, color=MUTED)

        # Labels are staggered downward whenever two mean ranks sit close enough for their
        # text to collide; tied ranks would otherwise print exactly on top of each other,
        # which is how the first draft rendered "YOLOv8n" and "YOLO11n" as one smudge.
        collision_width = (high - low) * 0.22
        # One step must exceed the text height, or a "stagger" still overprints: at 8 pt in
        # a panel this tall, one data unit is ~0.5 in, so a line of text is ~0.21 units.
        step = 0.30
        depth, deepest, previous = 0, 0, None
        for name, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
            depth = depth + 1 if previous is not None and rank - previous < collision_width else 0
            deepest = max(deepest, depth)
            stem = -0.20 - step * depth
            ax.plot([rank, rank], [0, stem], color=INK_SECONDARY, lw=1.0)
            ax.text(
                rank,
                stem - 0.04,
                ARCH_LABEL.get(name, name),
                ha="center",
                va="top",
                fontsize=8,
                color=INK,
                # Exactly tied ranks share an x, so a deeper model's stem runs straight
                # through the label above it; the knockout keeps the text readable.
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
            )
            previous = rank

        # Below the DEEPEST label, not the last one drawn: a tie early in the rank order
        # leaves `depth` back at 0, which put the bar through a label.
        ordered = sorted(ranks.items(), key=lambda kv: kv[1])
        level = -0.20 - step * deepest - 0.38
        if corrected_sig and cd:
            # A surviving omnibus: draw the standard Nemenyi bars spanning models whose
            # mean ranks sit within the critical difference.
            drawn: list[tuple[float, float]] = []
            for i in range(len(ordered)):
                j = i
                while j + 1 < len(ordered) and ordered[j + 1][1] - ordered[i][1] <= cd:
                    j += 1
                if j > i and not any(a <= ordered[i][1] and ordered[j][1] <= b for a, b in drawn):
                    ax.plot(
                        [ordered[i][1] - 0.03, ordered[j][1] + 0.03],
                        [level, level],
                        color=INK,
                        lw=3.0,
                        solid_capstyle="round",
                    )
                    drawn.append((ordered[i][1], ordered[j][1]))
                    level -= 0.20
        else:
            # No omnibus survives the multi-metric correction, so no pairwise separation is
            # interpreted: all three models are joined by one bar.
            ax.plot(
                [ordered[0][1] - 0.03, ordered[-1][1] + 0.03],
                [level, level],
                color=INK,
                lw=3.0,
                solid_capstyle="round",
            )
        if cd:
            ax.text(
                high + 0.20, -0.28, f"CD = {cd:g}", ha="right", va="center", fontsize=7, color=MUTED
            )

        verdict = "separated" if corrected_sig else "no difference detected"
        ax.set_title(
            f"{label}:  Friedman p = {praw:.4f} → Holm p = {padj:.3f} ({verdict})",
            fontsize=9,
            loc="left",
            color=INK,
        )

    fig.suptitle(
        "Critical-difference diagrams (Friedman p Holm-corrected across the five metrics): "
        "models joined by a bar are not distinguishable",
        fontsize=9.5,
        color=INK,
    )
    return save(fig, out, "X05-fig3-cd-diagram")


# ------------------------------------------------------------- 4 · efficiency / accuracy


def fig_efficiency_accuracy(rows: list[dict], efficiency: dict, out: Path) -> str:
    """The deployment trade-off, and the surprise in it (F29).

    Speed is a property of the architecture, accuracy a property of the pairing, so each
    architecture appears twice, once per training set, joined by a hairline to make the
    vertical dataset gap read as one object rather than six loose points.
    """
    fps: dict[str, float] = {}
    for run_id, timing in efficiency["timings"]["gpu"].items():
        fps.setdefault(run_id.split("-")[1], []).append(timing["fps_mean"])
    fps = {arch: float(np.mean(values)) for arch, values in fps.items()}

    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        scores[(r["architecture"], r["trained_on"])].append(r["in_domain"]["map50"])

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for arch in ARCH_ORDER:
        pair = [(fps[arch], float(np.mean(scores[(arch, d)]))) for d in ("chv", "sh17")]
        ax.plot(
            [p[0] for p in pair],
            [p[1] for p in pair],
            color=GRID,
            lw=1.0,
            zorder=1,
        )
        for (x, y), dataset in zip(pair, ("chv", "sh17")):
            ax.scatter(
                x,
                y,
                marker=ARCH_MARKER[arch],
                s=95,
                facecolor=DATASET_COLOUR[dataset],
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
        # Labelled below the lower (SH17) point: the CHV points sit near the top of the
        # axes, where a label above them collides with the chart title.
        ax.annotate(
            ARCH_LABEL[arch],
            (pair[1][0], pair[1][1]),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            va="top",
            fontsize=8,
            color=INK_SECONDARY,
        )

    ax.set_xlabel("inference speed (FPS, batch 1, GPU, end-to-end)")
    ax.set_ylabel("in-domain mAP@0.5 (test split)")
    ax.set_title("The oldest architecture is both the fastest and no less accurate", fontsize=10)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            markerfacecolor=DATASET_COLOUR[d],
            markeredgecolor="white",
            label=f"trained on {DATASET_LABEL[d]}",
        )
        for d in ("chv", "sh17")
    ] + [
        plt.Line2D(
            [],
            [],
            marker=ARCH_MARKER[a],
            ls="",
            ms=8,
            markerfacecolor=MUTED,
            markeredgecolor="white",
            label=ARCH_LABEL[a],
        )
        for a in ARCH_ORDER
    ]
    ax.legend(handles=handles, fontsize=8, loc="center right", ncol=1)
    fig.tight_layout()
    return save(fig, out, "X05-fig4-efficiency-accuracy")


# ------------------------------------------------------------------ 5 · convergence (M2)


def fig_convergence(out: Path) -> str:
    """Validation curves behind the stopping rule (⚑ M2).

    The supervisor's approval of an epoch cap was made **conditional** on documenting
    convergence clearly, so this figure discharges a commitment rather than illustrating
    one. Each run's chosen epoch is marked, which is what makes the patience rule auditable
    rather than merely described.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True, constrained_layout=True)
    panels = {"sh17": axes[0], "chv": axes[1]}
    counts = dict.fromkeys(panels, 0)

    for directory in sorted(RUNS.glob("X04-*")):
        results = directory / "results.csv"
        if not results.is_file():
            continue
        trained_on = directory.name.split("-")[-1]
        ax = panels.get(trained_on)
        if ax is None:
            continue

        rows = [line.split(",") for line in results.read_text(encoding="utf-8").splitlines()]
        header, body = rows[0], [r for r in rows[1:] if len(r) == len(rows[0])]
        epoch_i, map_i = header.index("epoch"), header.index("metrics/mAP50(B)")
        epochs = [float(r[epoch_i]) for r in body]
        values = [float(r[map_i]) for r in body]
        if not epochs:
            continue

        ax.plot(epochs, values, color=DATASET_COLOUR[trained_on], lw=1.0, alpha=0.75)
        best = int(np.argmax(values))
        ax.plot(
            epochs[best],
            values[best],
            marker="o",
            ms=3.5,
            color="white",
            markeredgecolor=DATASET_COLOUR[trained_on],
            markeredgewidth=1.2,
            zorder=3,
        )
        counts[trained_on] += 1

    for dataset, ax in panels.items():
        ax.set_title(f"trained on {DATASET_LABEL[dataset]}  ({counts[dataset]} runs)", fontsize=9)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("validation mAP@0.5")
    fig.suptitle(
        "Convergence and the stopping rule: every marked point is the epoch scored (best.pt)",
        fontsize=10,
        color=INK,
    )
    return save(fig, out, "X05-fig5-convergence")


# ------------------------------------------------------------------------ 6 · per-class


def fig_per_class(rows: list[dict], out: Path) -> str:
    """Where the score is actually lost: vest on SH17 (thin support, F1)."""
    classes = ["person", "helmet", "vest"]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        for name in classes:
            grouped[(r["trained_on"], name)].append(r["in_domain"]["per_class_map50"][name])

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    width, x = 0.36, np.arange(len(classes))
    for offset, dataset in ((-width / 2, "chv"), (width / 2, "sh17")):
        means = [float(np.mean(grouped[(dataset, c)])) for c in classes]
        lows = [float(np.min(grouped[(dataset, c)])) for c in classes]
        highs = [float(np.max(grouped[(dataset, c)])) for c in classes]
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=f"trained on {DATASET_LABEL[dataset]}",
            facecolor="white",
            edgecolor=DATASET_COLOUR[dataset],
            linewidth=1.6,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=[np.subtract(means, lows), np.subtract(highs, means)],
            fmt="none",
            ecolor=DATASET_COLOUR[dataset],
            elinewidth=1.2,
            capsize=3,
        )
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                fontsize=8,
                color=INK_SECONDARY,
            )

    ax.set_xticks(x, classes)
    ax.set_ylabel("in-domain mAP@0.5 (test split)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Per-class accuracy: SH17's vest class is where the score collapses",
        fontsize=10,
        pad=26,
    )
    # Above the axes: every bar reaches at least 0.45, so no in-plot corner is actually free.
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, borderaxespad=0)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, out, "X05-fig6-per-class")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the S5 figure set (PNG + vector PDF).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only", default=None, help="transfer|forest|cd|efficiency|convergence|per-class"
    )
    args = parser.parse_args()

    apply_style()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = accuracy_rows()
    stats = load(STATS)
    efficiency = load(EFFICIENCY)

    builders = {
        "transfer": lambda: fig_transfer_grid(rows, args.out),
        "forest": lambda: fig_forest(stats, args.out),
        "cd": lambda: fig_cd_diagram(stats, args.out),
        "efficiency": lambda: fig_efficiency_accuracy(rows, efficiency, args.out),
        "convergence": lambda: fig_convergence(args.out),
        "per-class": lambda: fig_per_class(rows, args.out),
    }
    if args.only:
        if args.only not in builders:
            logger.error("unknown figure %r; choose from %s", args.only, ", ".join(builders))
            return 1
        builders = {args.only: builders[args.only]}

    for name, build in builders.items():
        logger.info("building %s", name)
        build()
    logger.info("%d figures written to %s (PNG 200 dpi + vector PDF)", len(builders), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
