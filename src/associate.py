"""X01 / S1.4c — the person↔PPE association rule (the violation axis *and* the O6 demo).

The trained label space is `{person, helmet, vest}`: `no-helmet` and `no-vest` are not
classes, because no training set annotates worn-state (X01/F2) and the one set that labels
compliance — Pictor-PPE — is evaluation-only by supervisor condition. Compliance is
therefore **derived**: a detector emits boxes, this rule decides which helmet and which vest
belong to which person, and a person with no bound helmet is a helmet violation.

That makes this module load-bearing twice over. It is what the violation-recall axis
actually measures (scored zero-shot against Pictor's per-worker `W`/`WH`/`WV`/`WHV`), and it
is what the live demo checks inside the user's zone. So it is written once, here, imported by
both, and tested on synthetic geometry where the answer is known by construction.

**Provenance.** Nath, Behzadan & Paal (2020) — the Pictor-PPE paper (C11) — bind PPE to
workers in their Approach-1 by an explicit overlap rule, `IoU(worker, PPE) > 45 %`, with a
learned decision-tree/neural-net over normalised coordinates as the accurate alternative. We
adopt the explicit-overlap option (user decision, 2026-07-27; the learned variant is a parked
lever) with one deliberate change of measure:

**Containment, not IoU.** A helmet occupies a few percent of the person that wears it, so a
correct helmet↔person pair has an IoU near 0.03 — Nath's 0.45 threshold cannot be reached by
any helmet in our harmonised data (quantified in the calibration report). Asymmetric
containment, `area(PPE ∩ person) / area(PPE)`, asks the question the geometry actually poses:
*is this helmet on this person*, not *are these two boxes the same object*.

**Ties are broken by anchor distance, not by containment.** A helmet between two heads is
fully contained in both boxes, so containment alone is exactly 1.0 twice over. PPE has an
expected place on a body — a helmet at the top, a vest across the upper torso — so ties go to
the person whose anchor is nearest, normalised by that person's diagonal so the rule is
scale-free. Bindings are then resolved greedily and exclusively: one helmet and one vest per
person, each PPE box to at most one person.

Threshold: :data:`THRESHOLD`, calibrated on SH17 + CHV **training** splits only — never on
Pictor (the evaluation set) and never on an evaluation split. Protocol and evidence:
``python -m src.associate`` and the pack ``findings/X01-S1.4c-association-rule.md``.

Usage::

    python -m src.associate               # re-run the calibration: report + figures
    python -m src.associate --threshold 0.6   # score an alternative threshold
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.audit_labels import read_yolo_rows
from src.utils.logging import get_logger

logger = get_logger(__name__)

PERSON, HELMET, VEST = 0, 1, 2
CLASS_NAMES = {PERSON: "person", HELMET: "helmet", VEST: "vest"}
REQUIRED_PPE: tuple[int, ...] = (HELMET, VEST)

# Calibrated on SH17+CHV train splits (X01/S1.4c): the 0.05-grid threshold with the best
# separation (Youden's J) between true pairs and cross-image false pairs. The curve is a broad
# plateau, so neighbouring values behave the same — but changing it changes every violation
# number, so it is a frozen constant, re-derived only by `python -m src.associate`.
THRESHOLD = 0.80

# Where a class is expected to sit on a person, as a fraction of the person box height from
# its top edge. Used only to break ties between people who both contain the PPE box.
ANCHOR_HEIGHT = {HELMET: 0.05, VEST: 0.35}

# Pictor-PPE's per-worker compliance vocabulary (Nath et al. 2020), which the violation axis
# is scored against: W = worker only, WH = +helmet, WV = +vest, WHV = both.
PICTOR_CODES = {
    (False, False): "W",
    (True, False): "WH",
    (False, True): "WV",
    (True, True): "WHV",
}

DEFAULT_HARMONISED = Path("D:/Dissertation/harmonised")
DEFAULT_OUT = Path("D:/runs/X01-associate")
CALIBRATION_SPLIT = "train"  # never an eval split: the threshold must not see test geometry
GRID = [round(0.05 * i, 2) for i in range(1, 20)]
# Thresholds scoring within this of the best separation count as equivalent, and are reported
# as the plateau: a rule that is only right at one knife-edge value would not be trustworthy.
PLATEAU_TOLERANCE = 0.01


# ------------------------------------------------------------------------------ geometry


@dataclass(frozen=True)
class Box:
    """One normalised YOLO box: class id and centre/width/height in [0, 1]."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    @property
    def corners(self) -> tuple[float, float, float, float]:
        """``(x1, y1, x2, y2)`` — left, top, right, bottom."""
        return (
            self.xc - self.w / 2,
            self.yc - self.h / 2,
            self.xc + self.w / 2,
            self.yc + self.h / 2,
        )

    @property
    def area(self) -> float:
        return self.w * self.h


def intersection_area(a: Box, b: Box) -> float:
    """Area of the overlap of two boxes (0.0 when they do not overlap)."""
    ax1, ay1, ax2, ay2 = a.corners
    bx1, by1, bx2, by2 = b.corners
    width = min(ax2, bx2) - max(ax1, bx1)
    height = min(ay2, by2) - max(ay1, by1)
    return width * height if width > 0 and height > 0 else 0.0


def containment(ppe: Box, person: Box) -> float:
    """Fraction of the *PPE* box that lies inside the person box.

    Asymmetric on purpose: a helmet fully worn by a person scores 1.0 however large that
    person is, where IoU would score ~0.03 and reject the pair.
    """
    if ppe.area <= 0:
        return 0.0
    return intersection_area(ppe, person) / ppe.area


def iou(a: Box, b: Box) -> float:
    """Symmetric intersection-over-union — kept for the comparison against Nath's rule."""
    overlap = intersection_area(a, b)
    union = a.area + b.area - overlap
    return overlap / union if union > 0 else 0.0


def anchor_distance(ppe: Box, person: Box) -> float:
    """Distance from the PPE centre to where that class belongs on this person.

    Normalised by the person's diagonal, so the tie-break behaves the same for a worker
    close to the camera and one far away. Returns ``inf`` for a degenerate person box.
    """
    diagonal = (person.w**2 + person.h**2) ** 0.5
    if diagonal <= 0:
        return float("inf")
    _x1, y1, _x2, _y2 = person.corners
    anchor_y = y1 + ANCHOR_HEIGHT.get(ppe.cls, 0.5) * person.h
    return (((ppe.xc - person.xc) ** 2 + (ppe.yc - anchor_y) ** 2) ** 0.5) / diagonal


# --------------------------------------------------------------------------- association


@dataclass
class PersonState:
    """One person and the PPE bound to them — the unit the violation axis scores."""

    index: int
    box: Box
    bound: dict[int, int] = field(default_factory=dict)  # class id -> index of the PPE box

    def wears(self, cls: int) -> bool:
        return cls in self.bound

    @property
    def violations(self) -> tuple[int, ...]:
        """Required PPE this person is missing (the non-compliance the demo alerts on)."""
        return tuple(cls for cls in REQUIRED_PPE if cls not in self.bound)

    @property
    def pictor_code(self) -> str:
        """Compliance state in Pictor-PPE's vocabulary, for zero-shot scoring."""
        return PICTOR_CODES[(self.wears(HELMET), self.wears(VEST))]


@dataclass
class Association:
    """The result of binding every PPE box in one image to a person."""

    people: list[PersonState] = field(default_factory=list)
    unbound: list[int] = field(default_factory=list)  # PPE boxes belonging to nobody

    @property
    def codes(self) -> list[str]:
        return [person.pictor_code for person in self.people]


def associate(boxes: list[Box], threshold: float = THRESHOLD) -> Association:
    """Bind each PPE box to the person wearing it.

    A pair is *eligible* when the PPE box is at least ``threshold`` contained in the person
    box. Eligible pairs are taken greedily, strongest first, with ties broken by the smaller
    normalised anchor distance; a person accepts at most one item per class and each PPE box
    is bound at most once. Anything left over is reported in :attr:`Association.unbound` —
    a helmet on the ground has no wearer, and inventing one would manufacture compliance.

    Args:
        boxes: Every box detected (or annotated) in one image, in the harmonised space.
        threshold: Minimum containment for a binding, default the calibrated
            :data:`THRESHOLD`.

    Returns:
        An :class:`Association`: one :class:`PersonState` per person box, plus the indices of
        PPE boxes that were bound to nobody.
    """
    people = [PersonState(index=i, box=box) for i, box in enumerate(boxes) if box.cls == PERSON]
    ppe = [(i, box) for i, box in enumerate(boxes) if box.cls in REQUIRED_PPE]
    by_index = {person.index: person for person in people}

    candidates = []
    for ppe_index, item in ppe:
        for person in people:
            score = containment(item, person.box)
            if score >= threshold:
                candidates.append(
                    (score, anchor_distance(item, person.box), ppe_index, person.index)
                )

    # Strongest containment first; ties to the nearest anchor; then index order so the
    # outcome is deterministic for identical geometry.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))

    taken: set[int] = set()
    for _score, _distance, ppe_index, person_index in candidates:
        if ppe_index in taken:
            continue
        person = by_index[person_index]
        cls = boxes[ppe_index].cls
        if cls in person.bound:
            continue
        person.bound[cls] = ppe_index
        taken.add(ppe_index)

    return Association(people=people, unbound=[i for i, _ in ppe if i not in taken])


def boxes_from_rows(rows: list[tuple[int, float, float, float, float]]) -> list[Box]:
    """Convert parsed YOLO rows into :class:`Box` objects."""
    return [Box(cls, xc, yc, w, h) for cls, xc, yc, w, h in rows]


# --------------------------------------------------------------------------- calibration
#
# The threshold cannot be fitted on labelled associations, because no dataset we may train on
# says which helmet belongs to which worker. Two facts make it calibratable anyway:
#
#   Positives  — in a single-person image, every PPE box belongs to that person by
#                construction. Their containment distribution is the cost of setting the
#                threshold too high: a missed binding invents a violation.
#   Negatives  — a PPE box paired with a person from a *different* image is certainly not
#                worn by them. Their binding rate is the cost of setting it too low: an
#                accidental binding invents compliance. Random pairing puts small PPE boxes
#                against arbitrarily large person boxes, so this is an upper bound on
#                nuisance binding, not a natural error rate.
#
# The threshold maximises the separation between the two (Youden's J = true-pair retention −
# false-pair binding). True-pair containment turns out to be bimodal — worn PPE sits at 1.0,
# a small tail of unworn PPE sits at 0 — so J is a broad plateau rather than a peak, and the
# plateau is reported as a sensitivity band.


@dataclass
class Pairs:
    """Containment and IoU scores for one dataset's positive and negative pairs."""

    positive: dict[int, list[float]] = field(default_factory=lambda: {HELMET: [], VEST: []})
    positive_iou: dict[int, list[float]] = field(default_factory=lambda: {HELMET: [], VEST: []})
    negative: list[float] = field(default_factory=list)
    single_person_images: int = 0
    multi_person_images: int = 0
    ambiguous_ppe: int = 0  # PPE contained in >1 person: only the tie-break separates them
    # Worst true pairs (stem, PPE box, person box), kept for visual verification: any claim
    # about why the rule leaves them unbound has to be looked at, not assumed.
    weakest: list[tuple[str, Box, Box, float]] = field(default_factory=list)
    # Images holding more helmets than person boxes — PPE whose wearer is not annotated.
    helmet_images: int = 0
    excess_helmet_images: int = 0


def split_stems(root: Path, split: str) -> list[str]:
    """Image stems of one frozen split of a harmonised dataset."""
    listing = (root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    return [Path(line).stem for line in listing if line.strip()]


def collect_pairs(root: Path, split: str, threshold: float, seed: int = 0) -> Pairs:
    """Score positive and negative person↔PPE pairs across one dataset split."""
    pairs = Pairs()
    labels_dir = root / "labels"
    images: list[list[Box]] = []

    for stem in split_stems(root, split):
        rows, _ = read_yolo_rows(labels_dir / f"{stem}.txt")
        boxes = boxes_from_rows(rows)
        people = [b for b in boxes if b.cls == PERSON]
        ppe = [b for b in boxes if b.cls in REQUIRED_PPE]

        # Counted before anything is filtered out: an image with helmets and *no* person box
        # is the strongest evidence of an unannotated wearer, and excluding it here would
        # quietly shrink the very effect this measures.
        helmets = sum(1 for b in ppe if b.cls == HELMET)
        if helmets:
            pairs.helmet_images += 1
            if helmets > len(people):
                pairs.excess_helmet_images += 1

        if not people or not ppe:
            continue
        images.append(boxes)

        if len(people) == 1:
            pairs.single_person_images += 1
            for item in ppe:
                score = containment(item, people[0])
                pairs.positive[item.cls].append(score)
                pairs.positive_iou[item.cls].append(iou(item, people[0]))
                if score < threshold:
                    pairs.weakest.append((stem, item, people[0], score))
        else:
            pairs.multi_person_images += 1
            for item in ppe:
                hits = sum(1 for person in people if containment(item, person) >= threshold)
                if hits > 1:
                    pairs.ambiguous_ppe += 1

    # Negatives: PPE from one image against a person from another, drawn deterministically.
    rng = random.Random(seed)
    people_pool = [b for boxes in images for b in boxes if b.cls == PERSON]
    for boxes in images:
        for item in (b for b in boxes if b.cls in REQUIRED_PPE):
            other = rng.choice(people_pool)
            if other in boxes:  # same image: not provably a negative, so skip it
                continue
            pairs.negative.append(containment(item, other))
    return pairs


def retention(scores: list[float], threshold: float) -> float:
    """Fraction of scores at or above ``threshold`` (1.0 for an empty list)."""
    return sum(1 for s in scores if s >= threshold) / len(scores) if scores else 1.0


def sweep(datasets: dict[str, Pairs]) -> list[dict]:
    """Retention of true pairs, binding rate of false pairs, and their separation, per τ.

    Each dataset×class series is weighted equally in the separation so the larger dataset
    and the commoner class cannot decide the threshold on their own.
    """
    table = []
    for value in GRID:
        row: dict[str, float] = {"threshold": value}
        positives, negatives = [], []
        for name, pairs in datasets.items():
            for cls in REQUIRED_PPE:
                kept = retention(pairs.positive[cls], value)
                row[f"{name}-{CLASS_NAMES[cls]}"] = kept
                positives.append(kept)
            bound = retention(pairs.negative, value)
            row[f"{name}-negative"] = bound
            negatives.append(bound)
        row["separation"] = sum(positives) / len(positives) - sum(negatives) / len(negatives)
        table.append(row)
    return table


def choose_threshold(table: list[dict]) -> tuple[float, list[float]]:
    """Best-separating grid threshold, with the plateau of values that tie with it.

    Returns:
        ``(threshold, plateau)`` — the τ with the highest separation, and every τ within
        :data:`PLATEAU_TOLERANCE` of it. A wide plateau is the point: it says the derived
        compliance state is insensitive to the exact threshold, so the violation numbers do
        not hang on one arbitrary constant.
    """
    best = max(table, key=lambda row: row["separation"])
    plateau = [
        row["threshold"]
        for row in table
        if row["separation"] >= best["separation"] - PLATEAU_TOLERANCE
    ]
    return best["threshold"], plateau


# ------------------------------------------------------------------------------- figures


def fig_score_distributions(datasets: dict[str, Pairs], chosen: float, out: Path) -> None:
    """Why containment and not IoU: the same true pairs under both measures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.eda import CHV_COLOUR, INK, INK_SECONDARY, MUTED, SH17_COLOUR, apply_style

    apply_style()
    colours = {"sh17": SH17_COLOUR, "chv": CHV_COLOUR}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    bins = [i / 40 for i in range(41)]

    for name, pairs in datasets.items():
        positives = [s for cls in REQUIRED_PPE for s in pairs.positive[cls]]
        axes[0].hist(
            positives,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=colours[name],
            label=f"{name.upper()} true pairs",
        )
        axes[1].hist(
            [s for cls in REQUIRED_PPE for s in pairs.positive_iou[cls]],
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=colours[name],
            label=f"{name.upper()} true pairs",
        )
    negatives = [s for pairs in datasets.values() for s in pairs.negative]
    axes[0].hist(
        negatives,
        bins=bins,
        histtype="stepfilled",
        color=MUTED,
        alpha=0.45,
        label="cross-image false pairs",
    )

    axes[0].axvline(chosen, color=INK, linewidth=1.2, linestyle="--")
    axes[0].annotate(
        f"chosen τ = {chosen:g}",
        (chosen, 0.92),
        xycoords=("data", "axes fraction"),
        ha="right",
        va="top",
        fontsize=8,
        color=INK,
        rotation=90,
    )
    axes[1].axvline(0.45, color=INK, linewidth=1.2, linestyle="--")
    axes[1].annotate(
        "Nath (2020) IoU > 0.45",
        (0.45, 0.92),
        xycoords=("data", "axes fraction"),
        ha="right",
        va="top",
        fontsize=8,
        color=INK,
        rotation=90,
    )

    axes[0].set_title("Containment (∩ ÷ PPE area)", fontsize=10)
    axes[1].set_title("IoU — the measure we replace", fontsize=10)
    for axis in axes:
        axis.set_xlabel("score")
        axis.set_yscale("log")
        axis.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("pairs (log)", color=INK_SECONDARY)
    fig.suptitle(
        "Person↔PPE binding: true pairs separate under containment, not under IoU",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, out, "association-score-distributions")


def fig_threshold_sweep(
    table: list[dict], datasets: dict[str, Pairs], chosen: float, out: Path
) -> None:
    """The calibration curve: what each candidate threshold keeps and what it lets through."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.eda import CHV_COLOUR, INK, INK_SECONDARY, MUTED, SH17_COLOUR, apply_style

    apply_style()
    colours = {"sh17": SH17_COLOUR, "chv": CHV_COLOUR}
    styles = {HELMET: "-", VEST: "--"}
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    x = [row["threshold"] for row in table]

    for name in datasets:
        for cls in REQUIRED_PPE:
            key = f"{name}-{CLASS_NAMES[cls]}"
            axis.plot(
                x,
                [row[key] for row in table],
                styles[cls],
                color=colours[name],
                linewidth=1.6,
                label=f"{name.upper()} {CLASS_NAMES[cls]} kept",
            )
    axis.plot(
        x,
        [sum(row[f"{n}-negative"] for n in datasets) / len(datasets) for row in table],
        ":",
        color=MUTED,
        linewidth=1.8,
        label="false pairs bound (mean)",
    )

    axis.axvline(chosen, color=INK, linewidth=1.2, linestyle="--")
    axis.annotate(
        f"τ = {chosen:g}",
        (chosen, 0.5),
        xycoords=("data", "axes fraction"),
        ha="right",
        fontsize=9,
        color=INK,
        rotation=90,
    )
    axis.set_xlabel("containment threshold τ")
    axis.set_ylabel("fraction of pairs bound", color=INK_SECONDARY)
    axis.set_ylim(0, 1.02)
    axis.legend(fontsize=8, frameon=False, loc="center left")
    axis.set_title(
        "Threshold calibrated on train splits only: true pairs kept vs false pairs bound",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, out, "association-threshold-sweep")


def fig_unbound_examples(
    datasets: dict[str, Pairs],
    harmonised: Path,
    chosen: float,
    out: Path,
    count: int = 6,
) -> None:
    """Look at the true pairs the rule refuses to bind — rule failure, or bad ground truth?

    Single-person images whose PPE barely overlaps the worker are where the rule could be
    wrong by construction, so they are rendered rather than summarised. Inspection is what
    established that they are unannotated wearers rather than mis-binding. Person box in ink,
    PPE box in the dataset's colour.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from PIL import Image

    from src.eda import CHV_COLOUR, INK, SH17_COLOUR, apply_style

    apply_style()
    colours = {"sh17": SH17_COLOUR, "chv": CHV_COLOUR}

    chosen_examples: list[tuple[str, str, Box, Box, float]] = []
    for name, pairs in datasets.items():
        ranked = sorted(pairs.weakest, key=lambda item: item[3])
        chosen_examples += [(name, *item) for item in ranked[: count // 2]]
    if not chosen_examples:
        logger.info("no sub-threshold true pairs to render")
        return

    columns = 3
    rows_n = max(1, (len(chosen_examples) + columns - 1) // columns)
    fig, axes = plt.subplots(rows_n, columns, figsize=(11, 3.6 * rows_n), squeeze=False)
    for axis, (name, stem, item, person, score) in zip(axes.flat, chosen_examples, strict=False):
        images_dir = harmonised / name / "images"
        found = next((p for p in images_dir.glob(f"{stem}.*")), None)
        if found is None:
            continue
        with Image.open(found) as image:
            picture = image.convert("RGB")
            picture.thumbnail((520, 520))
        axis.imshow(picture)
        width, height = picture.size
        for box, colour in ((person, INK), (item, colours[name])):
            x1, y1, _x2, _y2 = box.corners
            axis.add_patch(
                patches.Rectangle(
                    (x1 * width, y1 * height),
                    box.w * width,
                    box.h * height,
                    fill=False,
                    edgecolor=colour,
                    linewidth=1.6,
                )
            )
        axis.set_title(
            f"{name.upper()} {stem[:20]} — {CLASS_NAMES[item.cls]} containment {score:.2f}",
            fontsize=8,
        )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
    fig.suptitle(
        f"True pairs below τ={chosen:g}: the PPE belongs to someone with no `person` "
        "annotation (person = dark box, PPE = coloured box)",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, out, "association-unbound-examples")


def save(figure, out: Path, stem: str) -> None:
    """Write a figure as PNG (reading) and PDF (vector, for the dissertation)."""
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    logger.info("figure: %s", out / stem)
    import matplotlib.pyplot as plt

    plt.close(figure)


# -------------------------------------------------------------------------------- report


def build_report(
    datasets: dict[str, Pairs], table: list[dict], chosen: float, plateau: list[float]
) -> str:
    """Markdown record of the calibration (the run-ledger stays canonical for findings)."""
    lines = [
        "# X01 / S1.4c — person↔PPE association rule: calibration",
        "",
        (
            "Rule: bind PPE to a person when `area(PPE ∩ person) / area(PPE) >= τ`, ties to the "
            "nearest normalised anchor, one item per class per person, each item bound once. "
            "Adapted from Nath et al. (2020) Approach-1, which uses `IoU > 0.45`."
        ),
        "",
        (
            f"**Chosen τ = {chosen:g}** — best separation (true-pair retention − false-pair "
            f"binding) on the 0.05 grid. Every τ in **[{min(plateau):g}, {max(plateau):g}]** ties "
            f"within {PLATEAU_TOLERANCE:.0%}, so the derived compliance state does not hang on "
            f"the exact value. Calibrated on the `{CALIBRATION_SPLIT}` splits of SH17 + CHV "
            "only; Pictor-PPE (evaluation-only) and every evaluation split are excluded by "
            "construction."
        ),
        "",
        "## Evidence base",
        "",
        "| dataset | single-person images (positives) | multi-person images | true pairs: helmet / vest | cross-image false pairs |",
        "|---|---|---|---|---|",
    ]
    for name, pairs in datasets.items():
        lines.append(
            f"| {name.upper()} | {pairs.single_person_images} | {pairs.multi_person_images} | "
            f"{len(pairs.positive[HELMET])} / {len(pairs.positive[VEST])} | {len(pairs.negative)} |"
        )

    lines += [
        "",
        "## Containment vs IoU on the same true pairs",
        "",
        "| dataset | class | median containment | retained at τ | median IoU | retained by Nath's IoU > 0.45 |",
        "|---|---|---|---|---|---|",
    ]
    for name, pairs in datasets.items():
        for cls in REQUIRED_PPE:
            scores, ious = pairs.positive[cls], pairs.positive_iou[cls]
            if not scores:
                continue
            median = sorted(scores)[len(scores) // 2]
            median_iou = sorted(ious)[len(ious) // 2]
            lines.append(
                f"| {name.upper()} | {CLASS_NAMES[cls]} | {median:.3f} | "
                f"{retention(scores, chosen):.1%} | {median_iou:.3f} | {retention(ious, 0.45):.1%} |"
            )

    lines += [
        "",
        "## Threshold sweep",
        "",
        "| τ | " + " | ".join(k for k in table[0] if k != "threshold") + " |",
        "|" + "---|" * len(table[0]),
    ]
    for row in table:
        cells = " | ".join(f"{row[k]:.3f}" for k in row if k != "threshold")
        lines.append(f"| {row['threshold']:g} | {cells} |")

    ambiguous = {name: pairs.ambiguous_ppe for name, pairs in datasets.items()}
    lines += [
        "",
        "## Ambiguity the threshold cannot resolve",
        "",
        f"PPE boxes in multi-person images contained by more than one person at τ={chosen:g}: "
        + ", ".join(f"**{name.upper()} {count}**" for name, count in ambiguous.items())
        + ". These are decided by the anchor-distance tie-break, which is why the rule needs "
        "one; a containment threshold alone would bind them arbitrarily.",
        "",
        "## The unbound tail: PPE whose wearer is not annotated",
        "",
        (
            "True pairs falling below τ are the calibration's floor. Rendering them "
            "(`association-unbound-examples`) shows they are **not** rule failures: the PPE box "
            "sits on a person who carries no `person` annotation — SH17 labels a torso-only "
            "figure `head`/`face` rather than `person`, and CHV frames often annotate one worker "
            "of several — plus a genuine case of unworn PPE (CHV `ppe_0531`: a helmet inside a "
            "storage box). So these images contain more helmets than people by construction, and "
            "no association rule can bind them."
        ),
        "",
        "| dataset | class | true pairs below τ | share | images with more helmets than person boxes |",
        "|---|---|---|---|---|",
    ]
    for name, pairs in datasets.items():
        excess = (
            f"{pairs.excess_helmet_images} / {pairs.helmet_images} "
            f"({pairs.excess_helmet_images / pairs.helmet_images:.1%})"
            if pairs.helmet_images
            else "—"
        )
        for cls in REQUIRED_PPE:
            scores = pairs.positive[cls]
            if not scores:
                continue
            below = sum(1 for s in scores if s < chosen)
            lines.append(
                f"| {name.upper()} | {CLASS_NAMES[cls]} | {below} / {len(scores)} | "
                f"{below / len(scores):.1%} | {excess if cls == HELMET else ''} |"
            )
    lines += [
        "",
        (
            "This is a property of the *training* sets, not of the violation axis: Pictor-PPE "
            "annotates every worker with a compliance state, so the zero-shot scoring the axis "
            "actually reports is unaffected. It does mean the person class is under-supervised "
            "in training (a limitation for the Discussion), and that these labels cannot serve "
            "as ground truth for association accuracy."
        ),
    ]
    lines += ["", "_Generated by `python -m src.associate`._"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="X01/S1.4c association-rule calibration.")
    parser.add_argument("--harmonised", type=Path, default=DEFAULT_HARMONISED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default=CALIBRATION_SPLIT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="score this threshold instead of choosing one from the sweep",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the negative pairing")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    if args.split != CALIBRATION_SPLIT:
        logger.warning(
            "calibrating on the %r split: only %r keeps evaluation geometry unseen",
            args.split,
            CALIBRATION_SPLIT,
        )

    datasets: dict[str, Pairs] = {}
    for name in ("sh17", "chv"):
        root = args.harmonised / name
        if not root.is_dir():
            logger.error(
                "harmonised build missing: %s — run `python -m src.harmonise` first",
                root,
            )
            return 1
        datasets[name] = collect_pairs(root, args.split, args.threshold or THRESHOLD, args.seed)
        pairs = datasets[name]
        logger.info(
            "%s: %d single-person images, %d true pairs, %d false pairs",
            name.upper(),
            pairs.single_person_images,
            sum(len(v) for v in pairs.positive.values()),
            len(pairs.negative),
        )

    table = sweep(datasets)
    best, plateau = choose_threshold(table)
    chosen = args.threshold if args.threshold is not None else best
    logger.info(
        "chosen threshold: %.2f (plateau %.2f-%.2f; module constant THRESHOLD = %.2f)",
        chosen,
        min(plateau),
        max(plateau),
        THRESHOLD,
    )
    if args.threshold is None and abs(chosen - THRESHOLD) > 1e-9:
        logger.warning(
            "calibration disagrees with src.associate.THRESHOLD (%.2f vs %.2f) — update the "
            "constant and re-run the tests, or the rule and its evidence have drifted apart",
            chosen,
            THRESHOLD,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "associate-report.md").write_text(
        build_report(datasets, table, chosen, plateau), encoding="utf-8"
    )
    (args.out / "association-calibration.json").write_text(
        json.dumps(
            {
                "threshold": chosen,
                "plateau": [min(plateau), max(plateau)],
                "plateau_tolerance": PLATEAU_TOLERANCE,
                "calibration_split": args.split,
                "anchor_height": ANCHOR_HEIGHT,
                "counts": {
                    name: {
                        "single_person_images": p.single_person_images,
                        "multi_person_images": p.multi_person_images,
                        "positive": {CLASS_NAMES[c]: len(p.positive[c]) for c in REQUIRED_PPE},
                        "negative": len(p.negative),
                        "ambiguous_ppe": p.ambiguous_ppe,
                    }
                    for name, p in datasets.items()
                },
                "sweep": table,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("report written: %s", args.out / "associate-report.md")

    if not args.no_figures:
        fig_score_distributions(datasets, chosen, args.out)
        fig_threshold_sweep(table, datasets, chosen, args.out)
        fig_unbound_examples(datasets, args.harmonised, chosen, args.out)

    negative_rate = Counter()
    for name, pairs in datasets.items():
        negative_rate[name] = retention(pairs.negative, chosen)
    logger.info(
        "false pairs bound at tau=%.2f: %s",
        chosen,
        {k: f"{v:.2%}" for k, v in negative_rate.items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
