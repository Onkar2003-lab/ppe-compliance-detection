"""X01 / S1.2: exploratory data analysis + SH17↔CHV domain-shift comparison.

Produces the figures and summary tables the Methodology and Implementation chapters need,
and the direct evidence for *why* a cross-dataset transfer gap should be expected (C7:
domain shift dominates label mismatch).

Three passes:

1. **Label pass** (fast, no pixels): per-class instance counts, instances per image,
   bbox scale + aspect ratio, class co-occurrence.
2. **Image pass** (decodes pixels at 1/8 scale): resolution spread, brightness and
   per-channel colour statistics, and a **dHash** fingerprint per image.
3. **Overlap pass**: nearest dHash between every CHV image and every SH17 image, so the
   "zero-shot" cross-dataset claim rests on pixels, not just on provenance metadata.

Box scale is reported two ways: at **native resolution** (what the annotation describes)
and at the **640 px training resolution** (what the detector actually sees), binned on the
COCO small/medium/large convention. The second is the one that predicts detection
difficulty, so it is the one Methodology should quote.

Usage::

    python src/eda.py --out D:/runs/X01-eda
    python src/eda.py --out D:/runs/X01-eda --sample 500   # quick pass while iterating

Outputs PNG figures + `summary.md` to ``--out``. No GPU, no training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.audit_labels import (
    derive_names_from_voc,
    load_roots,
    names_from_readme,
    read_yolo_rows,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

TRAIN_IMGSZ = 640
COCO_SMALL, COCO_MEDIUM = 32**2, 96**2

# Palette: categorical slots 1 and 2 of the project's validated chart palette.
# Verified with the six computable checks against a white (print) surface:
# adjacent CVD ΔE 24.7 (protan), normal-vision ΔE 33.6, both clear of the gates.
# Do not substitute ad-hoc colours: identity is fixed per dataset across every figure.
SH17_COLOUR, CHV_COLOUR = "#2a78d6", "#eb6834"
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Single-hue sequential ramp (blue, light→dark), never a rainbow for magnitude.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# The classes that carry the dissertation's story; everything else is context.
CORE_CLASSES = {"person", "helmet", "safety-vest", "vest", "head"}


def apply_style() -> None:
    """Chart chrome: thin marks, hairline recessive grid, no top/right spines."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",  # solid hairline; dashed grids read as thresholds
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def blue_cmap() -> matplotlib.colors.LinearSegmentedColormap:
    """One-hue sequential colormap for magnitude (heatmaps)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("project_blue", BLUE_RAMP)


@dataclass
class Dataset:
    """A resolved dataset: where its images and labels are, and what its classes mean."""

    name: str
    image_dir: Path
    label_dir: Path
    names: dict[int, str]
    label_files: list[Path] = field(default_factory=list)

    def image_for(self, stem: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = self.image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None


def resolve(which: str) -> Dataset:
    """Locate a dataset and recover its class names (layouts differ; see audit_labels)."""
    root = load_roots()[which]
    if which == "sh17":
        label_dir = root / "labels"
        names, _votes, _n = derive_names_from_voc(label_dir, root / "voc_labels")
        dataset = Dataset("SH17", root / "images", label_dir, names)
    else:
        inner = root / "CHV_dataset" if (root / "CHV_dataset").is_dir() else root
        label_dir = inner / "annotations"
        dataset = Dataset(
            "CHV", inner / "images", label_dir, names_from_readme(label_dir / "README.md")
        )
    dataset.label_files = sorted(dataset.label_dir.glob("*.txt"))
    return dataset


# ------------------------------------------------------------------- label pass


@dataclass
class LabelStats:
    """Everything derivable from the label files alone."""

    per_class: Counter = field(default_factory=Counter)
    per_image: list[int] = field(default_factory=list)
    rel_area: list[float] = field(default_factory=list)
    aspect: list[float] = field(default_factory=list)
    class_rel_area: dict[int, list[float]] = field(default_factory=dict)
    cooccurrence: Counter = field(default_factory=Counter)
    stems: list[str] = field(default_factory=list)


def label_pass(dataset: Dataset) -> LabelStats:
    """Collect class counts, box geometry and co-occurrence from the labels."""
    stats = LabelStats()
    for path in dataset.label_files:
        rows, _ = read_yolo_rows(path)
        stats.stems.append(path.stem)
        stats.per_image.append(len(rows))
        present = sorted({row[0] for row in rows})
        for i, a in enumerate(present):
            for b in present[i:]:
                stats.cooccurrence[(a, b)] += 1
        for cls, _xc, _yc, w, h in rows:
            stats.per_class[cls] += 1
            area = w * h
            stats.rel_area.append(area)
            stats.class_rel_area.setdefault(cls, []).append(area)
            if h > 0:
                stats.aspect.append(w / h)
    return stats


# ------------------------------------------------------------------- image pass


@dataclass
class ImageStats:
    """Everything that requires opening the image files."""

    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    brightness: list[float] = field(default_factory=list)
    channel_means: list[tuple[float, float, float]] = field(default_factory=list)
    saturation: list[float] = field(default_factory=list)
    hashes: dict[str, int] = field(default_factory=dict)


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash of a greyscale array (resized internally to 9×8)."""
    small = np.asarray(Image.fromarray(gray).resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def image_pass(dataset: Dataset, stems: list[str], sample: int | None = None) -> ImageStats:
    """Read image headers for size, then decode at reduced scale for colour + hash."""
    stats = ImageStats()
    targets = stems if sample is None else stems[:sample]
    for i, stem in enumerate(targets):
        path = dataset.image_for(stem)
        if path is None:
            continue
        try:
            with Image.open(path) as img:
                stats.widths.append(img.width)
                stats.heights.append(img.height)
                img.draft("RGB", (img.width // 8, img.height // 8))  # fast JPEG downscale
                rgb = img.convert("RGB")
                arr = np.asarray(rgb, dtype=np.uint8)
        except OSError as exc:
            logger.warning("%s: unreadable image %s (%s)", dataset.name, path.name, exc)
            continue

        stats.channel_means.append(tuple(float(v) for v in arr.reshape(-1, 3).mean(axis=0)))
        gray = arr.mean(axis=2)
        stats.brightness.append(float(gray.mean()))
        peak = arr.max(axis=2).astype(np.float32)
        trough = arr.min(axis=2).astype(np.float32)
        stats.saturation.append(float(np.mean((peak - trough) / np.maximum(peak, 1.0))))
        stats.hashes[stem] = dhash(gray.astype(np.uint8))

        if (i + 1) % 1000 == 0:
            logger.info("%s: %d/%d images read", dataset.name, i + 1, len(targets))
    return stats


def nearest_hash_distances(
    query: dict[str, int], reference: dict[str, int]
) -> list[tuple[str, int]]:
    """For each query image, the Hamming distance to its closest reference image."""
    if not query or not reference:
        return []
    ref = np.fromiter(reference.values(), dtype=np.uint64, count=len(reference))
    out: list[tuple[str, int]] = []
    for stem, value in query.items():
        xor = np.bitwise_xor(ref, np.uint64(value))
        # popcount via byte view; no np.bitwise_count on older numpy
        bits = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        out.append((stem, int(bits.min())))
    return out


# ---------------------------------------------------------------------- figures


def coco_bins(rel_areas: list[float]) -> dict[str, int]:
    """Bin boxes into COCO small/medium/large **at the 640 px training resolution**."""
    if not rel_areas:
        return {"small": 0, "medium": 0, "large": 0}
    scale = float(TRAIN_IMGSZ**2)
    counts = {"small": 0, "medium": 0, "large": 0}
    for area in rel_areas:
        pixels = area * scale
        if pixels < COCO_SMALL:
            counts["small"] += 1
        elif pixels < COCO_MEDIUM:
            counts["medium"] += 1
        else:
            counts["large"] += 1
    return counts


def save(fig: plt.Figure, out: Path, name: str) -> str:
    """Write PNG (review) + PDF (vector, for the LaTeX submission copy)."""
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("figure: %s (.png/.pdf)", out / name)
    return f"{name}.png"


def fig_class_balance(dataset: Dataset, stats: LabelStats, out: Path, colour: str) -> str:
    """Class balance with **emphasis**: the core PPE classes carry the colour.

    A uniform bar chart of 17 classes buries the finding. Colouring only the classes
    the dissertation argues about (person / helmet / vest / head) and greying the rest
    makes the support asymmetry legible at a glance, and lets the value labels be
    selective rather than a number on every bar.
    """
    items = sorted(stats.per_class.items(), key=lambda kv: -kv[1])
    labels = [dataset.names.get(cls, str(cls)) for cls, _ in items]
    values = [count for _, count in items]
    is_core = [str(name).lower() in CORE_CLASSES for name in labels]
    total = max(sum(values), 1)

    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.30 * len(labels))))
    ax.barh(
        labels[::-1],
        values[::-1],
        color=[colour if core else GRID for core in is_core[::-1]],
        height=0.72,
    )
    ax.set_xlabel("instances")
    ax.set_title(f"{dataset.name}: class balance ({total:,} instances)", loc="left")
    for y, (value, core) in enumerate(zip(values[::-1], is_core[::-1])):
        if core:  # selective direct labels: the classes the argument rests on
            ax.text(
                value,
                y,
                f"  {value:,}  ({value / total * 100:.1f}%)",
                va="center",
                fontsize=8,
                color=INK,
            )
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.20)
    return save(fig, out, f"class_balance_{dataset.name.lower()}")


def fig_domain_shift(
    sh17: tuple[LabelStats, ImageStats], chv: tuple[LabelStats, ImageStats], out: Path
) -> str:
    """The headline figure: where the two domains actually differ.

    Distributions are drawn as **step outlines, not translucent fills**: overlapping
    alpha fills produce a third colour in the overlap region that belongs to neither
    series, which is exactly where these two datasets need to be compared.
    """
    (sl, si), (cl, ci) = sh17, chv
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))

    def step_hist(ax, series, bins, xlabel: str, title: str) -> None:
        for values, name, colour in series:
            ax.hist(
                values, bins=bins, histtype="step", lw=1.8, label=name, color=colour, density=True
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend()

    step_hist(
        axes[0][0],
        ((si.brightness, "SH17", SH17_COLOUR), (ci.brightness, "CHV", CHV_COLOUR)),
        40,
        "mean pixel value (0–255)",
        "Image brightness",
    )
    step_hist(
        axes[0][1],
        ((si.saturation, "SH17", SH17_COLOUR), (ci.saturation, "CHV", CHV_COLOUR)),
        40,
        "mean saturation (0–1)",
        "Colour saturation",
    )

    ax = axes[0][2]
    width, positions = 0.38, np.arange(3)
    for offset, (stats, name, colour) in enumerate(
        ((si, "SH17", SH17_COLOUR), (ci, "CHV", CHV_COLOUR))
    ):
        means = np.array(stats.channel_means) if stats.channel_means else np.zeros((1, 3))
        ax.bar(
            positions + (offset - 0.5) * width,
            means.mean(axis=0),
            width * 0.94,  # 2px-equivalent surface gap between adjacent bars
            label=name,
            color=colour,
        )
    ax.set_xticks(positions, ["red", "green", "blue"])
    ax.set_ylabel("mean intensity (0–255)")
    ax.set_title("Colour channels", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend()

    step_hist(
        axes[1][0],
        ((sl.rel_area, "SH17", SH17_COLOUR), (cl.rel_area, "CHV", CHV_COLOUR)),
        np.logspace(-6, 0, 45),
        "box area / image area",
        "Relative box area (log scale)",
    )
    axes[1][0].set_xscale("log")

    ax = axes[1][1]
    for offset, (lab, name, colour) in enumerate(
        ((sl, "SH17", SH17_COLOUR), (cl, "CHV", CHV_COLOUR))
    ):
        bins_ = coco_bins(lab.rel_area)
        total = max(sum(bins_.values()), 1)
        ax.bar(
            positions + (offset - 0.5) * width,
            [bins_[k] / total * 100 for k in ("small", "medium", "large")],
            width * 0.94,
            label=name,
            color=colour,
        )
    ax.set_xticks(positions, ["small", "medium", "large"])
    ax.set_ylabel("% of boxes")
    ax.set_title(f"Object scale, binned at {TRAIN_IMGSZ} px", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend()

    step_hist(
        axes[1][2],
        ((sl.per_image, "SH17", SH17_COLOUR), (cl.per_image, "CHV", CHV_COLOUR)),
        range(40),
        "annotated objects",
        "Objects per image",
    )

    fig.tight_layout()
    return save(fig, out, "domain_shift")


def fig_resolution(sh17: ImageStats, chv: ImageStats, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(
        sh17.widths, sh17.heights, s=9, alpha=0.30, label="SH17", color=SH17_COLOUR, linewidths=0
    )
    ax.scatter(
        chv.widths, chv.heights, s=9, alpha=0.30, label="CHV", color=CHV_COLOUR, linewidths=0
    )
    ax.axhline(TRAIN_IMGSZ, lw=0.8, color=AXIS)
    ax.axvline(TRAIN_IMGSZ, lw=0.8, color=AXIS)
    ax.annotate(
        f"{TRAIN_IMGSZ} px training size",
        xy=(TRAIN_IMGSZ, TRAIN_IMGSZ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
    )
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Native image resolution", loc="left")
    ax.legend()
    return save(fig, out, "resolution")


def fig_cooccurrence(dataset: Dataset, stats: LabelStats, out: Path, top: int = 12) -> str:
    classes = [cls for cls, _ in stats.per_class.most_common(top)]
    matrix = np.zeros((len(classes), len(classes)))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            key = (min(a, b), max(a, b))
            matrix[i][j] = stats.cooccurrence.get(key, 0)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    image = ax.imshow(matrix, cmap=blue_cmap())  # one hue, light→dark; never a rainbow
    labels = [dataset.names.get(c, str(c)) for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), labels, fontsize=8)
    ax.set_title(f"{dataset.name}: class co-occurrence", loc="left")
    ax.grid(visible=False)
    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.set_label("images containing both classes", fontsize=8, color=INK_SECONDARY)
    bar.outline.set_visible(False)
    return save(fig, out, f"cooccurrence_{dataset.name.lower()}")


# ----------------------------------------------------------------------- report


def describe(values: list[float]) -> str:
    if not values:
        return "-"
    array = np.array(values, dtype=float)
    return (
        f"{array.mean():.1f} ± {array.std():.1f} "
        f"(p5 {np.percentile(array, 5):.1f} · p50 {np.percentile(array, 50):.1f} "
        f"· p95 {np.percentile(array, 95):.1f})"
    )


def build_report(
    sh17: Dataset,
    chv: Dataset,
    sl: LabelStats,
    si: ImageStats,
    cl: LabelStats,
    ci: ImageStats,
    overlap: list[tuple[str, int]],
    figures: list[str],
) -> str:
    sh17_bins = coco_bins(sl.rel_area)
    chv_bins = coco_bins(cl.rel_area)
    sh17_total = max(sum(sh17_bins.values()), 1)
    chv_total = max(sum(chv_bins.values()), 1)
    near = sorted(overlap, key=lambda kv: kv[1])[:5]
    identical = sum(1 for _, d in overlap if d == 0)
    very_close = sum(1 for _, d in overlap if d <= 5)

    lines = [
        "# X01 / S1.2: EDA + SH17↔CHV domain shift",
        "",
        "## Domain-shift summary (the transfer-gap evidence)",
        "",
        "| property | SH17 | CHV |",
        "|---|---|---|",
        f"| images analysed | {len(si.brightness):,} | {len(ci.brightness):,} |",
        f"| mean brightness (0–255) | {describe(si.brightness)} | {describe(ci.brightness)} |",
        (
            f"| mean saturation (0–1) | {describe([v * 100 for v in si.saturation])} /100 "
            f"| {describe([v * 100 for v in ci.saturation])} /100 |"
        ),
        (
            "| median resolution "
            f"| {int(np.median(si.widths or [0]))}×{int(np.median(si.heights or [0]))} "
            f"| {int(np.median(ci.widths or [0]))}×{int(np.median(ci.heights or [0]))} |"
        ),
        (
            f"| instances per image (mean) | {np.mean(sl.per_image or [0]):.1f} "
            f"| {np.mean(cl.per_image or [0]):.1f} |"
        ),
        (
            f"| boxes small / medium / large @{TRAIN_IMGSZ}px | "
            f"{sh17_bins['small'] / sh17_total * 100:.1f}% / "
            f"{sh17_bins['medium'] / sh17_total * 100:.1f}% / "
            f"{sh17_bins['large'] / sh17_total * 100:.1f}% | "
            f"{chv_bins['small'] / chv_total * 100:.1f}% / "
            f"{chv_bins['medium'] / chv_total * 100:.1f}% / "
            f"{chv_bins['large'] / chv_total * 100:.1f}% |"
        ),
        (
            f"| median relative box area | {np.median(sl.rel_area or [0]):.5f} "
            f"| {np.median(cl.rel_area or [0]):.5f} |"
        ),
        "",
        "## Near-duplicate check (dHash, CHV → SH17)",
        "",
        f"- CHV images fingerprinted: **{len(ci.hashes):,}** · SH17 reference set: **{len(si.hashes):,}**",
        f"- exact hash matches (distance 0): **{identical}**",
        f"- near-duplicates (distance ≤ 5 of 64 bits): **{very_close}**",
        "- closest pairs: " + (", ".join(f"`{s}` (d={d})" for s, d in near) if near else "-"),
        "",
        "_Interpretation: a 64-bit dHash distance of 0–5 indicates the same or a trivially_",
        "_re-encoded image. Anything above ~10 is a different photograph._",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{name}`" for name in figures]
    lines += [
        "",
        "## Per-class relative box area (median)",
        "",
        "| dataset | class | instances | median rel. area |",
        "|---|---|---|---|",
    ]
    for dataset, stats in ((sh17, sl), (chv, cl)):
        for cls, count in stats.per_class.most_common():
            areas = stats.class_rel_area.get(cls, [])
            lines.append(
                f"| {dataset.name} | {dataset.names.get(cls, cls)} | {count:,} "
                f"| {np.median(areas) if areas else 0:.5f} |"
            )
    return "\n".join(lines) + "\n"


def cache_path(out: Path) -> Path:
    return out / "stats-cache.json"


def save_cache(out: Path, blocks: dict[str, tuple[LabelStats, ImageStats]]) -> None:
    """Persist both passes so figures can be restyled without re-decoding images."""
    payload = {
        name: {
            "per_class": {str(k): v for k, v in lab.per_class.items()},
            "per_image": lab.per_image,
            "rel_area": lab.rel_area,
            "aspect": lab.aspect,
            "class_rel_area": {str(k): v for k, v in lab.class_rel_area.items()},
            "cooccurrence": {f"{a}|{b}": v for (a, b), v in lab.cooccurrence.items()},
            "stems": lab.stems,
            "widths": img.widths,
            "heights": img.heights,
            "brightness": img.brightness,
            "channel_means": [list(m) for m in img.channel_means],
            "saturation": img.saturation,
            "hashes": {k: str(v) for k, v in img.hashes.items()},
        }
        for name, (lab, img) in blocks.items()
    }
    cache_path(out).write_text(json.dumps(payload), encoding="utf-8")
    logger.info("stats cached: %s", cache_path(out))


def load_cache(out: Path) -> dict[str, tuple[LabelStats, ImageStats]] | None:
    """Rehydrate cached stats, or None if absent/unreadable."""
    path = cache_path(out)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cache unreadable (%s), recomputing", exc)
        return None

    blocks: dict[str, tuple[LabelStats, ImageStats]] = {}
    for name, data in payload.items():
        lab = LabelStats(
            per_class=Counter({int(k): v for k, v in data["per_class"].items()}),
            per_image=data["per_image"],
            rel_area=data["rel_area"],
            aspect=data["aspect"],
            class_rel_area={int(k): v for k, v in data["class_rel_area"].items()},
            cooccurrence=Counter(
                {tuple(int(p) for p in k.split("|")): v for k, v in data["cooccurrence"].items()}
            ),
            stems=data["stems"],
        )
        img = ImageStats(
            widths=data["widths"],
            heights=data["heights"],
            brightness=data["brightness"],
            channel_means=[tuple(m) for m in data["channel_means"]],
            saturation=data["saturation"],
            hashes={k: int(v) for k, v in data["hashes"].items()},
        )
        blocks[name] = (lab, img)
    logger.info("stats loaded from cache: %s", path)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="X01 / S1.2 EDA + domain-shift comparison.")
    parser.add_argument("--out", default="D:/runs/X01-eda", help="output directory")
    parser.add_argument("--sample", type=int, default=None, help="limit images per dataset")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the stats cache and re-read every image (slow)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    sh17, chv = resolve("sh17"), resolve("chv")
    cached = None if args.refresh else load_cache(out)
    if cached and {"sh17", "chv"} <= cached.keys():
        (sl, si), (cl, ci) = cached["sh17"], cached["chv"]
    else:
        logger.info("label pass")
        sl, cl = label_pass(sh17), label_pass(chv)
        logger.info("image pass: SH17 (%d files)", len(sl.stems))
        si = image_pass(sh17, sl.stems, args.sample)
        logger.info("image pass: CHV (%d files)", len(cl.stems))
        ci = image_pass(chv, cl.stems, args.sample)
        save_cache(out, {"sh17": (sl, si), "chv": (cl, ci)})

    logger.info("near-duplicate check")
    overlap = nearest_hash_distances(ci.hashes, si.hashes)

    figures = [
        fig_class_balance(sh17, sl, out, SH17_COLOUR),
        fig_class_balance(chv, cl, out, CHV_COLOUR),
        fig_domain_shift((sl, si), (cl, ci), out),
        fig_resolution(si, ci, out),
        fig_cooccurrence(sh17, sl, out),
        fig_cooccurrence(chv, cl, out),
    ]

    report = build_report(sh17, chv, sl, si, cl, ci, overlap, figures)
    (out / "summary.md").write_text(report, encoding="utf-8")
    # The report holds → and ±, which a cp1252 console cannot encode: without this the whole
    # job dies on its last line, after every figure has already been written.
    sys.stdout.reconfigure(errors="replace")
    print(report)
    logger.info("report written: %s", out / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
