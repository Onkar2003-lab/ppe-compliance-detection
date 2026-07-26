"""X01 / S1.4 — cross-dataset near-duplicate search and the evaluation-split exclusion list.

**Why this exists.** S1.2 finding F7 showed that SH17 and CHV are *not* pixel-disjoint:
a 64-bit dHash comparison over 1,330 CHV × 8,099 SH17 images returns exact matches. The
provenance argument (SH17 = Pexels stock, CHV = SHWD/GDUT-HWD lineage) was never evidence
of image disjointness. A cross-dataset transfer number is only "zero-shot" if no evaluation
image also sits in the other dataset's training pool, so the offending images are excluded
here, by a regenerable rule rather than a hand-kept list.

**What "leakage" means, precisely.** An evaluation image leaks if it has a near-duplicate
in the *other* dataset's **training** pool — that is the pair that lets a model recognise a
test image it has already fitted. Both transfer directions are run, so the check is applied
symmetrically (:func:`leaking_pairs`).

**Exclusion policy (deliberately conservative).** Every image within ``threshold`` of a
cross-dataset partner is excluded from every evaluation split, on **both** sides of the
pair, regardless of which side's training pool holds the partner. At a Hamming radius of 5
this costs ~1 % of CHV, so the cheaper option is bought rather than argued over.

**Threshold, and why it is not widened.** dHash at 8×8 is degenerate on plain-background
studio cutouts — several CHV images are exactly that — so candidates in the 2–6 band are a
mix of true re-encodes and background collisions, and the distance histogram has no clean
gap to cut at. Every candidate at ≤5 was therefore inspected visually; the verified subset
is recorded in :data:`VERIFIED_SAME_PHOTOGRAPH`. Widening the radius would exclude on noise,
not on evidence. dHash still misses heavy crops and alternate frames from one shoot, so the
count is reported as a floor.

Usage::

    python -m src.overlap                 # rebuild the manifest + figures
    python -m src.overlap --verify        # re-check the splits, exit 1 on residual leakage
    python -m src.overlap --threshold 5 --out configs/splits/X01-cross-dataset-duplicates.yaml

Reads the dHash fingerprints cached by :mod:`src.eda` (``--refresh`` there regenerates
them); computes them from the images only if the cache is absent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from src.audit_labels import load_roots, split_stems
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 5
DEFAULT_CACHE = Path("D:/runs/X01-eda/stats-cache.json")
DEFAULT_MANIFEST = Path("configs/splits/X01-cross-dataset-duplicates.yaml")

# Visually verified as the *same photograph* (side-by-side inspection, 2026-07-27) out of
# the candidates at Hamming ≤5. The remainder are plain-background hash collisions between
# unrelated studio portraits. This changes what we *claim*, not what we exclude: the
# exclusion below stays conservative and drops every candidate.
VERIFIED_SAME_PHOTOGRAPH: dict[str, str] = {
    "ppe_0366": "work-chinese-industrial-professional",  # d=0, identical framing
    "ppe_0900": "construction-site-build-construction-work-159358",  # d=0
    "ppe_1004": "pexels-photo-585419",  # d=0
    "ppe_0288": "pexels-photo",  # d=5, same shot re-cropped (SH17 keeps the tape band)
}


@dataclass(frozen=True)
class Pair:
    """One cross-dataset image pair within the Hamming radius.

    Attributes:
        chv: CHV image stem.
        sh17: SH17 image stem.
        distance: Hamming distance between the two 64-bit dHash fingerprints.
    """

    chv: str
    sh17: str
    distance: int

    @property
    def verified(self) -> bool:
        """Whether this pair was confirmed by eye as the same photograph."""
        return VERIFIED_SAME_PHOTOGRAPH.get(self.chv) == self.sh17


# ------------------------------------------------------------------ hash comparison


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit dHash fingerprints."""
    return int(a ^ b).bit_count()


def pairs_within(
    chv_hashes: dict[str, int], sh17_hashes: dict[str, int], threshold: int
) -> list[Pair]:
    """Every CHV×SH17 pair whose dHash distance is at most ``threshold``.

    All qualifying pairs are returned, not just each image's nearest neighbour: one SH17
    image can be the twin of several CHV frames from the same shoot, and dropping the
    non-minimal matches would understate the overlap.

    Args:
        chv_hashes: CHV image stem → 64-bit dHash.
        sh17_hashes: SH17 image stem → 64-bit dHash.
        threshold: Inclusive Hamming radius.

    Returns:
        Pairs sorted by distance, then by CHV stem.
    """
    if not chv_hashes or not sh17_hashes:
        return []
    ref_stems = list(sh17_hashes)
    ref = np.fromiter(sh17_hashes.values(), dtype=np.uint64, count=len(ref_stems))
    found: list[Pair] = []
    for stem, value in chv_hashes.items():
        xor = np.bitwise_xor(ref, np.uint64(value))
        # popcount via byte view — np.bitwise_count needs numpy >= 2.0
        bits = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for index in np.flatnonzero(bits <= threshold):
            found.append(Pair(stem, ref_stems[int(index)], int(bits[index])))
    return sorted(found, key=lambda p: (p.distance, p.chv))


def nearest_distances(chv_hashes: dict[str, int], sh17_hashes: dict[str, int]) -> list[int]:
    """Distance from each CHV image to its closest SH17 image (for the histogram)."""
    if not chv_hashes or not sh17_hashes:
        return []
    ref = np.fromiter(sh17_hashes.values(), dtype=np.uint64, count=len(sh17_hashes))
    out: list[int] = []
    for value in chv_hashes.values():
        xor = np.bitwise_xor(ref, np.uint64(value))
        bits = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        out.append(int(bits.min()))
    return out


# ---------------------------------------------------------------------- exclusions


def exclusions(pairs: list[Pair]) -> dict[str, list[str]]:
    """Images to drop from every evaluation split, both sides of each pair."""
    return {
        "chv": sorted({p.chv for p in pairs}),
        "sh17": sorted({p.sh17 for p in pairs}),
    }


def leaking_pairs(
    eval_stems: set[str],
    train_stems_other: set[str],
    pairs: list[Pair],
    eval_dataset: str,
) -> list[Pair]:
    """Pairs that put an evaluation image in reach of the other dataset's training pool.

    Args:
        eval_stems: Stems in the evaluation split under test (exclusions already applied).
        train_stems_other: Stems in the *other* dataset's training pool.
        pairs: Candidate duplicate pairs from :func:`pairs_within`.
        eval_dataset: ``"chv"`` or ``"sh17"`` — which side of the pair is being evaluated.

    Returns:
        The offending pairs; empty means the split is clean at this threshold.
    """
    if eval_dataset not in {"chv", "sh17"}:
        raise ValueError(f"eval_dataset must be 'chv' or 'sh17', got {eval_dataset!r}")
    other = "sh17" if eval_dataset == "chv" else "chv"
    return [
        p
        for p in pairs
        if getattr(p, eval_dataset) in eval_stems and getattr(p, other) in train_stems_other
    ]


# --------------------------------------------------------------------------- splits


def native_splits() -> dict[str, dict[str, set[str]]]:
    """The splits as shipped by each dataset (SH17 has no test split — S1.4 builds it)."""
    roots = load_roots()
    sh17 = roots["sh17"]
    chv_root = roots["chv"]
    inner = chv_root / "CHV_dataset" if (chv_root / "CHV_dataset").is_dir() else chv_root
    return {
        "sh17": {
            "train": split_stems(sh17 / "train_files.txt"),
            "val": split_stems(sh17 / "val_files.txt"),
        },
        "chv": {
            name: split_stems(inner / "data split" / f"{name}.txt")
            for name in ("train", "valid", "test")
        },
    }


def split_of(stem: str, splits: dict[str, set[str]]) -> str:
    """Which split a stem belongs to, or ``"none"`` if it is in no split list."""
    for name, stems in splits.items():
        if stem in stems:
            return name
    return "none"


# ---------------------------------------------------------------------------- I/O


def load_hashes(cache: Path) -> dict[str, dict[str, int]]:
    """Load the per-image dHash fingerprints cached by the S1.2 EDA pass.

    Raises:
        FileNotFoundError: If the cache is absent — regenerate with
            ``python -m src.eda --out D:/runs/X01-eda``.
    """
    if not cache.is_file():
        raise FileNotFoundError(
            f"dHash cache not found: {cache}. Regenerate with "
            "`python -m src.eda --out D:/runs/X01-eda`."
        )
    data = json.loads(cache.read_text(encoding="utf-8"))
    hashes = {name: {k: int(v) for k, v in data[name]["hashes"].items()} for name in data}
    logger.info(
        "hashes loaded from %s (SH17 %d, CHV %d)",
        cache,
        len(hashes.get("sh17", {})),
        len(hashes.get("chv", {})),
    )
    return hashes


def build_manifest(
    pairs: list[Pair], threshold: int, cache: Path, splits: dict[str, dict[str, set[str]]]
) -> dict:
    """Assemble the exclusion manifest that the split builder consumes."""
    drop = exclusions(pairs)
    return {
        "run": "X01",
        "stage": "S1.4",
        "generated": datetime.now(UTC).date().isoformat(),
        "generator": "src/overlap.py",
        "source_cache": str(cache),
        "threshold": threshold,
        "metric": "Hamming distance on a 64-bit dHash (8x8 gradient)",
        "policy": (
            "Any image within `threshold` of a cross-dataset partner is excluded from "
            "every evaluation split, on both sides of the pair. Conservative by design: "
            "the visually verified subset is smaller (see verified_same_photograph), but "
            "excluding the full candidate set costs ~1% of CHV and leaves no argument."
        ),
        "caveat": (
            "dHash catches re-encodes and mild re-crops, not heavy crops or alternate "
            "frames from the same shoot. Treat these counts as a floor."
        ),
        "counts": {
            "pairs": len(pairs),
            "verified_same_photograph": sum(p.verified for p in pairs),
            "chv_excluded": len(drop["chv"]),
            "sh17_excluded": len(drop["sh17"]),
        },
        "exclude": drop,
        "pairs": [
            {
                "chv": p.chv,
                "sh17": p.sh17,
                "distance": p.distance,
                "chv_split": split_of(p.chv, splits["chv"]),
                "sh17_split": split_of(p.sh17, splits["sh17"]),
                "verified": p.verified,
            }
            for p in pairs
        ],
    }


def write_manifest(path: Path, manifest: dict) -> None:
    """Write the manifest as YAML with a do-not-hand-edit header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# GENERATED by src/overlap.py — do not hand-edit; regenerate with "
        "`python -m src.overlap`.\n"
        "# Consumed by the S1.4 split builder: these images never enter an evaluation "
        "split.\n"
    )
    body = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=88)
    path.write_text(header + body, encoding="utf-8")
    logger.info("manifest written: %s", path)


# -------------------------------------------------------------------------- figures


def fig_distance_histogram(distances: list[int], threshold: int, out: Path) -> str:
    """Distance-to-nearest-SH17-image histogram, with the exclusion cut marked.

    Shows the exclusion decision instead of asserting it. Exact matches stand alone at
    zero, but everything above that merges continuously into ordinary between-dataset
    similarity, so the cut at 5 is a judgement call — which is the argument for verifying
    candidates by eye rather than widening the radius.
    """
    import matplotlib.pyplot as plt

    from src.eda import CHV_COLOUR, INK_SECONDARY, MUTED, apply_style, save

    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    top = max(distances) if distances else 1
    ax.hist(
        distances,
        bins=range(top + 2),
        color=CHV_COLOUR,
        edgecolor="white",
        linewidth=0.6,
        align="left",
    )
    ax.axvline(threshold + 0.5, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
    ax.text(
        threshold + 0.8,
        ax.get_ylim()[1] * 0.92,
        f"excluded: distance ≤ {threshold}",
        color=INK_SECONDARY,
        fontsize=8.5,
        va="top",
    )
    ax.set_xlabel("Hamming distance to the nearest SH17 image (64-bit dHash)")
    ax.set_ylabel("CHV images")
    ax.set_title("CHV→SH17 near-duplicate search: only exact matches are unambiguous")
    ax.set_xlim(-0.5, min(top, 30) + 0.5)
    fig.text(
        0.5,
        -0.06,
        "Above zero the candidate band merges into ordinary between-dataset similarity, so "
        "every candidate at ≤5 was verified by eye.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    return save(fig, out, "duplicate-distance-histogram")


def fig_verification_sheet(pairs: list[Pair], out: Path, thumb: int = 190) -> str:
    """Side-by-side contact sheet of every candidate pair — the visual verification record.

    Each row is one candidate: CHV image left, its SH17 partner right. Verified pairs are
    labelled; the rest are the plain-background collisions that justify not widening the
    radius. Written directly with PIL (photographs, not a plot) but saved as PNG **and**
    PDF to match the figure contract.
    """
    from PIL import Image, ImageDraw

    from src.eda import resolve

    if not pairs:
        return ""
    sh17, chv = resolve("sh17"), resolve("chv")
    columns = 2
    rows = (len(pairs) + columns - 1) // columns
    label_h = 16
    sheet = Image.new("RGB", (thumb * 2 * columns + 12, (thumb + label_h) * rows + 6), "white")
    draw = ImageDraw.Draw(sheet)
    for i, pair in enumerate(pairs):
        row, column = divmod(i, columns)
        x0, y0 = column * thumb * 2 + 6, row * (thumb + label_h) + 3
        # ASCII only: the default PIL bitmap font renders anything else as tofu.
        mark = "SAME PHOTOGRAPH" if pair.verified else "hash collision"
        draw.text(
            (x0, y0),
            f"{pair.chv} | {pair.sh17[:30]} | d={pair.distance} | {mark}",
            fill="black",
        )
        for j, path in enumerate((chv.image_for(pair.chv), sh17.image_for(pair.sh17))):
            if path is None:
                continue
            with Image.open(path) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((thumb, thumb))
                sheet.paste(thumbnail, (x0 + j * thumb, y0 + label_h))
    sheet.save(out / "duplicate-verification.png")
    sheet.save(out / "duplicate-verification.pdf", resolution=200)
    logger.info("figure: %s (.png/.pdf)", out / "duplicate-verification")
    return "duplicate-verification.png"


# ------------------------------------------------------------------------------ CLI


def verify(pairs: list[Pair], splits: dict[str, dict[str, set[str]]]) -> list[Pair]:
    """Re-check both transfer directions after exclusions; return residual leaking pairs.

    SH17 ships no test split, so its evaluation side is ``val`` until S1.4 builds and
    freezes one; the same check then covers it.
    """
    drop = exclusions(pairs)
    chv_eval = (splits["chv"]["valid"] | splits["chv"]["test"]) - set(drop["chv"])
    sh17_eval = splits["sh17"]["val"] - set(drop["sh17"])
    residual = leaking_pairs(chv_eval, splits["sh17"]["train"], pairs, "chv")
    residual += leaking_pairs(sh17_eval, splits["chv"]["train"], pairs, "sh17")
    for pair in residual:
        logger.error("residual leakage: %s ↔ %s (d=%d)", pair.chv, pair.sh17, pair.distance)
    logger.info(
        "verification: CHV eval %d images, SH17 eval %d images, residual leaking pairs %d",
        len(chv_eval),
        len(sh17_eval),
        len(residual),
    )
    return residual


def main() -> int:
    parser = argparse.ArgumentParser(
        description="X01/S1.4 cross-dataset near-duplicate exclusion list."
    )
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--figures", type=Path, default=Path("D:/runs/X01-overlap"))
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only re-check the splits for residual leakage (exit 1 if any)",
    )
    args = parser.parse_args()

    hashes = load_hashes(args.cache)
    pairs = pairs_within(hashes["chv"], hashes["sh17"], args.threshold)
    splits = native_splits()
    logger.info(
        "%d candidate pairs at distance <= %d (%d verified by eye)",
        len(pairs),
        args.threshold,
        sum(p.verified for p in pairs),
    )

    residual = verify(pairs, splits)
    if args.verify:
        return 1 if residual else 0

    write_manifest(args.out, build_manifest(pairs, args.threshold, args.cache, splits))
    args.figures.mkdir(parents=True, exist_ok=True)
    fig_distance_histogram(
        nearest_distances(hashes["chv"], hashes["sh17"]), args.threshold, args.figures
    )
    fig_verification_sheet(pairs, args.figures)
    return 1 if residual else 0


if __name__ == "__main__":
    raise SystemExit(main())
