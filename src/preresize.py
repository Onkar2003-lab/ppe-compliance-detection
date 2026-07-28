"""X03 / S3 — offline pre-resize of the harmonised images to the training resolution.

**Why this exists.** The S3 timing pilot was I/O-bound, not compute-bound (finding **F21**):
~270 s/epoch for yolov8n at 640 with the **GPU at 0 %**, because SH17's median image is
4475x4057 (~18 MP) and every one of them was being decoded at full size, once per epoch, only
to be immediately letterboxed down to 640. ``cache='disk'`` made it worse rather than better:
it stores the *decoded* array, ~60 MB per image, so an epoch over the 5,832-image train split
read ~350 GB. Measured on 40 images: full-size decode + resize **99.8 ms**, pre-resized decode
**1.3 ms** — a 76x difference on the data path, and 63 GB -> 6.6 GB on disk.

**Why this does not change the experiment.** Ultralytics letterboxes every image to
``imgsz=640`` regardless, so the array the network sees is the same either way; this performs
that reduction **once** instead of once per epoch. The training input is therefore unchanged,
which is the ground on which the supervisor approved it (2026-07-28) and the reason
comparability with C3/C4 — who also train at 640 — is unaffected. It amends, but does not
contradict, the S1.3 decision to leave resizing framework-native: the resize still happens
exactly once and to exactly the same target, only earlier.

**What is given up**, stated plainly: training at a *higher* ``imgsz`` later would need this
root rebuilding, and each downsized image carries one extra resample + one JPEG re-encode at
q95. Against a 17.7 MP -> 0.4 MP reduction that was already happening every epoch, that is a
rounding error — but it is a real one, so it is recorded rather than glossed.

**Mirrors, never mutates.** A new root is written beside the harmonised one; the harmonised
build (and the untouched originals behind its hard links) stays pristine. Labels are
**hard-linked, not rewritten**: YOLO coordinates are normalised to [0, 1], so a proportional
resize leaves them bit-for-bit correct, and :func:`validate` asserts exactly that rather than
trusting it. Images already within the target are hard-linked too, so no image is re-encoded
without cause.

Usage::

    python -m src.preresize                    # build both datasets at 640
    python -m src.preresize --datasets sh17    # SH17 only (the one that matters for speed)
    python -m src.preresize --verify           # re-check an existing build, exit 1 on problems

Emits ``configs/data/<dataset>-640.yaml``; the run config points ``data:`` at that instead of
the full-resolution YAML. Regenerable from scratch at any time — nothing here is hand-kept.
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("D:/Dissertation/harmonised")
DEFAULT_OUT = Path("D:/Dissertation/harmonised-640")
DEFAULT_YAML_DIR = Path("configs/data")
DEFAULT_DATASETS = ("sh17", "chv")
SPLITS = ("train", "val", "test")

TARGET_LONG_SIDE = 640  # == configs/base.yaml imgsz; the two must not drift apart
JPEG_QUALITY = 95
TARGET_NAMES = {0: "person", 1: "helmet", 2: "vest"}


@dataclass
class Build:
    """Outcome of one dataset's pre-resize pass."""

    dataset: str
    resized: int = 0
    linked: int = 0  # already within target — carried across untouched
    failed: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.resized + self.linked


# ----------------------------------------------------------------------------- geometry


def target_size(
    width: int, height: int, long_side: int = TARGET_LONG_SIDE
) -> tuple[int, int] | None:
    """Return the downscaled ``(width, height)``, or ``None`` if the image is already small.

    Scales the **longest** side to ``long_side`` and preserves aspect ratio — the same
    reduction Ultralytics' letterbox performs, so the pixels the network sees are unchanged.
    Never upscales: an image below the target is left exactly as it is.
    """
    longest = max(width, height)
    if longest <= long_side:
        return None
    scale = long_side / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link ``source`` to ``destination``, falling back to a copy across volumes."""
    if destination.exists():
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


# ----------------------------------------------------------------------------- one image


def process_image(task: tuple[str, str]) -> tuple[str, str]:
    """Resize one image if it exceeds the target; otherwise hard-link it.

    Top-level (not a closure) so :class:`ProcessPoolExecutor` can pickle it. Returns
    ``(outcome, detail)`` where outcome is ``resized`` / ``linked`` / ``failed``.
    """
    source, destination = Path(task[0]), Path(task[1])
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        return "failed", f"unreadable: {source.name}"

    height, width = image.shape[:2]
    size = target_size(width, height)
    if size is None:
        link_or_copy(source, destination)
        return "linked", source.name

    # INTER_AREA is the correct kernel for downscaling — it averages the source pixels that
    # fall inside each destination pixel, so it does not alias the way INTER_LINEAR does.
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    params = (
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        if destination.suffix.lower() in {".jpg", ".jpeg"}
        else []
    )
    if not cv2.imwrite(str(destination), resized, params):
        return "failed", f"unwritable: {destination.name}"
    return "resized", source.name


# ----------------------------------------------------------------------------- one dataset


def read_split(path: Path) -> list[str]:
    """Read one Ultralytics split list into image paths (absolute, as harmonise writes them)."""
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(dataset: str, source_root: Path, out_root: Path, workers: int) -> Build:
    """Write the pre-resized mirror of one harmonised dataset."""
    source_images, source_labels = source_root / "images", source_root / "labels"
    out_images, out_labels = out_root / "images", out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in source_images.iterdir() if path.is_file())
    tasks = [(str(path), str(out_images / path.name)) for path in images]

    result = Build(dataset=dataset)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (outcome, detail) in enumerate(pool.map(process_image, tasks, chunksize=16), 1):
            if outcome == "resized":
                result.resized += 1
            elif outcome == "linked":
                result.linked += 1
            else:
                result.failed.append(detail)
            if index % 1000 == 0:
                logger.info("%s: %d/%d images", dataset, index, len(tasks))

    # Labels are byte-identical by construction (normalised coordinates); link, never rewrite.
    for label in sorted(source_labels.glob("*.txt")):
        link_or_copy(label, out_labels / label.name)

    # Split lists: same membership, repointed at the new root. Rebuilding them from the source
    # lists (rather than re-globbing) is what guarantees the frozen splits survive intact.
    for split in SPLITS:
        lines = read_split(source_root / f"{split}.txt")
        if not lines:
            continue
        repointed = [str((out_images / Path(line).name).resolve()) for line in lines]
        (out_root / f"{split}.txt").write_text("\n".join(repointed), encoding="utf-8")

    return result


def write_dataset_yaml(dataset: str, out_root: Path, yaml_dir: Path) -> Path:
    """Write the Ultralytics dataset YAML pointing at the pre-resized root."""
    yaml_dir.mkdir(parents=True, exist_ok=True)
    path = yaml_dir / f"{dataset}-640.yaml"
    document = {
        "path": str(out_root.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": dict(TARGET_NAMES),
    }
    header = (
        f"# GENERATED by src/preresize.py — do not hand-edit; rebuild with "
        f"`python -m src.preresize`.\n"
        f"# Images pre-scaled to a {TARGET_LONG_SIDE}px long side (X03/F21: the full-resolution\n"
        f"# loader was I/O-bound). Labels and splits are identical to the harmonised build —\n"
        f"# normalised YOLO coordinates are unaffected by a proportional resize.\n"
    )
    path.write_text(header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- validation


def validate(dataset: str, source_root: Path, out_root: Path) -> list[str]:
    """Assert the mirror is faithful. Returns a list of problems (empty = good).

    Checks the four things that would silently invalidate every downstream run: a lost image,
    a changed label, a changed split, and an image that did not actually come down to size.
    """
    problems: list[str] = []
    source_images, out_images = source_root / "images", out_root / "images"
    source_labels, out_labels = source_root / "labels", out_root / "labels"

    source_names = {path.name for path in source_images.iterdir() if path.is_file()}
    out_names = {path.name for path in out_images.iterdir() if path.is_file()}
    if source_names != out_names:
        missing, extra = source_names - out_names, out_names - source_names
        problems.append(
            f"{dataset}: image set differs ({len(missing)} missing, {len(extra)} extra; "
            f"e.g. {sorted(missing)[:3] or sorted(extra)[:3]})"
        )

    # Labels must be identical, not merely present — this is the claim the whole approach rests on.
    for label in sorted(source_labels.glob("*.txt")):
        mirrored = out_labels / label.name
        if not mirrored.exists():
            problems.append(f"{dataset}: label missing after mirror ({label.name})")
        elif mirrored.read_bytes() != label.read_bytes():
            problems.append(
                f"{dataset}: label CHANGED by the resize ({label.name}) — must not happen"
            )

    for split in SPLITS:
        before = [Path(line).name for line in read_split(source_root / f"{split}.txt")]
        after = [Path(line).name for line in read_split(out_root / f"{split}.txt")]
        if before != after:
            problems.append(
                f"{dataset}/{split}: split membership changed ({len(before)} -> {len(after)})"
            )
        for line in read_split(out_root / f"{split}.txt"):
            if not Path(line).exists():
                problems.append(f"{dataset}/{split}: listed image does not exist ({line})")
                break

    oversized = 0
    for path in out_images.iterdir():
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
        if image is None:
            problems.append(f"{dataset}: unreadable after build ({path.name})")
            break
        if max(image.shape[:2]) * 8 > TARGET_LONG_SIDE * 1.2:  # /8 decode, 20% slack
            oversized += 1
    if oversized:
        problems.append(f"{dataset}: {oversized} images still exceed {TARGET_LONG_SIDE}px")

    return problems


def directory_size_gb(root: Path) -> float:
    """Total size of a directory tree, in GB (hard links counted once per name, as on disk)."""
    total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return total / 1024**3


# ---------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-resize harmonised images to the train size.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--yaml-dir", type=Path, default=DEFAULT_YAML_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify", action="store_true", help="validate an existing build only")
    arguments = parser.parse_args()

    problems: list[str] = []
    for dataset in arguments.datasets:
        source_root, out_root = arguments.source / dataset, arguments.out / dataset
        if not source_root.exists():
            logger.error(
                "harmonised root missing: %s (run `python -m src.harmonise` first)", source_root
            )
            return 1

        if not arguments.verify:
            logger.info("%s: building %s -> %s", dataset, source_root, out_root)
            result = build(dataset, source_root, out_root, arguments.workers)
            logger.info(
                "%s: %d images (%d resized, %d already within target), %d failed",
                dataset,
                result.total,
                result.resized,
                result.linked,
                len(result.failed),
            )
            for failure in result.failed[:10]:
                logger.error("%s: %s", dataset, failure)
            problems.extend(f"{dataset}: {failure}" for failure in result.failed)
            written = write_dataset_yaml(dataset, out_root, arguments.yaml_dir)
            logger.info("%s: wrote %s", dataset, written)
            logger.info(
                "%s: %.1f GB -> %.1f GB",
                dataset,
                directory_size_gb(source_root / "images"),
                directory_size_gb(out_root / "images"),
            )

        found = validate(dataset, source_root, out_root)
        problems.extend(found)
        logger.info("%s: validation %s", dataset, "OK" if not found else f"{len(found)} PROBLEMS")

    if problems:
        for problem in problems[:20]:
            logger.error("PROBLEM: %s", problem)
        return 1

    logger.info(
        "pre-resize complete and validated — point the run config at configs/data/<ds>-640.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
