"""X01 / S1.4 harmonisation: one shared label space, and the frozen splits (the O1 contribution).

SH17 and CHV describe the same world with different vocabularies: SH17 has 17 classes with
`safety-vest` at id 16, CHV has 6 with four *colours* of helmet. Training and cross-testing in
both directions needs one vocabulary, and the mapping between them is a documented method
contribution rather than an implementation detail, so it lives here, in one regenerable
script, and never as a hand-edited label file.

**Target space: `{0: person, 1: helmet, 2: vest}`.** `no-helmet` / `no-vest` are deliberately
**not** classes: no training set annotates worn-state (X01/F2), and the only set that labels
compliance is evaluation-only by supervisor condition. Those states are *derived* at inference
by the association rule and scored zero-shot on Pictor-PPE.

What this script does, and why each step is defensible:

- **Maps and drops.** SH17 keeps person/helmet/safety-vest and drops 14 body-part and
  equipment classes; CHV collapses four helmet colours into one. Every dropped class is
  recorded in the report, because a silent drop is indistinguishable from a bug.
- **Mirrors, never mutates.** Harmonised labels are written to a new root beside hard-linked
  images (no pixel is copied and no source file is touched), so the originals stay pristine
  and the whole tree can be deleted and rebuilt.
- **Freezes splits with an ID.** SH17's shipped `val` is C4's test partition (decision
  2026-07-26), so it becomes our **test** set and validation is carved out of train with a
  fixed seed. CHV's shipped three-way split is used as-is. Each split gets a content hash so a
  number in the dissertation can be traced to the exact image list that produced it.
- **Applies the near-duplicate exclusions** (X01/S1.4 F10) to every *evaluation* split, so the
  zero-shot claim is enforced by construction.
- **Validates before it exits.** Class ids, coordinate ranges, image presence, split disjointness
  and residual leakage are all asserted; the script fails loudly rather than emitting a subtly
  broken dataset.

Usage::

    python -m src.harmonise                    # build everything + report + sanity figure
    python -m src.harmonise --check            # validate an existing build, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.audit_labels import load_roots, read_yolo_rows, split_stems
from src.eda import Dataset, resolve
from src.overlap import load_hashes, matches_within
from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_NAMES = {0: "person", 1: "helmet", 2: "vest"}

# Source id -> target id. Anything absent is dropped on purpose (reported, never silent).
SOURCE_MAPS: dict[str, dict[int, int]] = {
    # X01/F1-corrected ids: person 0, helmet 10, safety-vest 16 (the vault's old 1/13/15 was wrong).
    "sh17": {0: 0, 10: 1, 16: 2},
    # CHV ships `0 person, 1 vest, 2 blue helmet, 3 red helmet, 4 white helmet, 5 yellow helmet`
    # (annotations/README.md). Colour is not a safety property: all four collapse to `helmet`.
    "chv": {0: 0, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1},
}

DEFAULT_OUT = Path("D:/Dissertation/harmonised")
DEFAULT_EXCLUSIONS = Path("configs/splits/X01-cross-dataset-duplicates.yaml")
DEFAULT_YAML_DIR = Path("configs/data")
VAL_FRACTION = 0.10  # carved out of SH17's shipped train, for early stopping only
SPLIT_SEED = 0

# Splits a model is scored on. Near-duplicate exclusions apply to these and only these:
# removing an image from training would shrink supervision without making any claim safer.
EVAL_SPLITS = ("val", "test")


@dataclass
class Conversion:
    """Outcome of converting one dataset into the shared label space."""

    images: int = 0
    written: int = 0
    kept: Counter = field(default_factory=Counter)
    dropped: Counter = field(default_factory=Counter)
    emptied: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)


def convert(dataset: Dataset, mapping: dict[int, int], out_root: Path) -> Conversion:
    """Write harmonised labels and hard-link the images into ``out_root``.

    Hard links keep the build free of duplicated pixels; if the destination is on another
    volume (where links are impossible) the file is copied instead.
    """
    images_dir, labels_dir = out_root / "images", out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    result = Conversion()

    for label_path in dataset.label_files:
        stem = label_path.stem
        source_image = dataset.image_for(stem)
        result.images += 1
        if source_image is None:
            result.missing_images.append(stem)
            continue

        rows, _ = read_yolo_rows(label_path)
        lines = []
        for cls, x, y, w, h in rows:
            target = mapping.get(cls)
            if target is None:
                result.dropped[cls] += 1
                continue
            result.kept[target] += 1
            lines.append(f"{target} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        if not lines:
            result.emptied.append(stem)

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        destination = images_dir / source_image.name
        if not destination.exists():
            try:
                destination.hardlink_to(source_image)
            except OSError:
                shutil.copy2(source_image, destination)
        result.written += 1
    return result


# ---------------------------------------------------------------------------- splits


def stratum(label_file: Path) -> str:
    """Stratification key: which target classes the image contains.

    Splitting at random would let the rare classes (helmet, vest) drift between train and
    val; keying on the class combination keeps their proportions stable.
    """
    rows, _ = read_yolo_rows(label_file)
    present = sorted({row[0] for row in rows})
    return "-".join(str(c) for c in present) if present else "background"


def stratified_split(
    stems: list[str], labels_dir: Path, fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split ``stems`` into (majority, minority) holding class proportions, deterministically."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        buckets[stratum(labels_dir / f"{stem}.txt")].append(stem)

    rng = random.Random(seed)
    majority: list[str] = []
    minority: list[str] = []
    for key in sorted(buckets):
        members = sorted(buckets[key])
        rng.shuffle(members)
        cut = round(len(members) * fraction)
        minority.extend(members[:cut])
        majority.extend(members[cut:])
    return sorted(majority), sorted(minority)


def split_id(stems: list[str]) -> str:
    """Stable content hash of a split, so a reported number traces to an exact image list."""
    joined = "\n".join(sorted(stems)).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:12]


def build_splits(
    dataset: str, out_root: Path, drop: set[str], seed: int = SPLIT_SEED
) -> dict[str, list[str]]:
    """Construct the frozen splits for one dataset, with exclusions applied to eval splits.

    SH17 ships train/val only. Its shipped ``val`` is C4's 20 % test partition (decision
    2026-07-26), so it becomes **test**, putting our in-domain number beside C4's published
    one, and validation is carved out of train. CHV's shipped three-way split is used as-is.
    """
    roots = load_roots()
    if dataset == "sh17":
        shipped_train = {Path(s).stem for s in split_stems(roots["sh17"] / "train_files.txt")}
        shipped_val = {Path(s).stem for s in split_stems(roots["sh17"] / "val_files.txt")}
        train, val = stratified_split(
            sorted(shipped_train), out_root / "labels", VAL_FRACTION, seed
        )
        splits = {"train": train, "val": val, "test": sorted(shipped_val)}
    else:
        root = roots["chv"]
        inner = root / "CHV_dataset" if (root / "CHV_dataset").is_dir() else root
        splits = {
            name: sorted(Path(s).stem for s in split_stems(inner / "data split" / f"{source}.txt"))
            for name, source in (("train", "train"), ("val", "valid"), ("test", "test"))
        }

    for name in EVAL_SPLITS:
        splits[name] = [stem for stem in splits[name] if stem not in drop]
    return splits


def write_split_lists(out_root: Path, splits: dict[str, list[str]]) -> dict[str, Path]:
    """Write one image-path list per split (the form Ultralytics reads)."""
    images_dir = out_root / "images"
    by_stem = {path.stem: path for path in images_dir.iterdir() if path.is_file()}
    written = {}
    for name, stems in splits.items():
        path = out_root / f"{name}.txt"
        lines = [str(by_stem[stem].resolve()) for stem in stems if stem in by_stem]
        path.write_text("\n".join(lines), encoding="utf-8")
        written[name] = path
    return written


def write_dataset_yaml(dataset: str, out_root: Path, yaml_dir: Path) -> Path:
    """Write the Ultralytics dataset YAML for one harmonised dataset."""
    yaml_dir.mkdir(parents=True, exist_ok=True)
    path = yaml_dir / f"{dataset}.yaml"
    document = {
        "path": str(out_root.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": dict(TARGET_NAMES),
    }
    header = (
        f"# GENERATED by src/harmonise.py. Do not hand-edit; rebuild with "
        f"`python -m src.harmonise`.\n"
        f"# Harmonised label space (O1): {TARGET_NAMES}.\n"
        f"# Cross-dataset evaluation = train on one of these YAMLs, validate against the other.\n"
    )
    path.write_text(header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------------ validation


def validate(out_root: Path, splits: dict[str, list[str]]) -> list[str]:
    """Assert the harmonised build is well formed. Returns a list of problems (empty = good)."""
    problems: list[str] = []
    labels_dir, images_dir = out_root / "labels", out_root / "images"
    by_stem = {path.stem for path in images_dir.iterdir() if path.is_file()}

    for label_path in sorted(labels_dir.glob("*.txt")):
        for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{label_path.name}:{number}: expected 5 fields, got {len(parts)}")
                continue
            cls = int(parts[0])
            if cls not in TARGET_NAMES:
                problems.append(f"{label_path.name}:{number}: class {cls} outside the target space")
            for value in (float(v) for v in parts[1:]):
                if not 0.0 <= value <= 1.0:
                    problems.append(f"{label_path.name}:{number}: coordinate {value} outside [0,1]")

    for name, stems in splits.items():
        missing = [stem for stem in stems if stem not in by_stem]
        if missing:
            problems.append(f"split {name}: {len(missing)} images missing (e.g. {missing[:3]})")

    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlap = set(splits[first]) & set(splits[second])
            if overlap:
                problems.append(f"splits {first}/{second} overlap by {len(overlap)} images")
    return problems


def check_residual_leakage(built: dict[str, dict[str, list[str]]], cache: Path) -> list[str]:
    """Re-run the cross-dataset duplicate check against the *frozen* splits.

    The exclusion list was computed before these splits existed; this proves it actually
    landed, including on the newly constructed SH17 test split.
    """
    problems: list[str] = []
    try:
        hashes = load_hashes(cache)
    except FileNotFoundError:
        logger.warning("dHash cache absent; skipping the post-build leakage check")
        return problems

    for evaluated, other in (("chv", "sh17"), ("sh17", "chv")):
        eval_stems = {s for name in EVAL_SPLITS for s in built[evaluated][name]}
        train_stems = set(built[other]["train"])
        query = {k: v for k, v in hashes[evaluated].items() if k in eval_stems}
        reference = {k: v for k, v in hashes[other].items() if k in train_stems}
        leaks = matches_within(query, reference, threshold=5)
        logger.info("leakage check %s eval vs %s train: %d pairs", evaluated, other, len(leaks))
        problems += [f"leak: {a} ~ {b} (d={d})" for a, b, d in leaks]
    return problems


# --------------------------------------------------------------------------- figures


def fig_sanity_render(out_root: Path, dataset: str, figures: Path, count: int = 6) -> None:
    """Draw harmonised boxes on sample images: the visual proof the mapping is right."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from PIL import Image

    from src.eda import CHV_COLOUR, INK, SH17_COLOUR, apply_style

    apply_style()
    colours = {0: INK, 1: SH17_COLOUR if dataset == "sh17" else CHV_COLOUR, 2: "#2e8b57"}
    labels_dir, images_dir = out_root / "labels", out_root / "images"
    by_stem = {path.stem: path for path in images_dir.iterdir() if path.is_file()}

    # Prefer images that exercise the whole mapping (all three classes present).
    candidates = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        rows, _ = read_yolo_rows(label_path)
        present = {row[0] for row in rows}
        if len(present) >= 2 and label_path.stem in by_stem:
            candidates.append((len(present), label_path))
        if len(candidates) > 400:
            break
    candidates.sort(key=lambda item: -item[0])
    chosen = [path for _score, path in candidates[:count]]

    columns = 3
    rows_n = max(1, (len(chosen) + columns - 1) // columns)
    fig, axes = plt.subplots(rows_n, columns, figsize=(11, 3.6 * rows_n))
    for axis, label_path in zip(axes.flat, chosen, strict=False):
        with Image.open(by_stem[label_path.stem]) as image:
            picture = image.convert("RGB")
            picture.thumbnail((520, 520))
        axis.imshow(picture)
        width, height = picture.size
        for cls, x, y, w, h in read_yolo_rows(label_path)[0]:
            axis.add_patch(
                patches.Rectangle(
                    ((x - w / 2) * width, (y - h / 2) * height),
                    w * width,
                    h * height,
                    fill=False,
                    edgecolor=colours[cls],
                    linewidth=1.4,
                )
            )
        axis.set_title(f"{label_path.stem[:26]}", fontsize=8)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
    legend = " · ".join(f"{TARGET_NAMES[c]} = {colours[c]}" for c in sorted(TARGET_NAMES))
    fig.suptitle(f"{dataset.upper()}: harmonised boxes ({legend})", fontsize=10)
    fig.tight_layout()
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"harmonised-sample-{dataset}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("figure: %s", figures / f"harmonised-sample-{dataset}")


# ---------------------------------------------------------------------------- report


def build_report(
    conversions: dict[str, Conversion],
    built: dict[str, dict[str, list[str]]],
    manifest: dict,
    problems: list[str],
) -> str:
    """Markdown summary of the harmonisation (the ledger stays canonical for findings)."""
    out = [
        "# X01 / S1.4: harmonisation + frozen splits",
        "",
        f"Target label space: `{TARGET_NAMES}`",
        "",
        "## Mapping applied",
        "",
        "| dataset | source id → target | dropped classes (instances) |",
        "|---|---|---|",
    ]
    for name, conversion in conversions.items():
        mapped = ", ".join(f"{k}→{v}" for k, v in sorted(SOURCE_MAPS[name].items()))
        dropped = ", ".join(f"{k}: {v}" for k, v in sorted(conversion.dropped.items())) or "-"
        out.append(f"| {name.upper()} | {mapped} | {dropped} |")

    out += [
        "",
        "## Instances kept",
        "",
        "| dataset | person | helmet | vest | empty images |",
        "|---|---|---|---|---|",
    ]
    for name, conversion in conversions.items():
        out.append(
            f"| {name.upper()} | {conversion.kept[0]} | {conversion.kept[1]} | "
            f"{conversion.kept[2]} | {len(conversion.emptied)} |"
        )

    out += [
        "",
        "## Frozen splits",
        "",
        "| dataset | split | images | split ID |",
        "|---|---|---|---|",
    ]
    for name, splits in built.items():
        for split, stems in splits.items():
            out.append(f"| {name.upper()} | {split} | {len(stems)} | `{split_id(stems)}` |")

    out += [
        "",
        (
            f"Validation seed: `{manifest['seed']}` · SH17 val fraction carved from train: "
            f"`{VAL_FRACTION}` · exclusions from `{manifest['exclusions_source']}` "
            f"(CHV {manifest['excluded']['chv']}, SH17 {manifest['excluded']['sh17']} images, "
            "eval splits only)"
        ),
        "",
        "## Validation",
        "",
        f"- Problems found: **{len(problems)}**",
    ]
    out += [f"  - {problem}" for problem in problems[:20]]
    out += ["", "_Generated by `python -m src.harmonise`._"]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="X01 / S1.4 harmonisation + split freeze.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--yaml-dir", type=Path, default=DEFAULT_YAML_DIR)
    parser.add_argument("--figures", type=Path, default=Path("D:/runs/X01-harmonise"))
    parser.add_argument("--cache", type=Path, default=Path("D:/runs/X01-eda/stats-cache.json"))
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--check", action="store_true", help="validate an existing build only")
    args = parser.parse_args()

    manifest_exclusions = yaml.safe_load(args.exclusions.read_text(encoding="utf-8"))
    drop = manifest_exclusions["exclude"]
    logger.info(
        "exclusions loaded: CHV %d, SH17 %d (from %s)",
        len(drop["chv"]),
        len(drop["sh17"]),
        args.exclusions,
    )

    conversions: dict[str, Conversion] = {}
    built: dict[str, dict[str, list[str]]] = {}
    problems: list[str] = []

    for name in ("sh17", "chv"):
        out_root = args.out / name
        if not args.check:
            logger.info("converting %s → %s", name.upper(), out_root)
            conversions[name] = convert(resolve(name), SOURCE_MAPS[name], out_root)
            logger.info(
                "%s: %d labels written, %d instances kept, %d images emptied",
                name.upper(),
                conversions[name].written,
                sum(conversions[name].kept.values()),
                len(conversions[name].emptied),
            )
        built[name] = build_splits(name, out_root, set(drop[name]), args.seed)
        logger.info("%s splits: %s", name.upper(), {k: len(v) for k, v in built[name].items()})
        if not args.check:
            write_split_lists(out_root, built[name])
            write_dataset_yaml(name, out_root, args.yaml_dir)
        problems += [f"{name}: {p}" for p in validate(out_root, built[name])]

    problems += check_residual_leakage(built, args.cache)

    manifest = {
        "generated_by": "src/harmonise.py",
        "target_space": TARGET_NAMES,
        "source_maps": SOURCE_MAPS,
        "seed": args.seed,
        "val_fraction_sh17": VAL_FRACTION,
        "exclusions_source": str(args.exclusions),
        "excluded": {name: len(stems) for name, stems in drop.items()},
        "splits": {
            dataset: {
                split: {"images": len(stems), "split_id": split_id(stems)}
                for split, stems in splits.items()
            }
            for dataset, splits in built.items()
        },
    }
    if not args.check:
        (args.out / "split-manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        for name in ("sh17", "chv"):
            fig_sanity_render(args.out / name, name, args.figures)
        args.figures.mkdir(parents=True, exist_ok=True)
        (args.figures / "harmonise-report.md").write_text(
            build_report(conversions, built, manifest, problems), encoding="utf-8"
        )
        logger.info("report written: %s", args.figures / "harmonise-report.md")

    if problems:
        for problem in problems[:20]:
            logger.error("validation: %s", problem)
        logger.error("%d validation problem(s); build is NOT usable", len(problems))
        return 1
    logger.info("validation clean: harmonised build is usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
