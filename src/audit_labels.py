"""X01 / S1.1 — SH17 + CHV label-file audit (descriptive; no training, no GPU).

Answers the three questions S1.1 owes the build plan:
  1. **A1** — do SH17 and CHV really share {helmet, vest, person} at the *label-file*
     level (not just in the papers)?
  2. **Violation route** — how does SH17 encode non-compliance? It ships no explicit
     ``no-helmet``/``no-vest`` class, so the audit reports the full class list and the
     per-class counts that decide "SH17 worn-state proxy vs adopt the E5 set".
  3. **Data integrity** — split coverage, image/label parity, malformed lines, bbox
     coordinates outside [0, 1] — the asserts every later training run depends on.

Layout is **discovered per dataset, never assumed** (the two roots differ):

*SH17* — ``images/`` + ``labels/`` (YOLO txt) + ``voc_labels/`` (Pascal-VOC xml) +
``meta-data/`` (Pexels provenance json) + ``train_files.txt`` / ``val_files.txt``.
It has **no names file**, so class ids are named by geometry-matching each YOLO row to
its VOC object (:func:`derive_names_from_voc`).

*CHV* — ``images/`` + ``annotations/`` (YOLO txt + a ``README.md`` naming the classes) +
``data split/`` (``train.txt`` / ``valid.txt`` / ``test.txt`` listing image paths).

Directory recursion is deliberately avoided: ``SH17_dataset/meta-data/Documents`` holds
unrelated personal files, so a naive ``rglob`` would walk (and report on) them.

Usage::

    python src/audit_labels.py --dataset sh17 --out D:/runs/X01-audit/sh17.md
    python src/audit_labels.py --dataset chv  --out D:/runs/X01-audit/chv.md

Roots come from ``configs/base.yaml`` (``datasets.sh17_root`` / ``datasets.chv_root``).
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CORE = ("helmet", "no-helmet", "vest", "no-vest", "person")  # O1 target label space
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


# --------------------------------------------------------------------------- scan


@dataclass
class LabelScan:
    """Aggregate of one pass over a set of YOLO ``.txt`` label files."""

    per_class: Counter = field(default_factory=Counter)
    n_files: int = 0
    n_empty: int = 0
    n_malformed: int = 0
    n_out_of_range: int = 0
    instances_per_file: list[int] = field(default_factory=list)
    box_wh: list[tuple[float, float]] = field(default_factory=list)

    @property
    def n_instances(self) -> int:
        return sum(self.per_class.values())


def read_yolo_rows(path: Path) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Parse one YOLO label file.

    Returns:
        ``(rows, n_malformed)`` where each row is ``(cls, xc, yc, w, h)``. A line is
        malformed if it has fewer than 5 fields or non-numeric values; malformed lines
        are counted, not raised, so the audit can report the whole dataset in one pass.
    """
    rows: list[tuple[int, float, float, float, float]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            malformed += 1
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            malformed += 1
            continue
        rows.append((cls, xc, yc, w, h))
    return rows, malformed


def scan_labels(files: list[Path]) -> LabelScan:
    """Scan label files for class counts, integrity problems and bbox scale."""
    scan = LabelScan()
    for path in files:
        scan.n_files += 1
        rows, malformed = read_yolo_rows(path)
        scan.n_malformed += malformed
        if not rows:
            scan.n_empty += 1
        scan.instances_per_file.append(len(rows))
        for cls, xc, yc, w, h in rows:
            scan.per_class[cls] += 1
            scan.box_wh.append((w, h))
            if any(v < 0.0 or v > 1.0 for v in (xc, yc, w, h)):
                scan.n_out_of_range += 1
    return scan


# ------------------------------------------------------------------- class names


def voc_objects(path: Path) -> tuple[list[tuple[str, float, float, float, float]], tuple[int, int]]:
    """Read a Pascal-VOC xml and return objects as normalised ``(name, xc, yc, w, h)``."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        return [], (0, 0)
    img_w = int(float(size.findtext("width", "0")))
    img_h = int(float(size.findtext("height", "0")))
    if img_w <= 0 or img_h <= 0:
        return [], (img_w, img_h)

    objects: list[tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymin = float(box.findtext("ymin", "0"))
        ymax = float(box.findtext("ymax", "0"))
        objects.append(
            (
                (obj.findtext("name") or "?").strip(),
                (xmin + xmax) / 2 / img_w,
                (ymin + ymax) / 2 / img_h,
                (xmax - xmin) / img_w,
                (ymax - ymin) / img_h,
            )
        )
    return objects, (img_w, img_h)


def derive_names_from_voc(
    label_dir: Path, voc_dir: Path, sample: int = 1500
) -> tuple[dict[int, str], dict[int, Counter], int]:
    """Recover SH17's class-id → name map by matching YOLO rows to VOC objects.

    SH17 ships no ``data.yaml``/``classes.txt``, but every image has both a YOLO ``.txt``
    and a VOC ``.xml``. Each YOLO row is matched to the VOC object whose normalised
    centre+size is nearest; the winning object's ``<name>`` votes for that class id.

    Args:
        label_dir: Directory of YOLO ``.txt`` files.
        voc_dir: Directory of matching ``.xml`` files (same stems).
        sample: How many label files to vote over (the map converges long before the
            full 8,099; kept configurable so the whole set can be checked).

    Returns:
        ``(names, votes, n_matched)`` — the winning name per class id, the full vote
        tally per id (so purity/conflicts are inspectable), and the number of files
        actually matched.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    n_matched = 0
    for path in sorted(label_dir.glob("*.txt"))[:sample]:
        xml = voc_dir / f"{path.stem}.xml"
        if not xml.exists():
            continue
        rows, _ = read_yolo_rows(path)
        objects, _ = voc_objects(xml)
        if not rows or not objects:
            continue
        n_matched += 1
        for cls, xc, yc, w, h in rows:
            name, _dist = min(
                (
                    (o[0], (o[1] - xc) ** 2 + (o[2] - yc) ** 2 + (o[3] - w) ** 2 + (o[4] - h) ** 2)
                    for o in objects
                ),
                key=lambda pair: pair[1],
            )
            votes[cls][name] += 1
    names = {cls: tally.most_common(1)[0][0] for cls, tally in votes.items()}
    return names, dict(votes), n_matched


def names_from_readme(path: Path) -> dict[int, str]:
    """Parse CHV's ``annotations/README.md`` class list (``0 : person`` lines)."""
    names: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        match = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
        if match:
            names[int(match.group(1))] = match.group(2)
    return names


def suggest_core_mapping(name: str) -> str:
    """Heuristic map of a dataset class name onto :data:`CORE` (for human review)."""
    low = name.lower().replace("_", "-")
    negated = low.startswith(("no-", "non-", "without")) or "no-" in low
    if "helmet" in low or "hardhat" in low or "hard-hat" in low:
        return "no-helmet" if negated else "helmet"
    if "vest" in low:
        return "no-vest" if negated else "vest"
    if any(k in low for k in ("person", "worker", "people")):
        return "person"
    return "— (extra / out of scope)"


# ------------------------------------------------------------------------ splits


def split_stems(path: Path) -> set[str]:
    """Read a split list file and return the image stems it names."""
    return {
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines()
        if line.strip()
    }


def percentile(values: list[float], q: float) -> float:
    """Return the ``q``-th percentile (0–100) of ``values`` by nearest rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(round(q / 100.0 * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[idx]


# ------------------------------------------------------------------------ report


def audit_sh17(root: Path) -> list[str]:
    """Audit the SH17 root and return the markdown report body."""
    images, labels, voc = root / "images", root / "labels", root / "voc_labels"
    for required in (images, labels, voc):
        if not required.is_dir():
            raise FileNotFoundError(f"SH17 layout unexpected — missing {required}")

    label_files = sorted(labels.glob("*.txt"))
    image_stems = {p.stem for p in images.iterdir() if p.suffix.lower() in IMG_EXTS}
    label_stems = {p.stem for p in label_files}
    voc_stems = {p.stem for p in voc.glob("*.xml")}

    logger.info("SH17: scanning %d label files", len(label_files))
    scan = scan_labels(label_files)
    logger.info("SH17: deriving class names from VOC (geometry match)")
    names, votes, n_matched = derive_names_from_voc(labels, voc)

    train = split_stems(root / "train_files.txt")
    val = split_stems(root / "val_files.txt")

    out = [
        "# X01 / S1.1 — SH17 audit",
        "",
        f"- root: `{root}`",
        (
            f"- images: **{len(image_stems)}** · YOLO labels: **{len(label_stems)}** "
            f"· VOC xml: **{len(voc_stems)}**"
        ),
        (
            f"- class names recovered from VOC by geometry match over **{n_matched}** paired "
            "files (SH17 ships no names file)"
        ),
        "",
        "## Integrity",
        "",
        "| check | result |",
        "|---|---|",
        f"| images without a label | {len(image_stems - label_stems)} |",
        f"| labels without an image | {len(label_stems - image_stems)} |",
        f"| empty label files | {scan.n_empty} |",
        f"| malformed lines | {scan.n_malformed} |",
        f"| bbox coords outside [0,1] | {scan.n_out_of_range} |",
        f"| total instances | {scan.n_instances} |",
        (
            "| instances per image (min/median/max) | "
            f"{min(scan.instances_per_file, default=0)} / "
            f"{percentile([float(v) for v in scan.instances_per_file], 50):.0f} / "
            f"{max(scan.instances_per_file, default=0)} |"
        ),
        "",
        "## Splits (as shipped)",
        "",
        "| split | images | coverage issue |",
        "|---|---|---|",
        f"| train_files.txt | {len(train)} | {len(train - image_stems)} not in images/ |",
        f"| val_files.txt | {len(val)} | {len(val - image_stems)} not in images/ |",
        "| **test** | **0 — not shipped** | must be built + frozen (S1.4) |",
        f"| unassigned images | {len(image_stems - train - val)} | — |",
        f"| train∩val overlap | {len(train & val)} | must be 0 |",
        "",
        "## Classes (id → name, instances)",
        "",
        "| id | name | instances | share | vote purity | → core |",
        "|---|---|---|---|---|---|",
    ]
    total = scan.n_instances or 1
    for cls, count in sorted(scan.per_class.items(), key=lambda kv: -kv[1]):
        name = names.get(cls, "?")
        tally = votes.get(cls, Counter())
        purity = (tally.most_common(1)[0][1] / sum(tally.values()) * 100) if tally else 0.0
        out.append(
            f"| {cls} | {name} | {count} | {count / total * 100:.1f}% | {purity:.1f}% "
            f"| {suggest_core_mapping(name)} |"
        )
    out += [
        "",
        "## Violation route — evidence",
        "",
        (
            "SH17 has no explicit `no-helmet` / `no-vest` class; non-compliance must be "
            "inferred (head box without an overlapping helmet; person without vest) or "
            "sourced elsewhere (E5 set). The class table above is the evidence for that "
            "decision — see the S1.4 entry in the build plan."
        ),
        "",
    ]
    return out


def audit_chv(root: Path) -> list[str]:
    """Audit the CHV root and return the markdown report body."""
    inner = root / "CHV_dataset" if (root / "CHV_dataset").is_dir() else root
    images, annotations, splits = inner / "images", inner / "annotations", inner / "data split"
    for required in (images, annotations):
        if not required.is_dir():
            raise FileNotFoundError(f"CHV layout unexpected — missing {required}")

    label_files = sorted(annotations.glob("*.txt"))
    image_stems = {p.stem for p in images.iterdir() if p.suffix.lower() in IMG_EXTS}
    label_stems = {p.stem for p in label_files}

    logger.info("CHV: scanning %d label files", len(label_files))
    scan = scan_labels(label_files)
    readme = annotations / "README.md"
    names = names_from_readme(readme) if readme.exists() else {}

    out = [
        "# X01 / S1.1 — CHV audit",
        "",
        f"- root: `{inner}`" + ("  _(nested inside the download folder)_" if inner != root else ""),
        f"- images: **{len(image_stems)}** · YOLO labels: **{len(label_stems)}**",
        f"- class names: `annotations/README.md` ({len(names)} classes, shipped with the dataset)",
        "",
        "## Integrity",
        "",
        "| check | result |",
        "|---|---|",
        f"| images without a label | {len(image_stems - label_stems)} |",
        f"| labels without an image | {len(label_stems - image_stems)} |",
        f"| empty label files | {scan.n_empty} |",
        f"| malformed lines | {scan.n_malformed} |",
        f"| bbox coords outside [0,1] | {scan.n_out_of_range} |",
        f"| total instances | {scan.n_instances} |",
        (
            "| instances per image (min/median/max) | "
            f"{min(scan.instances_per_file, default=0)} / "
            f"{percentile([float(v) for v in scan.instances_per_file], 50):.0f} / "
            f"{max(scan.instances_per_file, default=0)} |"
        ),
        "",
    ]

    if splits.is_dir():
        out += [
            "## Splits (as shipped)",
            "",
            "| split | images | coverage issue |",
            "|---|---|---|",
        ]
        seen: dict[str, set[str]] = {}
        for name in ("train", "valid", "test"):
            path = splits / f"{name}.txt"
            if not path.exists():
                continue
            stems = split_stems(path)
            seen[name] = stems
            out.append(f"| {name} | {len(stems)} | {len(stems - image_stems)} not in images/ |")
        pairs = list(seen.items())
        overlaps = sum(len(a[1] & b[1]) for i, a in enumerate(pairs) for b in pairs[i + 1 :])
        covered = set().union(*seen.values()) if seen else set()
        out += [
            f"| **cross-split overlap** | {overlaps} | must be 0 |",
            f"| images in no split | {len(image_stems - covered)} | — |",
            "",
        ]

    out += [
        "## Classes (id → name, instances)",
        "",
        "| id | name | instances | share | → core |",
        "|---|---|---|---|---|",
    ]
    total = scan.n_instances or 1
    for cls, count in sorted(scan.per_class.items(), key=lambda kv: -kv[1]):
        name = names.get(cls, "?")
        out.append(
            f"| {cls} | {name} | {count} | {count / total * 100:.1f}% "
            f"| {suggest_core_mapping(name)} |"
        )
    out.append("")
    return out


def load_roots() -> dict[str, Path]:
    """Read dataset roots from ``configs/base.yaml`` (single source of truth)."""
    doc = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    datasets = doc.get("datasets", {})
    return {
        "sh17": Path(datasets["sh17_root"]),
        "chv": Path(datasets["chv_root"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="X01 / S1.1 dataset label audit.")
    parser.add_argument("--dataset", required=True, choices=("sh17", "chv"))
    parser.add_argument("--out", default=None, help="markdown report path")
    args = parser.parse_args()

    roots = load_roots()
    root = roots[args.dataset]
    if not root.is_dir():
        logger.error("dataset root not found: %s", root)
        return 1

    body = audit_sh17(root) if args.dataset == "sh17" else audit_chv(root)
    report = "\n".join(body)
    print(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        logger.info("report written: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
