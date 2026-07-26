"""X01 / S1.1b — audit of Pictor-PPE, the evaluation-only violation set.

Pictor-PPE (Nath, Behzadan & Paal 2020) is the only dataset in this project that labels
**compliance** rather than objects: each worker is tagged with what they are wearing. It is
therefore the ground truth for the violation-recall axis — and it is **evaluation-only** by
supervisor condition, so it never enters a training config (enforced in :mod:`src.guards`).

The audit answers what S1.1 answered for SH17 and CHV, plus two questions unique to this set:

1. **Structure + integrity** — what is on disk, do labels and images agree, are the boxes valid.
2. **Class encoding, recovered rather than assumed** — the release ships no names file. The
   three label sets correspond to the paper's three approaches; the mapping is *derived* by
   cross-tabulating which PPE boxes fall inside which worker box against the compliance class
   on the same worker, so the decoding rests on the data rather than on a guess.
3. **Contamination** — a dHash screen against SH17 and CHV, because the violation numbers are
   claimed zero-shot too.

⚠️ **The containment figure this audit reports is a *decoding check*, not a calibration.** The
association rule's operative threshold is calibrated on SH17/CHV (decision 2026-07-26);
tuning it on Pictor would be fitting the evaluation set.

Usage::

    python -m src.audit_pictor --out D:/runs/X01-audit-pictor
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.audit_labels import load_roots
from src.overlap import matches_within
from src.utils.logging import get_logger

logger = get_logger(__name__)

APPROACHES = ("01", "02", "03")
SPLITS = ("train", "valid", "test")
LABEL_PATTERN = "pictor_ppe_crowdsourced_approach-{approach}_{split}.txt"

# Recovered by cross-tabulation (see `verify_class_mapping`), not taken on trust.
A1_NAMES = {0: "hat", 1: "vest", 2: "worker"}
A2_NAMES = {0: "W (no PPE)", 1: "WH (hat)", 2: "WV (vest)", 3: "WHV (hat+vest)"}
# The evaluation target: each compliance class decoded into the two yes/no questions the
# association rule has to answer for a person.
COMPLIANCE_DECODING = {
    0: {"helmet": False, "vest": False},
    1: {"helmet": True, "vest": False},
    2: {"helmet": False, "vest": True},
    3: {"helmet": True, "vest": True},
}
CONTAINMENT_PROBE = 0.5  # decoding probe only — NOT the operative association threshold

Box = tuple[int, int, int, int, int]


@dataclass
class Labels:
    """Parsed label files for one approach: image name → boxes, keyed also by split."""

    by_image: dict[str, list[Box]] = field(default_factory=dict)
    split_of: dict[str, str] = field(default_factory=dict)
    malformed: list[str] = field(default_factory=list)


def parse_labels(path: Path, split: str, into: Labels) -> None:
    """Parse one ``image.jpg\\tx1,y1,x2,y2,cls\\t…`` label file into ``into``."""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        name, boxes = fields[0], []
        for chunk in fields[1:]:
            if not chunk.strip():
                continue
            parts = chunk.split(",")
            if len(parts) != 5 or not all(p.strip().lstrip("-").isdigit() for p in parts):
                into.malformed.append(f"{path.name}:{number}: {chunk!r}")
                continue
            boxes.append(tuple(int(p) for p in parts))
        into.by_image[name] = boxes
        into.split_of[name] = split


def load_approach(root: Path, approach: str) -> Labels:
    """Load all three splits of one approach's labels."""
    labels = Labels()
    for split in SPLITS:
        path = root / "Labels" / LABEL_PATTERN.format(approach=approach, split=split)
        if not path.is_file():
            raise FileNotFoundError(f"missing label file: {path}")
        parse_labels(path, split, labels)
    return labels


# ------------------------------------------------------------------ class-map recovery


def containment(inner: Box, outer: Box) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``.

    Containment, not IoU: a hat box is a rounding error beside a full-body worker box, so
    IoU would be near zero even for a perfectly worn hat.
    """
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ((x2 - x1) * (y2 - y1)) / area if area > 0 else 0.0


def verify_class_mapping(a1: Labels, a2: Labels) -> tuple[Counter, int, int]:
    """Cross-tabulate compliance class against which PPE boxes sit inside the worker box.

    If the assumed encoding is right, class 1 workers contain a hat and no vest, class 2 a
    vest and no hat, class 3 both, and class 0 neither — with the geometry agreeing on
    (nearly) every instance.

    Returns:
        ``(table, agreements, total)`` where ``table`` maps
        ``(compliance_class, has_hat, has_vest)`` to a count.
    """
    table: Counter = Counter()
    agreements = total = 0
    for image, workers in a2.by_image.items():
        ppe = [b for b in a1.by_image.get(image, []) if b[4] != 2]
        for worker in workers:
            has = {0: False, 1: False}
            for box in ppe:
                if containment(box, worker) >= CONTAINMENT_PROBE:
                    has[box[4]] = True
            table[(worker[4], has[0], has[1])] += 1
            expected = COMPLIANCE_DECODING.get(worker[4], {})
            total += 1
            if has[0] == expected.get("helmet") and has[1] == expected.get("vest"):
                agreements += 1
    return table, agreements, total


# ------------------------------------------------------------------------- integrity


@dataclass
class Integrity:
    """Findings from checking labels against the images on disk."""

    on_disk: int = 0
    labelled: int = 0
    missing_images: list[str] = field(default_factory=list)
    unlabelled_images: list[str] = field(default_factory=list)
    byte_duplicates: dict[str, list[str]] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    bad_geometry: list[str] = field(default_factory=list)
    out_of_bounds: list[str] = field(default_factory=list)


def check_integrity(root: Path, labels: Labels) -> tuple[Integrity, dict[str, int]]:
    """Validate images and boxes; also return a dHash per labelled image."""
    import hashlib

    from src.eda import dhash

    images = root / "Images"
    on_disk = {p.name: p for p in sorted(images.glob("*.jpg"))}
    report = Integrity(on_disk=len(on_disk), labelled=len(labels.by_image))
    report.missing_images = sorted(set(labels.by_image) - set(on_disk))
    report.unlabelled_images = sorted(set(on_disk) - set(labels.by_image))

    digests: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, int] = {}
    for name, path in on_disk.items():
        digests[hashlib.md5(path.read_bytes()).hexdigest()].append(name)
        if name not in labels.by_image:
            continue
        try:
            with Image.open(path) as img:
                width, height = img.width, img.height
                img.draft("RGB", (img.width // 8, img.height // 8))
                grey = np.asarray(img.convert("RGB"), dtype=np.uint8).mean(axis=2)
        except OSError as exc:
            report.unreadable.append(f"{name} ({exc})")
            continue
        hashes[Path(name).stem] = dhash(grey.astype(np.uint8))

        boxes = labels.by_image[name]
        if not boxes:
            report.empty_labels.append(name)
        for x1, y1, x2, y2, _cls in boxes:
            if x2 <= x1 or y2 <= y1:
                report.bad_geometry.append(f"{name}: {(x1, y1, x2, y2)}")
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                report.out_of_bounds.append(f"{name}: {(x1, y1, x2, y2)} in {width}x{height}")
    report.byte_duplicates = {k: v for k, v in digests.items() if len(v) > 1}
    return report, hashes


# ---------------------------------------------------------------------------- report


def build_report(
    root: Path,
    labels: dict[str, Labels],
    integrity: Integrity,
    mapping: tuple[Counter, int, int],
    overlaps: dict[str, list[tuple[str, str, int]]],
) -> str:
    """Assemble the markdown audit report (numbers graduate to the ledger, not from here)."""
    table, agreements, total = mapping
    a1, a2 = labels["01"], labels["02"]
    counts = {
        a: Counter(b[4] for boxes in labels[a].by_image.values() for b in boxes) for a in APPROACHES
    }
    split_sizes = Counter(a2.split_of.values())

    out = [
        "# X01 / S1.1b — Pictor-PPE audit (evaluation-only violation set)",
        "",
        f"Root: `{root}`",
        "",
        "## 1. What is on disk",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Images on disk | {integrity.on_disk} |",
        f"| Images referenced by labels | {integrity.labelled} |",
        f"| Splits (train/valid/test) | {split_sizes['train']} / {split_sizes['valid']} / {split_sizes['test']} |",
        f"| Worker instances (approach-03) | {sum(counts['03'].values())} |",
        "| Label sets | 3 (the paper's approaches 1-3) |",
        "",
        "## 2. Integrity",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Labelled images missing from disk | {len(integrity.missing_images)} |",
        f"| Images on disk with no label | {len(integrity.unlabelled_images)} |",
        f"| Byte-identical duplicate files | {len(integrity.byte_duplicates)} group(s) |",
        f"| Unreadable images | {len(integrity.unreadable)} |",
        f"| Empty label rows | {len(integrity.empty_labels)} |",
        f"| Malformed label entries | {len(a1.malformed) + len(a2.malformed)} |",
        f"| Boxes with x2<=x1 or y2<=y1 | {len(integrity.bad_geometry)} |",
        f"| Boxes outside the image | {len(integrity.out_of_bounds)} |",
        "",
    ]
    if integrity.unlabelled_images:
        out += ["Unlabelled files:", ""]
        out += [f"- `{name}`" for name in integrity.unlabelled_images[:15]]
        out += [""]
    if integrity.byte_duplicates:
        out += ["Byte-identical groups:", ""]
        out += [
            f"- {' == '.join(f'`{n}`' for n in group)}"
            for group in integrity.byte_duplicates.values()
        ]
        out += [""]

    out += [
        "## 3. Class encoding (recovered, not assumed)",
        "",
        "Approach-01 (objects) and approach-02 (per-worker compliance) label the same images.",
        "Cross-tabulating which approach-01 PPE boxes fall inside each approach-02 worker box",
        f"(containment >= {CONTAINMENT_PROBE}) against that worker's compliance class recovers the",
        "encoding without a names file:",
        "",
        "| compliance class | hat box inside | vest box inside | workers |",
        "|---|---|---|---|",
    ]
    for (cls, has_hat, has_vest), count in sorted(table.items()):
        out.append(
            f"| {cls} | {'yes' if has_hat else 'no'} | {'yes' if has_vest else 'no'} | {count} |"
        )
    out += [
        "",
        (
            f"**Geometry agrees with the labels on {agreements}/{total} workers "
            f"({agreements / total * 100:.1f}%).**"
        ),
        "",
        "Recovered mapping:",
        "",
        "| set | class ids |",
        "|---|---|",
        f"| approach-01 (objects) | {A1_NAMES} |",
        f"| approach-02 (compliance) | {A2_NAMES} |",
        "| approach-03 (worker only) | {0: 'worker'} |",
        "",
        "> ⚠️ The containment probe above **decodes** the labels. It is **not** a calibration of",
        "> the association rule: that threshold is fitted on SH17/CHV, never on this set.",
        "",
        "## 4. Class distribution",
        "",
        "| set | counts |",
        "|---|---|",
        f"| approach-01 | {dict(sorted(counts['01'].items()))} ({A1_NAMES}) |",
        f"| approach-02 | {dict(sorted(counts['02'].items()))} ({A2_NAMES}) |",
        f"| approach-03 | {dict(sorted(counts['03'].items()))} |",
        "",
        "## 5. Evaluation-target decoding",
        "",
        "| compliance class | helmet worn | vest worn |",
        "|---|---|---|",
    ]
    for cls, state in COMPLIANCE_DECODING.items():
        out.append(
            f"| {cls} — {A2_NAMES[cls]} | {'yes' if state['helmet'] else 'NO (violation)'} "
            f"| {'yes' if state['vest'] else 'NO (violation)'} |"
        )
    out += [
        "",
        "## 6. Contamination screen (dHash vs the training sets)",
        "",
        "| against | pairs at distance <= 5 |",
        "|---|---|",
    ]
    for name, matches in overlaps.items():
        out.append(f"| {name} | {len(matches)} |")
    out += [""]
    for name, matches in overlaps.items():
        for query, reference, distance in matches[:10]:
            out.append(f"- `{query}` ~ {name} `{reference}` (d={distance})")
    out += ["", "_Generated by `python -m src.audit_pictor`._"]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="X01 / S1.1b Pictor-PPE audit.")
    parser.add_argument("--out", default="D:/runs/X01-audit-pictor")
    parser.add_argument("--cache", type=Path, default=Path("D:/runs/X01-eda/stats-cache.json"))
    args = parser.parse_args()

    roots = load_roots()
    if "pictor" not in roots:
        logger.error("pictor_root is not set in configs/base.yaml")
        return 1
    root = roots["pictor"]
    if not root.is_dir():
        logger.error("Pictor root not found: %s", root)
        return 1

    labels = {approach: load_approach(root, approach) for approach in APPROACHES}
    logger.info("labels parsed: %d images", len(labels["02"].by_image))
    integrity, hashes = check_integrity(root, labels["02"])
    logger.info("integrity checked: %d images hashed", len(hashes))
    mapping = verify_class_mapping(labels["01"], labels["02"])
    logger.info("class mapping: %d/%d workers agree with the geometry", mapping[1], mapping[2])

    overlaps: dict[str, list[tuple[str, str, int]]] = {}
    from src.overlap import load_hashes

    try:
        others = load_hashes(args.cache)
        for name in ("sh17", "chv"):
            overlaps[name.upper()] = matches_within(hashes, others[name], threshold=5)
            logger.info("overlap vs %s: %d pairs", name.upper(), len(overlaps[name.upper()]))
    except FileNotFoundError:
        logger.warning("dHash cache absent — skipping the contamination screen")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(root, labels, integrity, mapping, overlaps)
    (out / "audit-pictor.md").write_text(report, encoding="utf-8")
    logger.info("report written: %s", out / "audit-pictor.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
