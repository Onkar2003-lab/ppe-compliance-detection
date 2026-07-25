"""X01 — SH17 / CHV label-file audit + harmonisation map.

First coding task of the project. DESCRIPTIVE only — no training. It auto-discovers each
dataset's structure (data.yaml, images/labels dirs, train/val/test splits), parses the
YOLO-format label files, and reports what is actually there so we can:
  1. close assumption A1 (do SH17 & CHV share helmet / vest / person?),
  2. decide the violation-recall route (how SH17 encodes non-compliance / worn-state),
  3. draft the O1 harmonisation mapping to {helmet, no-helmet, vest, no-vest, person}.

The script assumes NOTHING about folder layout — it discovers and reports. Run it once per
dataset root; it prints a summary and writes a markdown report.

Usage (PANDORA, after data is at D:\\datasets\\):
    python src/audit_labels.py --name SH17 --root "D:/datasets/SH17" --out "D:/runs/X01-audit/sh17.md"
    python src/audit_labels.py --name CHV  --root "D:/datasets/CHV"  --out "D:/runs/X01-audit/chv.md"

Requires: pyyaml (in requirements.txt). No GPU / torch needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

CORE = ("helmet", "no-helmet", "vest", "no-vest", "person")  # O1 target label space
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_data_yaml(root: Path) -> Path | None:
    """Return the most likely dataset yaml (data.yaml / *.yaml with a 'names' key)."""
    candidates = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    for p in candidates:
        if yaml is None:
            return p
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if isinstance(doc, dict) and "names" in doc:
            return p
    return candidates[0] if candidates else None


def load_class_names(data_yaml: Path | None) -> dict[int, str]:
    """Parse class-id -> name from a YOLO data.yaml ('names' as list or dict)."""
    if data_yaml is None or yaml is None:
        return {}
    try:
        doc = yaml.safe_load(data_yaml.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    names = doc.get("names") if isinstance(doc, dict) else None
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def discover_label_files(root: Path) -> dict[str, list[Path]]:
    """Group *.txt label files by an inferred split (train/val/test/unknown)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*.txt"):
        low = str(p).lower()
        if any(skip in p.name.lower() for skip in ("readme", "requirements", "classes")):
            continue
        split = next((s for s in ("train", "valid", "val", "test") if s in low), "unknown")
        split = "val" if split == "valid" else split
        groups[split].append(p)
    return groups


def scan_labels(files: list[Path]) -> dict:
    """Scan YOLO label files: per-class instances, malformed lines, bbox-range check."""
    per_class = Counter()
    malformed = 0
    out_of_range = 0
    n_label_files = 0
    n_empty = 0
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        n_label_files += 1
        if not any(l.strip() for l in lines):
            n_empty += 1
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                malformed += 1
                continue
            try:
                cls = int(float(parts[0]))
                coords = [float(x) for x in parts[1:5]]
            except ValueError:
                malformed += 1
                continue
            per_class[cls] += 1
            if any(c < 0.0 or c > 1.0 for c in coords):
                out_of_range += 1
    return {
        "per_class": per_class,
        "malformed": malformed,
        "out_of_range": out_of_range,
        "n_label_files": n_label_files,
        "n_empty": n_empty,
    }


def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def suggest_harmonisation(names: dict[int, str]) -> list[str]:
    """Heuristic mapping of found class names onto the CORE target space (for review)."""
    rows = []
    for idx, name in sorted(names.items()):
        n = name.lower()
        if "no" in n and "helmet" in n or "no_helmet" in n or "no-hardhat" in n:
            tgt = "no-helmet"
        elif "helmet" in n or "hardhat" in n or "hat" in n:
            tgt = "helmet"
        elif "no" in n and "vest" in n or "no-safety-vest" in n:
            tgt = "no-vest"
        elif "vest" in n:
            tgt = "vest"
        elif "person" in n or "worker" in n or "people" in n:
            tgt = "person"
        else:
            tgt = "— (extra / out of scope — review)"
        rows.append(f"| {idx} | {name} | {tgt} |")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="X01 dataset label audit (YOLO format).")
    ap.add_argument("--name", required=True, help="dataset label, e.g. SH17 or CHV")
    ap.add_argument("--root", required=True, help="dataset root folder")
    ap.add_argument("--out", default=None, help="markdown report path (optional)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[FAIL] root not found: {root}")
        return 1
    if yaml is None:
        print("[WARN] pyyaml not installed — class names from data.yaml will be skipped. "
              "pip install pyyaml")

    data_yaml = find_data_yaml(root)
    names = load_class_names(data_yaml)
    groups = discover_label_files(root)
    n_images = count_images(root)

    lines: list[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"# X01 audit — {args.name}")
    emit(f"- root: `{root}`")
    emit(f"- data.yaml: `{data_yaml}`" if data_yaml else "- data.yaml: (none found)")
    emit(f"- images found (recursive): {n_images}")
    emit(f"- class names ({len(names)}): {names if names else '(infer from label indices)'}")
    emit("")

    total = Counter()
    emit("## Per-split label scan")
    emit("| split | label files | empty | instances | malformed lines | bbox out-of-[0,1] |")
    emit("|---|---|---|---|---|---|")
    for split, files in sorted(groups.items()):
        r = scan_labels(files)
        total.update(r["per_class"])
        emit(f"| {split} | {r['n_label_files']} | {r['n_empty']} | "
             f"{sum(r['per_class'].values())} | {r['malformed']} | {r['out_of_range']} |")
    emit("")

    emit("## Instances per class (all splits)")
    emit("| class id | name | instances |")
    emit("|---|---|---|")
    for cls, cnt in sorted(total.items()):
        emit(f"| {cls} | {names.get(cls, '?')} | {cnt} |")
    emit("")

    emit("## Suggested harmonisation → {helmet, no-helmet, vest, no-vest, person}")
    emit("_Heuristic — REVIEW before committing to the O1 mapping table._")
    emit("| class id | dataset name | suggested core target |")
    emit("|---|---|---|")
    if names:
        for row in suggest_harmonisation(names):
            emit(row)
    else:
        emit("| — | (no data.yaml names) | map from instance-count table above |")
    emit("")

    core_present = [c for c in ("helmet", "vest", "person")
                    if any(c in n.lower() or (c == "helmet" and "hat" in n.lower())
                           or (c == "person" and "worker" in n.lower()) for n in names.values())]
    emit("## A1 check + notes")
    emit(f"- core classes detected by name: {core_present or '(none by name — check counts)'}")
    emit("- **Worn-state / violation encoding:** inspect the class list above — does the set "
         "include explicit no-helmet / no-vest, or on/off worn-state variants? Record which, "
         "to fix the violation-recall route (SH17 proxy vs adopt E5 set).")
    emit("- **Parent-set overlap (manual):** cannot be verified from labels alone; SH17 & CHV "
         "both trace to SHWD/GDUT-HWD — spot-check filenames/hashes before claiming zero-shot.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[OK] report written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
