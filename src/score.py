"""The accuracy and cross-dataset axes, scored on the frozen test splits (S5).

Two of the four deployment-readiness axes come from here: **accuracy** (how good is the
model on data from its own dataset) and **cross-dataset generalisation** (what survives when
it meets the other dataset). The second is the study's contribution, so both are produced by
one pass over the same weights — a model's in-domain and transfer scores must never come
from different evaluation settings.

**Why this is a separate module from `src.run`, rather than a flag on it.** Training runs
evaluate on **val**, deliberately: val drives early stopping and `best.pt` selection, so
scoring the headline on it would report a number the training loop was allowed to optimise
against. The frozen `test` splits were built at S1.4 with content IDs and no training run has
ever read them. Editing `src.run`'s split constant now would also make it misdescribe the 18
completed runs, whose `summary.json` files record val numbers and are already on disk. So the
training entry point keeps telling the truth about what it did, and final scoring lives here
(decision: user, 2026-08-06).

The test splits carry the exclusion of the SH17<->CHV near-duplicates (X01/S1.4 F10,
residual leakage 0), so a cross-dataset score cannot be inflated by an image the model met
during training.

Nothing here aggregates or compares. Per-run numbers are raw material: they become results
only after seed aggregation with a 95 % BCa CI and the named test, and no "X beats Y"
may be written from this module's output.

Usage::

    python -m src.score --limit 1              # one run, to check the wiring
    python -m src.score                        # all 18 runs on the frozen test splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.run import DATA_DIR, cross_yaml_for, dataset_of, metrics_of
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RUNS = Path("D:/runs")
DEFAULT_OUT = Path("D:/runs/X05-accuracy")
SPLIT = "test"  # the whole point of this module — see the docstring
CLASSES = ("person", "helmet", "vest")


def frozen_config(run_dir: Path, configs: Path) -> dict:
    """The settings a run was trained under, preferring the run's own written record.

    ``summary.json`` is what the run itself reported; the frozen config is the contract it
    was launched under. They should agree, and a mismatch is worth seeing rather than
    silently resolving, so the fallback is logged.
    """
    summary = run_dir / "summary.json"
    if summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))

    candidate = configs / f"{run_dir.name}.yaml"
    if candidate.is_file():
        logger.warning("%s: no summary.json — falling back to %s", run_dir.name, candidate)
        config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        return {
            "run_id": run_dir.name,
            "imgsz": config["imgsz"],
            "batch": config["batch"],
            "seed": config["seed"],
            "model": config["model"],
            "trained_on": dataset_of(config["data"]),
        }
    raise FileNotFoundError(f"{run_dir.name}: neither summary.json nor a frozen config")


def data_yaml_for(dataset: str, variant: str = "-640") -> Path:
    """The data YAML a run should be scored against, at the resolution it was trained on."""
    path = DATA_DIR / f"{dataset}{variant}.yaml"
    return path if path.is_file() else DATA_DIR / f"{dataset}.yaml"


def evaluate_on_test(weights: Path, data_yaml: Path, out_dir: Path, config: dict, tag: str) -> dict:
    """Score one set of weights against one dataset's frozen test split."""
    from ultralytics import YOLO

    logger.info("  %s on %s (%s split)", tag, data_yaml.stem, SPLIT)
    results = YOLO(str(weights)).val(
        data=str(data_yaml.resolve()),
        split=SPLIT,
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=0,
        project=str(out_dir),
        name=f"test-{tag}",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    scores = metrics_of(results)
    scores["per_class_map50"] = dict(zip(CLASSES, scores.pop("map50_per_class")))
    return scores


def score_run(run_dir: Path, out: Path, configs: Path) -> dict:
    """Both axes for one run: in-domain and cross-domain, on test, in one pass."""
    weights = run_dir / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)

    config = frozen_config(run_dir, configs)
    trained_on = config["trained_on"]
    in_yaml = data_yaml_for(trained_on)
    cross_yaml = cross_yaml_for(in_yaml)

    logger.info("%s (trained on %s, seed %s)", run_dir.name, trained_on, config.get("seed"))
    in_domain = evaluate_on_test(weights, in_yaml, out / run_dir.name, config, "in-domain")
    cross = evaluate_on_test(weights, cross_yaml, out / run_dir.name, config, "cross-domain")

    record = {
        "run_id": config.get("run_id", run_dir.name),
        "model": config.get("model"),
        "seed": config.get("seed"),
        "trained_on": trained_on,
        "tested_on": {"in_domain": in_yaml.stem, "cross_domain": cross_yaml.stem},
        "eval_split": SPLIT,
        "in_domain": in_domain,
        "cross_domain": cross,
        "transfer_delta_map50": round(cross["map50"] - in_domain["map50"], 4),
        "transfer_delta_map50_95": round(cross["map50_95"] - in_domain["map50_95"], 4),
        "weights": str(weights),
    }
    logger.info(
        "  mAP50 in %.4f -> cross %.4f (delta %+.4f)",
        in_domain["map50"],
        cross["map50"],
        record["transfer_delta_map50"],
    )
    return record


def build_report(records: list[dict], out: Path) -> Path:
    """Per-run table, grouped by direction. Deliberately no ranking and no comparison."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "X05-accuracy-per-run.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    lines = [
        "# X05 — accuracy + cross-dataset axes (frozen TEST splits)",
        "",
        "Per-run numbers, scored on the held-out test partitions that no training run has",
        "read. **Not results yet:** nothing here is a finding until seeds are aggregated with",
        "a 95 % BCa CI and the named test. No 'X beats Y' from this table.",
        "",
        "| run | trained on | seed | in-domain mAP50 | cross mAP50 | delta | in mAP50-95 | cross mAP50-95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: (r["trained_on"], str(r["model"]), r["seed"] or 0)):
        lines.append(
            f"| {r['run_id']} | {r['trained_on']} | {r['seed']} "
            f"| {r['in_domain']['map50']:.4f} | {r['cross_domain']['map50']:.4f} "
            f"| {r['transfer_delta_map50']:+.4f} "
            f"| {r['in_domain']['map50_95']:.4f} | {r['cross_domain']['map50_95']:.4f} |"
        )

    lines += [
        "",
        "## Per-class mAP50, in-domain (vest support is thin — F1)",
        "",
        "| run | person | helmet | vest |",
        "|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: (r["trained_on"], str(r["model"]), r["seed"] or 0)):
        per = r["in_domain"]["per_class_map50"]
        lines.append(
            f"| {r['run_id']} | {per['person']:.4f} | {per['helmet']:.4f} | {per['vest']:.4f} |"
        )

    path = out / "X05-accuracy-summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("report: %s", path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Score accuracy + transfer on the test splits.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--pattern", default="X04-*")
    parser.add_argument("--limit", type=int, default=None, help="first N runs (wiring check)")
    args = parser.parse_args()

    run_dirs = sorted(
        d for d in args.runs.glob(args.pattern) if (d / "weights" / "best.pt").is_file()
    )
    if not run_dirs:
        logger.error("no runs matching %s under %s", args.pattern, args.runs)
        return 1
    if args.limit:
        run_dirs = run_dirs[: args.limit]

    logger.info("scoring %d runs on the frozen %s splits", len(run_dirs), SPLIT)
    records = [score_run(d, args.out, args.configs) for d in run_dirs]
    build_report(records, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
