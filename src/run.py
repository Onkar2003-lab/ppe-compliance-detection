"""The training + evaluation entry point (S2 onward).

One run = one config = one run-ID. This module is the only place a model is trained, so the
reproducibility contract is enforced in one place rather than remembered in several: the
Pictor guard runs before training starts, the seed is set from the config, and the exact
versions and git commit are written beside the results. Nothing here decides *what* to run —
that comes from the config file, which is frozen per run-ID and never edited afterwards.

The cross-evaluation is the point of the whole project, so it is part of the standard run
rather than a later script: every run reports in-domain **and** on the other dataset.

Evaluation uses the **val** splits. The frozen `test` splits stay untouched until final
scoring (S5), so nothing that gets iterated on can quietly tune against them.

Usage::

    python -m src.run --run-id X02-smoke-yolov8n-s0 --subset 200 --epochs 3   # smoke slice
    python -m src.run --config configs/X03-yolov8n-s0.yaml                    # a real run
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from src.guards import assert_training_data_excludes_pictor
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/base.yaml")
DATA_DIR = Path("configs/data")
# Which dataset each training set is cross-evaluated against (the transfer study, both ways).
CROSS_TARGET = {"sh17": "chv", "chv": "sh17"}
EVAL_SPLIT = "val"


def load_config(path: Path, overrides: dict[str, Any]) -> dict:
    """Read a run config and apply command-line overrides (which are logged, not silent)."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if value is not None and config.get(key) != value:
            logger.info("config override: %s = %r (was %r)", key, value, config.get(key))
            config[key] = value
    return config


def git_commit() -> str:
    """Current commit SHA, so a result can be traced to the code that produced it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment() -> dict[str, str]:
    """Versions that change results — recorded with every run, not assumed stable."""
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
        "ultralytics": ultralytics.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": git_commit(),
    }


def dataset_of(data_yaml: str | Path) -> str:
    """Dataset name behind a data YAML (`configs/data/sh17.yaml` -> `sh17`)."""
    return Path(data_yaml).stem


def subset_yaml(data_yaml: Path, limit: int, out_dir: Path) -> Path:
    """Write a data YAML whose train list is truncated to ``limit`` images.

    Used by the walking skeleton only: it exercises the real loader and the real label space
    on a slice small enough to fail fast. The val list is left whole so the evaluation path is
    the same one the full runs use.
    """
    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(document["path"])
    lines = (root / document["train"]).read_text(encoding="utf-8").splitlines()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_list = out_dir / f"train-subset-{limit}.txt"
    train_list.write_text("\n".join(lines[:limit]), encoding="utf-8")
    document["train"] = str(train_list.resolve())

    path = out_dir / f"data-subset-{data_yaml.stem}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    logger.info("subset: %d of %d train images -> %s", len(lines[:limit]), len(lines), path)
    return path


def metrics_of(results: Any) -> dict[str, float]:
    """The accuracy-axis numbers, pulled out of an Ultralytics results object."""
    box = results.box
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50_per_class": [float(v) for v in box.ap50],
    }


def train(config: dict, data_yaml: Path, run_dir: Path) -> tuple[Path, Path]:
    """Train one model under the frozen recipe.

    Returns ``(save_dir, weights)`` read back from the trainer rather than assumed: if the
    run-ID directory already exists, Ultralytics writes to an incremented one, and silently
    reporting results from the wrong folder is exactly the kind of error a run-ID contract is
    supposed to prevent.
    """
    from ultralytics import YOLO

    model = YOLO(config["model"])
    model.train(
        data=str(data_yaml.resolve()),
        epochs=config["epochs"],
        patience=config["patience"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        amp=config["amp"],
        cache=config["cache"],
        workers=config["workers"],
        pretrained=config["pretrained"],
        optimizer=config["optimizer"],
        device=config["device"],
        seed=config["seed"],
        deterministic=True,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=config.get("exist_ok", False),
        plots=True,
    )
    save_dir = Path(model.trainer.save_dir)
    if save_dir.resolve() != run_dir.resolve():
        logger.warning(
            "Ultralytics wrote to %s, not %s — the run-ID directory already existed. "
            "Results below come from the directory actually written.",
            save_dir,
            run_dir,
        )
    weights = save_dir / "weights" / "best.pt"
    logger.info("weights: %s", weights)
    return save_dir, weights


def evaluate(weights: Path, data_yaml: Path, run_dir: Path, config: dict, tag: str) -> dict:
    """Score trained weights against one dataset's validation split."""
    from ultralytics import YOLO

    logger.info("evaluating (%s) on %s", tag, data_yaml.name)
    results = YOLO(str(weights)).val(
        data=str(data_yaml.resolve()),
        split=EVAL_SPLIT,
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        project=str(run_dir),
        name=f"eval-{tag}",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    scores = metrics_of(results)
    logger.info("%s: mAP50 %.4f · mAP50-95 %.4f", tag, scores["map50"], scores["map50_95"])
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Train + evaluate one run.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default=None, help="overrides run_id/name in the config")
    parser.add_argument("--data", type=Path, default=None, help="training dataset YAML")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None, help="smoke slice: N train images")
    args = parser.parse_args()

    config = load_config(
        args.config,
        {
            "run_id": args.run_id,
            "name": args.run_id,
            "epochs": args.epochs,
            "batch": args.batch,
            "seed": args.seed,
            "data": str(args.data) if args.data else None,
        },
    )

    # Supervisor condition, enforced before a single batch is loaded.
    assert_training_data_excludes_pictor(config, config["datasets"].get("pictor_root"))

    run_id = config["run_id"]
    run_dir = Path(config["project"]) / run_id
    data_yaml = Path(config["data"])
    trained_on = dataset_of(data_yaml)
    cross_yaml = DATA_DIR / f"{CROSS_TARGET[trained_on]}.yaml"

    env = environment()
    logger.info("run %s | train on %s | cross-eval on %s", run_id, trained_on, cross_yaml.stem)
    logger.info("env: %s", env)

    # Written outside the run directory: creating it here would make Ultralytics think the
    # run-ID was already used and quietly write to an incremented folder instead.
    train_yaml = (
        subset_yaml(data_yaml, args.subset, Path(config["project"]) / "_subsets" / run_id)
        if args.subset
        else data_yaml
    )

    started = time.time()
    run_dir, weights = train(config, train_yaml, run_dir)
    train_seconds = time.time() - started
    logger.info("training wall-clock: %.1f s", train_seconds)

    if not weights.exists():
        logger.error("no weights produced at %s", weights)
        return 1

    summary = {
        "run_id": run_id,
        "trained_on": trained_on,
        "subset": args.subset,
        "seed": config["seed"],
        "model": config["model"],
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "eval_split": EVAL_SPLIT,
        "train_seconds": round(train_seconds, 1),
        "environment": env,
        "in_domain": evaluate(weights, data_yaml, run_dir, config, "in-domain"),
        "cross_domain": evaluate(weights, cross_yaml, run_dir, config, "cross-domain"),
        "weights": str(weights),
    }
    summary["transfer_delta_map50"] = round(
        summary["cross_domain"]["map50"] - summary["in_domain"]["map50"], 4
    )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "in-domain mAP50 %.4f -> cross-domain %.4f (delta %+.4f) | summary: %s",
        summary["in_domain"]["map50"],
        summary["cross_domain"]["map50"],
        summary["transfer_delta_map50"],
        run_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
