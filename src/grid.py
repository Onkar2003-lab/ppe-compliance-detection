"""X04 / S4 — the training grid: lock the run set, then execute it as a resumable queue.

**The grid.** 3 model families x 3 seeds x 2 transfer directions = **18 runs**, nano tier only
(supervisor-approved 2026-07-28: the `small` variants were cut, justified by the edge-deployment
focus; the consequence — that the ranking is established at nano capacity only — is a stated
Discussion limitation, not a silent one).

**Ordering is seed-major, and that is the important design choice.** One *block* = all 3 models
x both directions at a single seed = 6 runs. Blocks run seed 0 -> 1 -> 2. After **every** block
there is a complete, comparable picture of the whole comparison at one more seed, instead of one
finished model and two missing. The comparison is the contribution; the seeds are the confidence
around it. If time runs out at the end, what is lost is CI width, not the study.

**One stopping protocol for every run: 200 epochs, patience 50.** No fixed epoch cap is baked in.
A cap applied to some runs and not others would make the seeds incomparable — the multi-seed
spread would then measure a protocol difference and report it as variance, which is exactly the
number the confidence intervals are supposed to mean. Patience-50 early stopping is itself the
evidence-based stop, applied identically everywhere; the epoch each run actually stopped at is
recorded, and those curves are the convergence documentation the supervisor's approval is
conditional on. If wall-clock forces a cap later it must be applied uniformly and re-run, not
retrofitted to part of the grid.

**Queue guarantees** (an overnight session must survive being interrupted):
- a config is **frozen before** its run starts — the run-ID contract, never edited afterwards;
- a run whose ``summary.json`` exists is **skipped**, so re-invoking is safe and idempotent;
- a run with a checkpoint but no summary is **resumed** from ``last.pt``;
- a failure is **logged and the queue continues** — one bad run must not waste a whole session.

Usage::

    python -m src.grid lock                  # freeze all 18 configs, print the ledger table
    python -m src.grid run                   # work the whole queue, in order
    python -m src.grid run --seeds 0         # just the seed-0 block (6 runs)
    python -m src.grid status                # what is done, what is left
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

EXPERIMENT = "X04"
CONFIG_DIR = Path("configs")
DATA_DIR = Path("configs/data")
RUNS_ROOT = Path("D:/runs")

# Nano tier only (2026-07-28). Keys are the ledger's short model tags.
MODELS = {"y8n": "yolov8n.pt", "y11n": "yolo11n.pt", "y26n": "yolo26n.pt"}
SEEDS = (0, 1, 2)
DIRECTIONS = ("sh17", "chv")  # the dataset each run TRAINS on; cross-eval is automatic

EPOCHS = 200
PATIENCE = 50
BATCH = 16  # nano at 640 on 8 GB
IMGSZ = 640
WORKERS = 6

DATASET_ROOTS = {
    "sh17_root": "D:/Dissertation/SH17_dataset",
    "chv_root": "D:/Dissertation/CHV_dataset",
    "pictor_root": "D:/Dissertation/pictor-ppe_dataset",
}


@dataclass(frozen=True)
class Run:
    """One cell of the grid."""

    model_key: str
    seed: int
    direction: str

    @property
    def run_id(self) -> str:
        return f"{EXPERIMENT}-{self.model_key}-s{self.seed}-{self.direction}"

    @property
    def config_path(self) -> Path:
        return CONFIG_DIR / f"{self.run_id}.yaml"

    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / self.run_id

    @property
    def summary(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / "weights" / "last.pt"

    @property
    def state(self) -> str:
        if self.summary.exists():
            return "done"
        if self.checkpoint.exists():
            return "resumable"
        return "pending"


def grid() -> list[Run]:
    """The 18 runs in execution order — seed-major (see the module docstring)."""
    return [
        Run(model_key, seed, direction)
        for seed in SEEDS
        for model_key in MODELS
        for direction in DIRECTIONS
    ]


# ------------------------------------------------------------------------------- locking


def config_for(run: Run) -> dict:
    """The frozen config for one run. Every value here is the reproducibility contract."""
    return {
        "run_id": run.run_id,
        "model": MODELS[run.model_key],
        "seed": run.seed,
        "datasets": dict(DATASET_ROOTS),
        "data": str(DATA_DIR / f"{run.direction}-640.yaml").replace("\\", "/"),
        "imgsz": IMGSZ,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch": BATCH,
        "amp": True,
        "cache": False,
        "workers": WORKERS,
        "pretrained": True,
        "optimizer": "auto",
        "device": 0,
        "project": str(RUNS_ROOT).replace("\\", "/"),
        "name": run.run_id,
        "exist_ok": False,
    }


def header_for(run: Run) -> str:
    """The provenance block that makes a frozen config self-explaining a year from now."""
    return (
        f"# FROZEN RUN CONFIG — {run.run_id} (S4 training grid), frozen 2026-07-28.\n"
        f"# DO NOT EDIT. Any change to these values = a new run-ID (run-ID contract).\n"
        f"#\n"
        f"# Grid: 3 nano families x 3 seeds x 2 directions = 18 runs, seed-major order.\n"
        f"# Trains on {run.direction}; cross-evaluates on the other dataset automatically\n"
        f"# (src/run.py), so the transfer gap is produced by every run rather than a later pass.\n"
        f"#\n"
        f"# Stopping protocol is IDENTICAL across all 18 runs: {EPOCHS} epochs, patience\n"
        f"# {PATIENCE}. No fixed epoch cap — capping some runs and not others would make the\n"
        f"# seeds incomparable and turn a protocol difference into apparent variance. The epoch\n"
        f"# each run stops at is the convergence evidence.\n"
        f"#\n"
        f"# Data is the pre-resized build (X03/F22): 640px long side, labels and frozen splits\n"
        f"# identical to the harmonised build. Ultralytics letterboxes to 640 regardless, so the\n"
        f"# training input is unchanged and comparability with C3/C4 holds.\n"
        f"# cache=false — disk caching of decoded 18 MP arrays was the original bottleneck (F21).\n"
    )


def lock(force: bool = False) -> list[Run]:
    """Freeze every config. Refuses to overwrite an existing one unless forced."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = grid()
    written, kept = 0, 0
    for run in runs:
        if run.config_path.exists() and not force:
            kept += 1
            continue
        body = yaml.safe_dump(config_for(run), sort_keys=False, default_flow_style=False)
        run.config_path.write_text(header_for(run) + body, encoding="utf-8")
        written += 1
    logger.info("locked %d configs (%d already frozen, left untouched)", written, kept)
    return runs


# ------------------------------------------------------------------------------ executing


def execute(run: Run, dry_run: bool = False) -> tuple[str, float]:
    """Run one cell. Returns ``(outcome, seconds)``; never raises on a training failure."""
    if run.state == "done":
        logger.info("%s: already complete — skipping", run.run_id)
        return "skipped", 0.0

    command = [sys.executable, "-u", "-m", "src.run", "--config", str(run.config_path)]
    if run.state == "resumable":
        command.append("--resume")
        logger.info("%s: checkpoint found — resuming", run.run_id)

    if dry_run:
        logger.info("%s: would run %s", run.run_id, " ".join(command))
        return "dry-run", 0.0

    run.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run.run_dir / "console.log"
    logger.info("%s: starting (log -> %s)", run.run_id, log_path)

    started = time.time()
    with log_path.open("a", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - started

    if completed.returncode != 0 or not run.summary.exists():
        # Deliberately not raised: one bad run must not cost the rest of the session.
        logger.error(
            "%s: FAILED (exit %d, %.1f min) — see %s; continuing to the next run",
            run.run_id,
            completed.returncode,
            elapsed / 60,
            log_path,
        )
        return "failed", elapsed

    logger.info("%s: complete in %.1f min", run.run_id, elapsed / 60)
    return "ok", elapsed


def report_row(run: Run) -> str:
    """One ledger-ready line per finished run (raw, per-seed — never a graduated number)."""
    if not run.summary.exists():
        return f"| {run.run_id} | — | — | — | {run.state} |"
    data = json.loads(run.summary.read_text(encoding="utf-8"))
    epochs = "?"
    results = run.run_dir / "results.csv"
    if results.exists():
        lines = [line for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
        epochs = str(len(lines) - 1)
    return (
        f"| {run.run_id} | {data['in_domain']['map50']:.4f} | "
        f"{data['cross_domain']['map50']:.4f} | {data['transfer_delta_map50']:+.4f} | "
        f"{epochs} ep, {data['train_seconds'] / 60:.0f} min |"
    )


def status(runs: list[Run]) -> None:
    """Print what is done, running and left — the queue's own YOU-ARE-HERE."""
    by_state: dict[str, int] = {}
    for run in runs:
        by_state[run.state] = by_state.get(run.state, 0) + 1
    logger.info("grid: %d runs — %s", len(runs), dict(sorted(by_state.items())))
    print("\n| run-id | in-domain mAP50 | cross mAP50 | transfer delta | stopped |")
    print("|---|---|---|---|---|")
    for run in runs:
        print(report_row(run))
    print(
        "\n⚠️ Per-seed numbers — NOT results. They graduate only seed-aggregated "
        "(mean ± 95 % BCa CI + named test)."
    )


# ---------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock and execute the S4 training grid.")
    parser.add_argument("action", choices=("lock", "run", "status"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="limit to these seeds")
    parser.add_argument("--models", nargs="+", default=None, help="limit to these model keys")
    parser.add_argument("--force", action="store_true", help="lock: overwrite frozen configs")
    parser.add_argument("--dry-run", action="store_true", help="run: print the plan, execute none")
    arguments = parser.parse_args()

    runs = lock(force=arguments.force) if arguments.action == "lock" else grid()
    if arguments.seeds:
        runs = [run for run in runs if run.seed in arguments.seeds]
    if arguments.models:
        runs = [run for run in runs if run.model_key in arguments.models]

    if arguments.action == "status":
        status(runs)
        return 0

    if arguments.action == "lock":
        status(runs)
        return 0

    missing = [run.run_id for run in runs if not run.config_path.exists()]
    if missing:
        logger.error(
            "no frozen config for %d run(s) — run `lock` first: %s", len(missing), missing[:3]
        )
        return 1

    outcomes: dict[str, int] = {}
    total = 0.0
    for index, run in enumerate(runs, 1):
        logger.info("[%d/%d] %s", index, len(runs), run.run_id)
        outcome, elapsed = execute(run, dry_run=arguments.dry_run)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total += elapsed

    logger.info("queue finished in %.1f h — %s", total / 3600, dict(sorted(outcomes.items())))
    status(runs)
    return 1 if outcomes.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
