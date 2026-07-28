"""Unit tests for the S4 grid definition and queue logic (X04).

The grid is 18 runs over ~18-28 GPU-hours, so the expensive failures are the ones that only
show up after hours of compute: a wrong ordering that leaves the comparison incomplete, a run
that silently repeats instead of resuming, or a config that drifts between seeds. Those are
checked here, where they cost nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.grid import (
    BATCH,
    DIRECTIONS,
    EPOCHS,
    MODELS,
    PATIENCE,
    SEEDS,
    Run,
    config_for,
    grid,
)

# --------------------------------------------------------------------------- the grid shape


def test_grid_is_eighteen_runs():
    assert len(grid()) == len(MODELS) * len(SEEDS) * len(DIRECTIONS) == 18


def test_every_run_id_is_unique():
    ids = [run.run_id for run in grid()]
    assert len(set(ids)) == len(ids)


def test_ordering_is_seed_major():
    """Each seed block must be complete before the next starts — the whole point of the order.

    Seed-major means an interruption costs CI width, not the comparison itself.
    """
    seeds = [run.seed for run in grid()]
    assert seeds == sorted(seeds), "seeds must not interleave"
    assert seeds[:6] == [0] * 6, "the first block must be a complete seed-0 sweep"


def test_each_seed_block_covers_every_model_and_direction():
    for seed in SEEDS:
        block = [run for run in grid() if run.seed == seed]
        assert len(block) == 6
        assert {run.model_key for run in block} == set(MODELS)
        assert {run.direction for run in block} == set(DIRECTIONS)


def test_run_id_encodes_model_seed_and_direction():
    run = Run("y11n", 2, "chv")
    assert run.run_id == "X04-y11n-s2-chv"


# ------------------------------------------------------------------------------ the config


def test_stopping_protocol_is_identical_across_every_run():
    """The load-bearing invariant: a per-run epoch difference would corrupt the seed variance."""
    configs = [config_for(run) for run in grid()]
    assert {config["epochs"] for config in configs} == {EPOCHS}
    assert {config["patience"] for config in configs} == {PATIENCE}
    assert {config["batch"] for config in configs} == {BATCH}
    assert {config["imgsz"] for config in configs} == {640}


def test_only_seed_model_and_data_vary_between_runs():
    varying = {"run_id", "name", "seed", "model", "data"}
    baseline = config_for(grid()[0])
    for run in grid()[1:]:
        differing = {key for key, value in config_for(run).items() if baseline[key] != value}
        assert differing <= varying, f"{run.run_id} changes more than seed/model/data: {differing}"


def test_config_points_at_the_preresized_data():
    """The grid must use the fixed pipeline, not the 270 s/epoch one."""
    for run in grid():
        assert config_for(run)["data"].endswith(f"{run.direction}-640.yaml")


def test_disk_cache_is_off_everywhere():
    """cache='disk' was the original bottleneck (F21) — it must not creep back in."""
    assert all(config_for(run)["cache"] is False for run in grid())


def test_pictor_is_never_a_training_target():
    """Supervisor condition, checked at the grid level as well as in src.guards."""
    for run in grid():
        assert "pictor" not in config_for(run)["data"].lower()


def test_both_transfer_directions_are_present():
    """O4 needs both directions; a one-way grid would quietly drop half the transfer study."""
    assert {run.direction for run in grid()} == {"sh17", "chv"}


# ------------------------------------------------------------------------------ queue state


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import src.grid as module

    monkeypatch.setattr(module, "RUNS_ROOT", tmp_path)
    return tmp_path


def test_untouched_run_is_pending(run_dir: Path):
    assert Run("y8n", 0, "sh17").state == "pending"


def test_run_with_a_checkpoint_but_no_summary_is_resumable(run_dir: Path):
    """The overnight-interruption case: it must resume, not silently start over."""
    run = Run("y8n", 0, "sh17")
    (run.run_dir / "weights").mkdir(parents=True)
    (run.run_dir / "weights" / "last.pt").write_bytes(b"checkpoint")
    assert run.state == "resumable"


def test_run_with_a_summary_is_done_and_gets_skipped(run_dir: Path):
    """Re-invoking the queue must be idempotent — finished work is never repeated."""
    run = Run("y8n", 0, "sh17")
    run.run_dir.mkdir(parents=True)
    run.summary.write_text(json.dumps({"run_id": run.run_id}), encoding="utf-8")
    assert run.state == "done"


def test_queue_log_never_lives_inside_a_run_directory(run_dir: Path):
    """Regression: pre-creating the run dir makes Ultralytics increment the folder name.

    That breaks the run-ID contract AND hides the finished run from this queue, which would
    then repeat it forever. The console log therefore lives in a sibling directory.
    """
    import src.grid as module

    run = Run("y8n", 0, "sh17")
    assert module.QUEUE_LOG_DIR != run.run_dir
    assert run.run_dir not in module.QUEUE_LOG_DIR.parents


def test_summary_takes_precedence_over_a_leftover_checkpoint(run_dir: Path):
    run = Run("y8n", 0, "sh17")
    (run.run_dir / "weights").mkdir(parents=True)
    (run.run_dir / "weights" / "last.pt").write_bytes(b"checkpoint")
    run.summary.write_text(json.dumps({"run_id": run.run_id}), encoding="utf-8")
    assert run.state == "done"
