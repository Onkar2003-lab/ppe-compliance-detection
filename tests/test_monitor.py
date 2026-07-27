"""Tests for the monitor's pure logic and the run entry point's config handling (S2).

Neither module's *whole* job can be unit-tested — one trains a network, the other drives a
detector — so what is tested here is the logic that decides what gets written down, plus the
two defects the first skeleton run exposed: a run directory that silently moved, and untracked
people collapsing into a single identity.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.associate import HELMET, PERSON
from src.monitor import Violation, boxes_from_result, write_log
from src.run import dataset_of, subset_yaml


class FakeBoxes:
    """The subset of an Ultralytics ``Boxes`` object the monitor actually reads."""

    def __init__(self, rows, classes, ids=None):
        self.xywhn = rows
        self.cls = classes
        self.id = ids


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_boxes_from_result_reads_classes_and_track_ids():
    result = FakeResult(
        FakeBoxes(
            rows=[(0.5, 0.5, 0.4, 0.8), (0.5, 0.15, 0.08, 0.06)],
            classes=[PERSON, HELMET],
            ids=[7, 7],
        )
    )
    boxes, track_ids = boxes_from_result(result)
    assert [b.cls for b in boxes] == [PERSON, HELMET]
    assert boxes[0].w == 0.4
    assert track_ids == [7, 7]


def test_boxes_from_result_survives_an_untracked_frame():
    """`id` is None until the tracker confirms a track — that must not crash the monitor."""
    result = FakeResult(FakeBoxes(rows=[(0.5, 0.5, 0.4, 0.8)], classes=[PERSON], ids=None))
    boxes, track_ids = boxes_from_result(result)
    assert len(boxes) == 1
    assert track_ids == [None]


def test_boxes_from_result_handles_a_frame_with_no_detections():
    boxes, track_ids = boxes_from_result(FakeResult(None))
    assert boxes == [] and track_ids == []


def test_write_log_round_trips_every_violation(tmp_path: Path):
    violations = [
        Violation("2026-07-28T00:00:00+00:00", 3, 7, "helmet", "snap-a.jpg"),
        Violation("2026-07-28T00:00:01+00:00", 9, 8, "helmet+vest", "snap-b.jpg"),
    ]
    path = tmp_path / "violations.csv"
    write_log(violations, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("timestamp,frame,track_id,missing_ppe")
    assert len(lines) == 3
    assert "helmet+vest" in lines[2]


def test_dataset_of_recovers_the_dataset_from_its_yaml_path():
    assert dataset_of("configs/data/sh17.yaml") == "sh17"
    assert dataset_of(Path("configs/data/chv.yaml")) == "chv"


def test_subset_yaml_truncates_training_and_leaves_validation_whole(tmp_path: Path):
    """The smoke slice shrinks the train list only — evaluation must stay the real one."""
    root = tmp_path / "harmonised"
    root.mkdir()
    (root / "train.txt").write_text("\n".join(f"img{i}.jpg" for i in range(10)), encoding="utf-8")
    source = tmp_path / "sh17.yaml"
    source.write_text(
        yaml.safe_dump({"path": str(root), "train": "train.txt", "val": "val.txt"}),
        encoding="utf-8",
    )

    written = subset_yaml(source, limit=3, out_dir=tmp_path / "out")
    document = yaml.safe_load(written.read_text(encoding="utf-8"))

    assert document["val"] == "val.txt"
    assert Path(document["train"]).read_text(encoding="utf-8").splitlines() == [
        "img0.jpg",
        "img1.jpg",
        "img2.jpg",
    ]
    # The subset must not land inside the run directory: creating it early makes Ultralytics
    # treat the run-ID as used and write results to an incremented folder instead.
    assert "out" in written.parts
