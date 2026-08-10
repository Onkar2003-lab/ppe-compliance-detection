"""Tests for the monitor's pure logic and the run entry point's config handling (S2, S6.3–S6.4).

Neither module's *whole* job can be unit-tested — one trains a network, the other drives a
detector — so what is tested here is the logic that decides what gets written down: which
people count as in scope, what the config means, and what reaches the log. The two defects the
first skeleton run exposed (a run directory that silently moved, and untracked people
collapsing into a single identity) are kept as regression cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.associate import HELMET, PERSON, VEST, Box, associate
from src.monitor import (
    DemoConfig,
    Violation,
    boxes_from_result,
    missing_for,
    parse_required,
    resolve_source,
    violations_in_zone,
    write_log,
)
from src.run import dataset_of, subset_yaml
from src.zone import Zone

# A zone over the left half of the frame.
LEFT_HALF = Zone(name="bay", points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))


class FakeBoxes:
    """The subset of an Ultralytics ``Boxes`` object the monitor actually reads."""

    def __init__(self, rows, classes, ids=None):
        self.xywhn = rows
        self.cls = classes
        self.id = ids


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def person_at(x: float, cls: int = PERSON) -> Box:
    """A person box standing at horizontal position ``x``, feet at y=0.9."""
    return Box(cls=cls, xc=x, yc=0.6, w=0.1, h=0.6)


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


# ------------------------------------------------------------------- who counts as violating


def test_only_people_inside_the_zone_are_alerted_on():
    """A bare-headed worker outside the marked area is not this zone's business."""
    boxes = [person_at(0.25), person_at(0.75)]
    people = associate(boxes).people
    violating, untracked = violations_in_zone(people, [1, 2], LEFT_HALF, (HELMET, VEST))
    assert list(violating) == [1]
    assert untracked == 0


def test_a_compliant_worker_in_the_zone_is_not_a_violation():
    person = person_at(0.25)
    helmet = Box(cls=HELMET, xc=0.25, yc=0.33, w=0.05, h=0.04)
    vest = Box(cls=VEST, xc=0.25, yc=0.5, w=0.08, h=0.1)
    people = associate([person, helmet, vest]).people
    violating, _ = violations_in_zone(people, [1, None, None], LEFT_HALF, (HELMET, VEST))
    assert violating == {}


def test_untracked_violators_are_counted_not_merged():
    """Folding every id-less detection into one identity logged a whole site as one incident."""
    boxes = [person_at(0.2), person_at(0.3)]
    people = associate(boxes).people
    violating, untracked = violations_in_zone(people, [None, None], LEFT_HALF, (HELMET,))
    assert violating == {}
    assert untracked == 2


def test_no_zone_means_the_whole_frame_is_in_scope():
    boxes = [person_at(0.25), person_at(0.75)]
    people = associate(boxes).people
    violating, _ = violations_in_zone(people, [1, 2], None, (HELMET,))
    assert sorted(violating) == [1, 2]


def test_required_ppe_is_what_the_zone_asks_for():
    """A helmet-only zone must not alert on a helmeted worker with no vest."""
    person = person_at(0.25)
    helmet = Box(cls=HELMET, xc=0.25, yc=0.33, w=0.05, h=0.04)
    people = associate([person, helmet]).people
    assert missing_for(people[0], (HELMET,)) == ()
    assert missing_for(people[0], (HELMET, VEST)) == (VEST,)


# --------------------------------------------------------------------------- config + source


def test_parse_required_maps_names_and_rejects_unknown_ppe():
    assert parse_required(["helmet", "vest"]) == (HELMET, VEST)
    assert parse_required(["Vest", "vest"]) == (VEST,)  # de-duplicated, case-insensitive
    with pytest.raises(ValueError, match="not in the trained label space"):
        parse_required(["gloves"])
    with pytest.raises(ValueError, match="not in the trained label space"):
        parse_required(["person"])  # a person is not equipment


@pytest.mark.parametrize(
    ("spec", "kind", "live"),
    [
        ("0", "camera", True),
        ("rtsp://cam.local/stream", "stream", True),
        ("D:/clips/yard.mp4", "file", False),
    ],
)
def test_resolve_source_reads_the_spec_without_opening_it(spec, kind, live):
    source = resolve_source(spec)
    assert source.kind == kind
    assert source.live is live


def test_resolve_source_treats_a_directory_as_stills(tmp_path):
    assert resolve_source(str(tmp_path)).kind == "images"


def test_config_round_trips_from_yaml(tmp_path):
    """The S6.4 gate: the demo runs from a config alone, with no hardcoded path."""
    path = tmp_path / "demo.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "weights": "D:/runs/X04-y8n-s0-sh17/weights/best.pt",
                "source": "D:/clips/yard.mp4",
                "zone": "configs/zones/bay.yaml",
                "required_ppe": ["helmet"],
                "conf": 0.25,
                "dwell_seconds": 3.0,
                "out": str(tmp_path / "out"),
                "display": False,
            }
        ),
        encoding="utf-8",
    )
    config = DemoConfig.from_yaml(path)
    assert config.required_ppe == (HELMET,)
    assert config.dwell_seconds == 3.0
    assert config.display is False
    assert config.zone == Path("configs/zones/bay.yaml")
    assert config.out == tmp_path / "out"


def test_config_defaults_to_the_scored_operating_point():
    """Demo and results chapter must share one confidence, or neither describes the other."""
    config = DemoConfig.from_dict({"weights": "w.pt", "source": "clip.mp4"})
    assert config.conf == 0.25
    assert config.association_threshold == 0.80
    assert config.required_ppe == (HELMET, VEST)
    assert config.zone is None


# ------------------------------------------------------------------------------------- log


def test_write_log_round_trips_every_violation(tmp_path: Path):
    violations = [
        Violation("2026-08-10T00:00:00+00:00", 3, 0.12, 7, "bay", "helmet", 0.0, "snap-a.jpg"),
        Violation("2026-08-10T00:00:01+00:00", 9, 0.36, 8, "bay", "helmet+vest", 3.0, "snap-b.jpg"),
    ]
    path = tmp_path / "violations.csv"
    write_log(violations, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("timestamp,frame,seconds,track_id,zone,missing_ppe")
    assert len(lines) == 3
    assert "helmet+vest" in lines[2]
    assert "3.0" in lines[2]  # the dwell that earned the alert is on the row


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
