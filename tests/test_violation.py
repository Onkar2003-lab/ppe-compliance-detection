"""Unit tests for the violation-recall axis (X05/S5).

This module turns detections into the project's headline safety claim: *did we catch the
worker with no helmet?* Its scoring conventions are tested on synthetic scenes where
the right answer is known by construction.

The cases that matter most are the ones where a plausible-looking implementation would
flatter itself: a worker the detector never found (must count as a violation missed, not as
an absent sample), and the vest class, whose 1.6 % support makes an always-flag baseline
look excellent. Both are pinned here so they cannot regress quietly.

Coordinates are normalised YOLO ``(xc, yc, w, h)``; the mental picture is a frame holding
one or two standing workers, each 0.4 wide and 0.8 tall.
"""

from __future__ import annotations

import pytest

from src.associate import HELMET, PERSON, VEST, Box
from src.violation import (
    Confusion,
    Score,
    Worker,
    iou,
    match,
    score_image,
    to_normalised,
)


def person(xc: float, yc: float = 0.5, w: float = 0.4, h: float = 0.8) -> Box:
    return Box(PERSON, xc, yc, w, h)


def helmet_on(p: Box) -> Box:
    """A helmet sitting on this person's head, fully contained, so it binds."""
    _x1, y1, _x2, _y2 = p.corners
    return Box(HELMET, p.xc, y1 + 0.04, 0.08, 0.06)


def vest_on(p: Box) -> Box:
    """A vest on this person's torso."""
    _x1, y1, _x2, _y2 = p.corners
    return Box(VEST, p.xc, y1 + 0.35 * p.h, 0.2, 0.2)


def worker(box: Box, helmet: bool, vest: bool, split: str = "test") -> Worker:
    return Worker(box=box, helmet=helmet, vest=vest, split=split)


def fresh() -> Score:
    return Score(run_id="test", weights="none", confidence=0.25, match_iou=0.5)


# ------------------------------------------------------------------------ box conversion


def test_to_normalised_centres_and_scales() -> None:
    box = to_normalised((100, 200, 300, 600, 1), width=1000, height=800)
    assert box.xc == pytest.approx(0.2)
    assert box.yc == pytest.approx(0.5)
    assert box.w == pytest.approx(0.2)
    assert box.h == pytest.approx(0.5)


def test_to_normalised_forces_person_class() -> None:
    """A Pictor row's class is a compliance state, never an object class.

    Letting class 1 (``WH``) through would hand the association rule a box it reads as a
    helmet, inventing PPE out of a worker label.
    """
    for compliance_class in (0, 1, 2, 3):
        box = to_normalised((0, 0, 100, 200, compliance_class), width=200, height=400)
        assert box.cls == PERSON


# ----------------------------------------------------------------------------- matching


def test_iou_identical_boxes_is_one() -> None:
    assert iou(person(0.5), person(0.5)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    assert iou(person(0.15), person(0.85)) == 0.0


def test_match_pairs_overlapping_person_to_worker() -> None:
    predicted = [person(0.5)]
    workers = [worker(person(0.5), helmet=True, vest=True)]
    assert match(predicted, workers) == {0: 0}


def test_match_rejects_weak_overlap() -> None:
    """A detection that barely grazes a worker is not that worker."""
    predicted = [person(0.5)]
    workers = [worker(person(0.9), helmet=True, vest=True)]
    assert match(predicted, workers) == {}


def test_match_is_one_to_one() -> None:
    """One detection cannot be credited with covering two workers standing together."""
    predicted = [person(0.5)]
    workers = [
        worker(person(0.5), helmet=True, vest=True),
        worker(person(0.52), helmet=False, vest=False),
    ]
    pairs = match(predicted, workers)
    assert len(pairs) == 1
    assert set(pairs.values()) == {0}


def test_match_prefers_the_stronger_overlap() -> None:
    predicted = [person(0.52), person(0.5)]
    workers = [worker(person(0.5), helmet=True, vest=True)]
    assert match(predicted, workers) == {0: 1}


# ------------------------------------------------------------------------------ scoring


def test_bare_headed_worker_is_a_caught_violation() -> None:
    p = person(0.5)
    score = fresh()
    score_image([p], [worker(p, helmet=False, vest=False)], score)

    helmet = score.classes["helmet"]
    assert (helmet.tp, helmet.fn, helmet.fp) == (1, 0, 0)
    assert helmet.recall == 1.0


def test_helmeted_worker_raises_no_alert() -> None:
    p = person(0.5)
    score = fresh()
    score_image([p, helmet_on(p)], [worker(p, helmet=True, vest=False)], score)

    helmet = score.classes["helmet"]
    assert (helmet.tp, helmet.fp, helmet.fn, helmet.tn) == (0, 0, 0, 1)


def test_undetected_worker_counts_as_a_missed_violation() -> None:
    """The rule that keeps the axis honest (decision 1).

    A worker the detector never finds raises no alert. Dropping them from the denominator
    would hide the pipeline's worst failure mode and inflate recall.
    """
    score = fresh()
    score_image([], [worker(person(0.5), helmet=False, vest=False)], score)

    helmet = score.classes["helmet"]
    assert helmet.tp == 0
    assert helmet.fn == 1
    assert helmet.missed_people == 1
    assert helmet.support == 1
    assert helmet.recall == 0.0


def test_undetected_compliant_worker_is_correct_silence_not_a_false_alarm() -> None:
    score = fresh()
    score_image([], [worker(person(0.5), helmet=True, vest=True)], score)

    helmet = score.classes["helmet"]
    assert (helmet.fp, helmet.tn) == (0, 1)
    assert helmet.support == 0


def test_helmet_on_the_ground_does_not_make_a_worker_compliant() -> None:
    """Unbound PPE must not be credited to anybody; that would invent compliance."""
    p = person(0.2)
    stray = Box(HELMET, 0.85, 0.9, 0.08, 0.06)
    score = fresh()
    score_image([p, stray], [worker(p, helmet=False, vest=False)], score)

    assert score.classes["helmet"].tp == 1


def test_two_workers_scored_independently() -> None:
    left, right = person(0.25), person(0.75)
    score = fresh()
    score_image(
        [left, right, helmet_on(left)],
        [worker(left, helmet=True, vest=False), worker(right, helmet=False, vest=False)],
        score,
    )

    helmet = score.classes["helmet"]
    assert (helmet.tp, helmet.tn, helmet.fn, helmet.fp) == (1, 1, 0, 0)
    assert score.workers == 2
    assert score.workers_detected == 2


def test_false_alarm_when_worn_ppe_is_not_detected() -> None:
    """The worker has a helmet; the detector missed it, so the alert is wrong."""
    p = person(0.5)
    score = fresh()
    score_image([p], [worker(p, helmet=True, vest=True)], score)

    assert score.classes["helmet"].fp == 1
    assert score.classes["helmet"].precision == 0.0


def test_vest_scored_alongside_helmet_on_the_same_worker() -> None:
    p = person(0.5)
    score = fresh()
    score_image([p, helmet_on(p)], [worker(p, helmet=True, vest=False)], score)

    assert score.classes["helmet"].tn == 1
    assert score.classes["vest"].tp == 1


def test_split_is_recorded_so_a_test_only_figure_stays_recoverable() -> None:
    score = fresh()
    p = person(0.5)
    score_image([p], [worker(p, helmet=False, vest=False, split="valid")], score)
    assert score.by_split["valid"] == 1


# ------------------------------------------------------------- metrics and the baseline


def test_confusion_metrics() -> None:
    c = Confusion(tp=6, fp=2, fn=4, tn=8)
    assert c.support == 10
    assert c.recall == pytest.approx(0.6)
    assert c.precision == pytest.approx(0.75)
    assert c.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert c.base_rate == pytest.approx(0.5)


def test_empty_confusion_does_not_divide_by_zero() -> None:
    c = Confusion()
    assert (c.recall, c.precision, c.f1, c.base_rate) == (0.0, 0.0, 0.0, 0.0)


def test_summary_always_carries_the_trivial_baseline() -> None:
    """The vest guard, enforced structurally (user decision A, X01/S1.1b F13).

    Vest violation recall looks superb for a model that simply never predicts a vest, so
    the baseline and the support count are part of the metric payload rather than a note
    somebody has to remember to add.
    """
    summary = Confusion(tp=98, fp=0, fn=0, tn=2).summary()
    assert summary["trivial_always_flag"]["recall"] == 1.0
    assert summary["trivial_always_flag"]["precision"] == pytest.approx(0.98)
    assert summary["support_violations"] == 98


def test_report_exposes_the_headline_and_the_undetected_count() -> None:
    p = person(0.5)
    score = fresh()
    score_image([], [worker(p, helmet=False, vest=False)], score)
    report = score.report()

    assert report["headline_helmet_violation_recall"] == 0.0
    assert report["person_detection_rate"] == 0.0
    assert report["axes"]["helmet"]["violations_missed_because_person_undetected"] == 1
    assert "trivial_always_flag" in report["axes"]["vest"]
