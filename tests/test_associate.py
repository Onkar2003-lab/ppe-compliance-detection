"""Unit tests for the person↔PPE association rule (X01/S1.4c).

This rule decides who is wearing what, so it decides every violation number the axis
reports and every alert the demo fires. It is tested on synthetic geometry where the right
answer is known by construction, including the cases that break a naive overlap rule: a
helmet between two heads, a partly occluded torso, two overlapping people, and PPE lying in
the scene with no wearer at all.

Coordinates are normalised YOLO ``(xc, yc, w, h)``. The mental picture for most tests is a
frame with one or two standing workers, each 0.4 wide and 0.8 tall.
"""

from __future__ import annotations

import pytest

from src.associate import (
    HELMET,
    PERSON,
    THRESHOLD,
    VEST,
    Box,
    anchor_distance,
    associate,
    boxes_from_rows,
    choose_threshold,
    containment,
    iou,
    retention,
    sweep,
)


def person(xc: float, yc: float = 0.5, w: float = 0.4, h: float = 0.8) -> Box:
    return Box(PERSON, xc, yc, w, h)


def helmet(xc: float, yc: float, w: float = 0.08, h: float = 0.06) -> Box:
    return Box(HELMET, xc, yc, w, h)


def vest(xc: float, yc: float, w: float = 0.2, h: float = 0.25) -> Box:
    return Box(VEST, xc, yc, w, h)


# ------------------------------------------------------------------------------ geometry


def test_containment_is_one_when_the_ppe_sits_inside_the_person():
    worker = person(0.5)
    assert containment(helmet(0.5, 0.15), worker) == pytest.approx(1.0)


def test_containment_is_zero_when_the_boxes_do_not_meet():
    assert containment(helmet(0.9, 0.9), person(0.2)) == 0.0


def test_containment_is_the_fraction_of_the_ppe_box_not_of_the_union():
    """Half a vest inside the person box scores 0.5 however large the person is."""
    worker = Box(PERSON, 0.5, 0.5, 0.4, 0.8)  # spans x in [0.3, 0.7]
    half_out = vest(0.7, 0.4, w=0.2, h=0.2)  # spans x in [0.6, 0.8]
    assert containment(half_out, worker) == pytest.approx(0.5)


def test_iou_rejects_the_helmet_pairs_containment_accepts():
    """Why we depart from Nath's IoU > 0.45: a worn helmet cannot reach it."""
    worker = person(0.5)
    worn = helmet(0.5, 0.15)
    assert containment(worn, worker) == pytest.approx(1.0)
    assert iou(worn, worker) < 0.45


def test_containment_of_a_degenerate_box_is_zero_not_an_error():
    assert containment(Box(HELMET, 0.5, 0.5, 0.0, 0.0), person(0.5)) == 0.0


def test_anchor_distance_prefers_the_expected_place_on_the_body():
    """A helmet belongs at the top of a person, a vest across the upper torso."""
    worker = person(0.5)  # spans y in [0.1, 0.9]
    at_head = helmet(0.5, 0.14)
    at_feet = helmet(0.5, 0.86)
    assert anchor_distance(at_head, worker) < anchor_distance(at_feet, worker)
    assert anchor_distance(vest(0.5, 0.38), worker) < anchor_distance(vest(0.5, 0.85), worker)


# --------------------------------------------------------------------------- association


def test_a_lone_worker_in_full_ppe_is_compliant():
    boxes = [person(0.5), helmet(0.5, 0.15), vest(0.5, 0.4)]
    result = associate(boxes)
    assert result.people[0].pictor_code == "WHV"
    assert result.people[0].violations == ()
    assert result.unbound == []


def test_a_worker_without_a_helmet_is_a_helmet_violation():
    result = associate([person(0.5), vest(0.5, 0.4)])
    state = result.people[0]
    assert state.pictor_code == "WV"
    assert state.violations == (HELMET,)


def test_a_bare_worker_is_both_violations():
    result = associate([person(0.5)])
    assert result.people[0].pictor_code == "W"
    assert result.people[0].violations == (HELMET, VEST)


def test_a_helmet_between_two_heads_goes_to_the_nearer_head():
    """Both workers fully contain the helmet, so containment alone cannot decide."""
    left, right = person(0.35), person(0.60)
    ambiguous = helmet(0.47, 0.14)  # inside both boxes, slightly nearer the left worker
    assert containment(ambiguous, left) == pytest.approx(1.0)
    assert containment(ambiguous, right) == pytest.approx(1.0)

    result = associate([left, right, ambiguous])
    assert result.people[0].wears(HELMET)  # left worker: nearer anchor
    assert not result.people[1].wears(HELMET)
    assert result.unbound == []


def test_a_helmet_is_never_shared_between_two_people():
    result = associate([person(0.35), person(0.60), helmet(0.47, 0.14)])
    bound = [p for p in result.people if p.wears(HELMET)]
    assert len(bound) == 1


def test_two_overlapping_people_keep_their_own_helmets():
    """The front worker contains part of the back worker; greedy binding must not take both."""
    back, front = person(0.45), person(0.55)
    back_helmet = helmet(0.45, 0.14)
    front_helmet = helmet(0.55, 0.18)
    result = associate([back, front, back_helmet, front_helmet])
    assert all(p.wears(HELMET) for p in result.people)
    assert result.people[0].bound[HELMET] != result.people[1].bound[HELMET]
    assert result.unbound == []


def test_an_occluded_torso_still_binds_its_vest():
    """A vest cut by the person box edge binds while it stays above the threshold."""
    worker = Box(PERSON, 0.5, 0.5, 0.4, 0.8)
    mostly_inside = Box(VEST, 0.68, 0.4, 0.1, 0.2)  # x in [0.63, 0.73]; 0.7 inside
    assert containment(mostly_inside, worker) == pytest.approx(0.7)
    assert associate([worker, mostly_inside], threshold=0.6).people[0].wears(VEST)
    assert not associate([worker, mostly_inside], threshold=0.8).people[0].wears(VEST)


def test_ppe_lying_in_the_scene_belongs_to_nobody():
    """A helmet on the ground must not make an uncovered worker look compliant."""
    worker = person(0.2)
    on_the_ground = helmet(0.85, 0.9)
    result = associate([worker, on_the_ground])
    assert result.people[0].violations == (HELMET, VEST)
    assert result.unbound == [1]


def test_a_person_takes_only_one_item_per_class():
    """Two helmets over one worker: one binds, the spare is reported, not stacked."""
    worker = person(0.5)
    result = associate([worker, helmet(0.5, 0.14), helmet(0.45, 0.16)])
    assert len(result.people[0].bound) == 1
    assert len(result.unbound) == 1


def test_an_image_with_no_people_binds_nothing():
    result = associate([helmet(0.5, 0.15), vest(0.5, 0.4)])
    assert result.people == []
    assert result.unbound == [0, 1]


def test_empty_input_is_handled():
    result = associate([])
    assert result.people == [] and result.unbound == [] and result.codes == []


def test_the_result_is_deterministic_for_identical_geometry():
    boxes = [person(0.35), person(0.35), helmet(0.35, 0.14)]
    first = [p.bound for p in associate(boxes).people]
    second = [p.bound for p in associate(boxes).people]
    assert first == second


def test_boxes_from_rows_round_trips_a_label_file():
    rows = [(PERSON, 0.5, 0.5, 0.4, 0.8), (HELMET, 0.5, 0.15, 0.08, 0.06)]
    assert associate(boxes_from_rows(rows)).people[0].wears(HELMET)


def test_the_frozen_threshold_binds_ordinary_worn_ppe():
    """Guards the calibrated constant: a normally worn kit must survive it."""
    result = associate([person(0.5), helmet(0.5, 0.15), vest(0.5, 0.4)], threshold=THRESHOLD)
    assert result.people[0].pictor_code == "WHV"


# --------------------------------------------------------------------------- calibration


def test_retention_counts_scores_at_or_above_the_threshold():
    assert retention([0.2, 0.8, 1.0], 0.8) == pytest.approx(2 / 3)
    assert retention([], 0.5) == 1.0


def test_choose_threshold_maximises_separation_and_reports_the_plateau():
    table = [
        {"threshold": 0.2, "separation": 0.50},
        {"threshold": 0.5, "separation": 0.80},
        {"threshold": 0.6, "separation": 0.795},  # within tolerance: same plateau
        {"threshold": 0.9, "separation": 0.60},
    ]
    chosen, plateau = choose_threshold(table)
    assert chosen == 0.5
    assert plateau == [0.5, 0.6]


def test_sweep_weights_every_dataset_and_class_equally():
    """A big dataset must not drag the threshold on its own."""
    from src.associate import Pairs

    big, small = Pairs(), Pairs()
    big.positive = {HELMET: [1.0] * 1000, VEST: [1.0] * 1000}
    big.negative = [0.0] * 1000
    small.positive = {HELMET: [0.1], VEST: [0.1]}
    small.negative = [0.0]
    row = next(r for r in sweep({"big": big, "small": small}) if r["threshold"] == 0.5)
    assert row["big-helmet"] == 1.0 and row["small-helmet"] == 0.0
    assert row["separation"] == pytest.approx(0.5)  # 0.5 kept - 0.0 bound
