"""S6.1 the zone layer, tested where its answers are knowable by construction.

Point-in-polygon, boundary handling and the feet-point rule are pure geometry with exact
answers, so they are proven here rather than sampled on footage. That matters for the
evaluation as well as the build: S6.5 evidences the zone logic *by this test suite*, because
a small labelled clip could only ever sample what these cases settle completely.
"""

from __future__ import annotations

import pytest
import yaml

from src.associate import PERSON, Box
from src.zone import Zone, feet_point, load_zone, point_in_polygon, save_zone

SQUARE = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))
# An L around a corner obstruction: the notch is the case a convex-only rule gets wrong.
L_SHAPE = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.4), (0.4, 0.4), (0.4, 0.9), (0.1, 0.9))


def test_interior_and_exterior_points():
    zone = Zone(name="square", points=SQUARE)
    assert zone.contains((0.5, 0.5))
    assert not zone.contains((0.1, 0.5))  # left of it
    assert not zone.contains((0.9, 0.5))  # right of it
    assert not zone.contains((0.5, 0.05))  # above it
    assert not zone.contains((0.5, 0.95))  # below it


def test_boundary_counts_as_inside():
    """A worker standing exactly on the painted line is in the zone, not a coin toss."""
    zone = Zone(name="square", points=SQUARE)
    assert zone.contains((0.2, 0.5))  # on the left edge
    assert zone.contains((0.5, 0.8))  # on the bottom edge
    assert zone.contains((0.2, 0.2))  # on a vertex
    assert zone.contains((0.8, 0.8))  # on the opposite vertex


def test_concave_zone_excludes_its_notch():
    zone = Zone(name="l-shape", points=L_SHAPE)
    assert zone.contains((0.2, 0.2))  # top arm
    assert zone.contains((0.2, 0.7))  # side arm
    assert not zone.contains((0.7, 0.7))  # the notch the L wraps around
    assert not zone.contains((0.95, 0.5))  # outside entirely


def test_ray_through_a_vertex_is_not_double_counted():
    """A vertex sitting exactly on the ray's height is the classic even-odd failure."""
    diamond = ((0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.2, 0.5))
    zone = Zone(name="diamond", points=diamond)
    assert zone.contains((0.5, 0.5))  # the ray leaves through the (0.8, 0.5) vertex
    assert not zone.contains((0.05, 0.5))  # the ray enters *and* leaves through vertices


def test_degenerate_polygons_are_rejected():
    with pytest.raises(ValueError, match="at least 3 points"):
        Zone(name="line", points=((0.1, 0.1), (0.9, 0.9)))
    # The bare function stays total rather than raising: it is called per person per frame.
    assert not point_in_polygon((0.5, 0.5), [(0.1, 0.1), (0.9, 0.9)])


def test_collinear_zone_has_no_interior():
    """Three points on a line enclose nothing; only the line itself is 'inside'."""
    zone = Zone(name="flat", points=((0.1, 0.5), (0.5, 0.5), (0.9, 0.5)))
    assert zone.contains((0.5, 0.5))  # on the degenerate boundary
    assert not zone.contains((0.5, 0.6))


def test_pixel_coordinates_are_rejected_rather_than_silently_wrong():
    """A pixel-coordinate file read as normalised would put every worker outside the zone."""
    with pytest.raises(ValueError, match="outside"):
        Zone(name="pixels", points=((120, 300), (640, 300), (640, 700)))


def test_feet_point_is_the_bottom_centre():
    box = Box(cls=PERSON, xc=0.5, yc=0.4, w=0.1, h=0.2)
    assert feet_point(box) == pytest.approx((0.5, 0.5))


def test_membership_follows_the_feet_not_the_centre():
    """A worker leaning over the boundary is judged by where they stand."""
    zone = Zone(name="square", points=SQUARE)
    leaning_in = Box(cls=PERSON, xc=0.5, yc=0.75, w=0.1, h=0.3)  # centre inside, feet at 0.9
    standing_in = Box(cls=PERSON, xc=0.5, yc=0.6, w=0.1, h=0.3)  # feet at 0.75
    assert not zone.contains_box(leaning_in)
    assert zone.contains_box(standing_in)


def test_zone_survives_the_round_trip_with_identical_membership(tmp_path):
    """The S6.1 gate: a reloaded zone must decide every point exactly as the original did."""
    zone = Zone(name="l-shape", points=L_SHAPE, source="clip.mp4", created="2026-08-10T00:00:00")
    path = tmp_path / "l-shape.yaml"
    save_zone(zone, path)
    reloaded = load_zone(path)

    assert reloaded.points == zone.points
    assert reloaded.name == zone.name
    assert reloaded.source == zone.source
    grid = [(i / 20, j / 20) for i in range(21) for j in range(21)]
    assert [reloaded.contains(p) for p in grid] == [zone.contains(p) for p in grid]


def test_load_rejects_a_file_that_is_not_a_zone(tmp_path):
    path = tmp_path / "not-a-zone.yaml"
    path.write_text(yaml.safe_dump({"epochs": 200}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a zone file"):
        load_zone(path)


def test_to_pixels_scales_to_the_frame():
    zone = Zone(name="square", points=SQUARE)
    assert zone.to_pixels(1000, 500) == [(200, 100), (800, 100), (800, 400), (200, 400)]
