"""Tests for the Pictor-PPE audit logic (X01/S1.1b).

The containment measure is the piece that matters: it decodes the compliance labels here,
and the same geometry (calibrated on SH17/CHV, never on Pictor) drives the association rule
and the demo. Parsing is tested for its failure modes, since a silently dropped label would
corrupt the evaluation ground truth.
"""

from __future__ import annotations

from src.audit_pictor import (
    COMPLIANCE_DECODING,
    Labels,
    containment,
    parse_labels,
    verify_class_mapping,
)


def test_containment_is_full_when_inside():
    hat = (10, 10, 20, 20, 0)
    worker = (0, 0, 100, 200, 2)
    assert containment(hat, worker) == 1.0


def test_containment_is_zero_when_disjoint():
    assert containment((0, 0, 10, 10, 0), (50, 50, 60, 60, 2)) == 0.0


def test_containment_is_partial_on_a_straddling_box():
    # Half of the hat's area sits inside the worker box.
    assert containment((0, 0, 10, 10, 0), (5, 0, 100, 100, 2)) == 0.5


def test_containment_not_iou_is_what_makes_small_ppe_bind():
    """A tiny hat fully inside a large worker: containment 1.0, IoU would be ~0.005."""
    hat, worker = (10, 10, 20, 20, 0), (0, 0, 100, 100, 2)
    assert containment(hat, worker) == 1.0


def test_parse_labels_reads_boxes_and_records_the_split():
    labels = Labels()
    path = _write("a.jpg\t1,2,3,4,0\t5,6,7,8,2\n")
    parse_labels(path, "train", labels)
    assert labels.by_image == {"a.jpg": [(1, 2, 3, 4, 0), (5, 6, 7, 8, 2)]}
    assert labels.split_of == {"a.jpg": "train"}
    assert labels.malformed == []


def test_parse_labels_flags_malformed_entries_instead_of_dropping_them_silently():
    labels = Labels()
    parse_labels(_write("a.jpg\t1,2,3\tnot,a,box,at,all\n"), "train", labels)
    assert labels.by_image["a.jpg"] == []
    assert len(labels.malformed) == 2


def test_verify_class_mapping_cross_tabulates_geometry_against_the_labels():
    a1, a2 = Labels(), Labels()
    # One worker wearing a hat, one wearing nothing.
    a1.by_image["x.jpg"] = [(10, 10, 20, 20, 0), (0, 0, 100, 200, 2)]
    a2.by_image["x.jpg"] = [(0, 0, 100, 200, 1), (200, 0, 300, 200, 0)]
    table, agreements, total = verify_class_mapping(a1, a2)
    assert table[(1, True, False)] == 1
    assert table[(0, False, False)] == 1
    assert (agreements, total) == (2, 2)


def test_compliance_decoding_covers_every_class_with_both_questions():
    assert set(COMPLIANCE_DECODING) == {0, 1, 2, 3}
    assert all({"helmet", "vest"} == set(v) for v in COMPLIANCE_DECODING.values())


def _write(text: str):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "labels.txt"
    path.write_text(text, encoding="utf-8")
    return path
