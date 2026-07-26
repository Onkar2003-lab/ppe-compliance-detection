"""Unit tests for the cross-dataset near-duplicate exclusion (X01/S1.4, finding F7).

The exclusion defends the zero-shot claim, so the logic that finds pairs and the check
that proves a split is clean are both tested on synthetic fingerprints where the right
answer is known by construction. A final regression test re-runs the real check, and skips
if the S1.2 dHash cache is not on this machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.overlap import (
    DEFAULT_CACHE,
    Pair,
    exclusions,
    hamming,
    leaking_pairs,
    load_hashes,
    native_splits,
    pairs_within,
    split_of,
    verify,
)


def test_hamming_counts_differing_bits():
    assert hamming(0, 0) == 0
    assert hamming(0b1011, 0b1001) == 1
    assert hamming(0, 2**64 - 1) == 64


def test_pairs_within_respects_the_radius():
    chv = {"a": 0b0000, "b": 0b1111}
    sh17 = {"x": 0b0001, "y": 0b1111_0000}
    pairs = pairs_within(chv, sh17, threshold=1)
    assert pairs == [Pair("a", "x", 1)]


def test_pairs_within_keeps_every_match_not_just_the_nearest():
    """One SH17 image can be the twin of several CHV frames from the same shoot."""
    chv = {"a": 0b0000, "b": 0b0001, "c": 0b0011}
    sh17 = {"x": 0b0000}
    pairs = pairs_within(chv, sh17, threshold=2)
    assert [(p.chv, p.distance) for p in pairs] == [("a", 0), ("b", 1), ("c", 2)]


def test_pairs_within_handles_empty_input():
    assert pairs_within({}, {"x": 1}, threshold=5) == []
    assert pairs_within({"a": 1}, {}, threshold=5) == []


def test_exclusions_cover_both_sides_and_deduplicate():
    pairs = [Pair("a", "x", 0), Pair("b", "x", 3)]
    assert exclusions(pairs) == {"chv": ["a", "b"], "sh17": ["x"]}


def test_leaking_pairs_fires_only_on_the_other_datasets_training_pool():
    pairs = [Pair("a", "x", 0)]
    # 'a' is evaluated and its twin 'x' is trained on -> leak.
    assert leaking_pairs({"a"}, {"x"}, pairs, "chv") == pairs
    # twin sits in the other dataset's *eval* split, not its training pool -> no leak.
    assert leaking_pairs({"a"}, {"other"}, pairs, "chv") == []
    # 'a' was excluded from the evaluation split -> no leak.
    assert leaking_pairs(set(), {"x"}, pairs, "chv") == []


def test_leaking_pairs_checks_the_reverse_direction_too():
    pairs = [Pair("a", "x", 0)]
    assert leaking_pairs({"x"}, {"a"}, pairs, "sh17") == pairs


def test_leaking_pairs_rejects_an_unknown_dataset():
    with pytest.raises(ValueError, match="chv"):
        leaking_pairs(set(), set(), [], "pictor")


def test_split_of_reports_membership():
    splits = {"train": {"a"}, "val": {"b"}}
    assert split_of("a", splits) == "train"
    assert split_of("b", splits) == "val"
    assert split_of("c", splits) == "none"


@pytest.mark.skipif(
    not Path(DEFAULT_CACHE).is_file(), reason="S1.2 dHash cache not present on this machine"
)
def test_no_residual_leakage_in_the_real_splits():
    """Regression guard on the actual datasets: the exclusion must leave zero leakage."""
    hashes = load_hashes(DEFAULT_CACHE)
    pairs = pairs_within(hashes["chv"], hashes["sh17"], threshold=5)
    assert verify(pairs, native_splits()) == []
