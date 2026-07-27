"""Tests for the run guards — chiefly the supervisor's evaluation-only condition.

A binding condition that is only enforced by good intentions is not enforced, so the guard
that keeps Pictor-PPE out of training is tested like any other piece of risky logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.guards import assert_training_data_excludes_pictor


def test_a_clean_training_config_passes():
    config = {
        "data": "D:/Dissertation/SH17_dataset/sh17.yaml",
        "datasets": {"sh17_root": "D:/Dissertation/SH17_dataset"},
    }
    assert_training_data_excludes_pictor(config)  # must not raise


def test_declaring_the_pictor_root_is_allowed():
    """`base.yaml` declares the root so the audit + eval paths can find it — that is legal."""
    config = {
        "data": "configs/data/sh17.yaml",
        "datasets": {
            "sh17_root": "D:/Dissertation/SH17_dataset",
            "pictor_root": "D:/Dissertation/pictor-ppe_dataset",
        },
    }
    assert_training_data_excludes_pictor(
        config, pictor_root=Path("D:/Dissertation/pictor-ppe_dataset")
    )


def test_the_declared_root_does_not_excuse_it_elsewhere():
    """Declaring the root is fine; pointing training at it is not, even in the same config."""
    config = {
        "data": "D:/Dissertation/pictor-ppe_dataset/data.yaml",
        "datasets": {"pictor_root": "D:/Dissertation/pictor-ppe_dataset"},
    }
    with pytest.raises(ValueError, match="EVALUATION-ONLY"):
        assert_training_data_excludes_pictor(config)


def test_another_dataset_root_pointing_at_pictor_is_refused():
    """The exemption is one key, not the whole `datasets` block."""
    config = {"data": "sh17.yaml", "datasets": {"sh17_root": "D:/Dissertation/pictor-ppe_dataset"}}
    with pytest.raises(ValueError, match="EVALUATION-ONLY"):
        assert_training_data_excludes_pictor(config)


def test_pictor_in_the_data_key_is_refused():
    config = {"data": "D:/Dissertation/pictor-ppe_dataset/pictor.yaml"}
    with pytest.raises(ValueError, match="EVALUATION-ONLY"):
        assert_training_data_excludes_pictor(config)


def test_pictor_smuggled_into_a_nested_key_is_refused():
    """A copied config can hide the path anywhere, so every string is checked."""
    config = {"data": "sh17.yaml", "extra": {"paths": ["D:/Dissertation/pictor-ppe_dataset"]}}
    with pytest.raises(ValueError, match="EVALUATION-ONLY"):
        assert_training_data_excludes_pictor(config)


def test_windows_separators_and_case_do_not_evade_the_guard():
    config = {"data": r"D:\Dissertation\PICTOR-PPE_dataset\data.yaml"}
    with pytest.raises(ValueError, match="EVALUATION-ONLY"):
        assert_training_data_excludes_pictor(config)


def test_a_renamed_root_is_still_caught_via_the_configured_path():
    config = {"data": "D:/data/eval-only-set/data.yaml"}
    with pytest.raises(ValueError, match="EVALUATION-ONLY"):
        assert_training_data_excludes_pictor(config, pictor_root=Path("D:/data/eval-only-set"))
