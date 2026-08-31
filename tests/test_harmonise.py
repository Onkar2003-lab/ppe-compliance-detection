"""Tests for harmonisation and split freezing (X01/S1.4).

These guard the two things that would corrupt every downstream number without failing
loudly: a mapping that sends a class to the wrong target, and a split that is not
reproducible. The split ID is the contract between a reported number and the exact image
list behind it, so its stability is tested explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.harmonise import (
    SOURCE_MAPS,
    TARGET_NAMES,
    VAL_FRACTION,
    split_id,
    stratified_split,
    stratum,
)


def test_every_mapping_target_exists_in_the_label_space():
    for dataset, mapping in SOURCE_MAPS.items():
        assert set(mapping.values()) <= set(TARGET_NAMES), dataset


def test_chv_collapses_all_four_helmet_colours_to_one_class():
    helmet = TARGET_NAMES_INVERSE["helmet"]
    assert [SOURCE_MAPS["chv"][colour] for colour in (2, 3, 4, 5)] == [helmet] * 4


def test_sh17_uses_the_corrected_class_ids():
    """The vault previously recorded 1/13/15; on disk it is 0/10/16 (X01/F1)."""
    assert SOURCE_MAPS["sh17"] == {0: 0, 10: 1, 16: 2}


def test_split_id_is_stable_and_order_independent():
    assert split_id(["b", "a", "c"]) == split_id(["a", "b", "c"])
    assert split_id(["a", "b"]) != split_id(["a", "c"])


def test_stratified_split_is_deterministic_for_a_seed(tmp_path: Path):
    stems = _labelled(tmp_path, {f"img{i}": "0 0.5 0.5 0.2 0.2" for i in range(20)})
    first = stratified_split(stems, tmp_path, VAL_FRACTION, seed=0)
    second = stratified_split(stems, tmp_path, VAL_FRACTION, seed=0)
    assert first == second


def test_stratified_split_changes_with_the_seed(tmp_path: Path):
    stems = _labelled(tmp_path, {f"img{i}": "0 0.5 0.5 0.2 0.2" for i in range(40)})
    assert stratified_split(stems, tmp_path, VAL_FRACTION, 0) != stratified_split(
        stems, tmp_path, VAL_FRACTION, 1
    )


def test_stratified_split_partitions_without_loss_or_overlap(tmp_path: Path):
    stems = _labelled(tmp_path, {f"img{i}": "0 0.5 0.5 0.2 0.2" for i in range(30)})
    majority, minority = stratified_split(stems, tmp_path, VAL_FRACTION, seed=0)
    assert set(majority) | set(minority) == set(stems)
    assert not set(majority) & set(minority)
    assert len(minority) == 3  # 10% of 30


def test_stratified_split_keeps_rare_classes_in_both_sides(tmp_path: Path):
    """A rare class must not land wholly in one side, which random splitting permits."""
    content = {f"common{i}": "0 0.5 0.5 0.2 0.2" for i in range(20)}
    content.update({f"rare{i}": "1 0.5 0.5 0.2 0.2" for i in range(20)})
    stems = _labelled(tmp_path, content)
    majority, minority = stratified_split(stems, tmp_path, 0.5, seed=0)
    assert sum(s.startswith("rare") for s in minority) == 10
    assert sum(s.startswith("rare") for s in majority) == 10


def test_stratum_keys_on_the_class_combination(tmp_path: Path):
    _labelled(tmp_path, {"a": "0 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1", "b": ""})
    assert stratum(tmp_path / "a.txt") == "0-2"
    assert stratum(tmp_path / "b.txt") == "background"


def test_released_split_lists_match_the_manifest_ids():
    """The committed split lists ARE the splits every reported number was scored on.

    S7 releases the six image lists so a reader can check our splits rather than take them on
    trust. That only means something if the lists cannot drift from the manifest that names
    them, so the IDs are recomputed here from the committed files. Editing a list without
    regenerating the manifest fails this test, which is the point.
    """
    splits = Path("configs/splits")
    manifest = json.loads((splits / "split-manifest.json").read_text(encoding="utf-8"))
    for dataset, entries in manifest["splits"].items():
        for name, expected in entries.items():
            listed = (splits / f"{dataset}-{name}.txt").read_text(encoding="utf-8").split()
            stems = sorted(Path(line).stem for line in listed)
            assert len(stems) == expected["images"], f"{dataset}-{name} image count"
            assert split_id(stems) == expected["split_id"], f"{dataset}-{name} split ID"


TARGET_NAMES_INVERSE = {name: cls for cls, name in TARGET_NAMES.items()}


def _labelled(directory: Path, content: dict[str, str]) -> list[str]:
    for stem, text in content.items():
        (directory / f"{stem}.txt").write_text(text, encoding="utf-8")
    return sorted(content)
