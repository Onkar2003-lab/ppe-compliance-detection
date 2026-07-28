"""Unit tests for the offline pre-resize (X03/S3, finding F21).

The whole approach rests on one claim: scaling the pixels does **not** change the labels or
the frozen splits, because YOLO coordinates are normalised. That claim is load-bearing for
every downstream run, so it is tested by construction here rather than assumed — a synthetic
dataset is built, mirrored, and checked byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preresize import (
    TARGET_LONG_SIDE,
    build,
    target_size,
    validate,
    write_dataset_yaml,
)

# ------------------------------------------------------------------------------ geometry


def test_oversized_image_scales_longest_side_to_target():
    assert target_size(4475, 4057) == (640, 580)


def test_landscape_and_portrait_are_treated_symmetrically():
    assert target_size(2000, 1000) == (640, 320)
    assert target_size(1000, 2000) == (320, 640)


def test_image_within_target_is_left_alone():
    assert target_size(640, 480) is None
    assert target_size(320, 200) is None


def test_never_upscales_a_small_image():
    """A below-target image must not be blown up — that would invent pixels."""
    assert target_size(100, 80) is None


def test_aspect_ratio_is_preserved_within_rounding():
    width, height = target_size(1920, 1080)
    assert abs((width / height) - (1920 / 1080)) < 0.01


def test_extreme_aspect_ratio_keeps_at_least_one_pixel():
    """A panorama must not round its short side to zero."""
    width, height = target_size(10000, 5)
    assert width == TARGET_LONG_SIDE
    assert height >= 1


# --------------------------------------------------------------------------- the mirror


@pytest.fixture
def harmonised(tmp_path: Path) -> Path:
    """A miniature harmonised dataset: two oversized images, one already small."""
    root = tmp_path / "source" / "toy"
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    sizes = {"big-a": (1600, 1200), "big-b": (2000, 900), "small-c": (400, 300)}
    for stem, (width, height) in sizes.items():
        canvas = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(images / f"{stem}.jpg"), canvas)
        # Normalised coordinates: identical whatever the pixel dimensions become.
        (labels / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.25 0.4\n1 0.2 0.3 0.1 0.1", encoding="utf-8"
        )

    (root / "train.txt").write_text(
        "\n".join(str(images / f"{stem}.jpg") for stem in ("big-a", "big-b")), encoding="utf-8"
    )
    (root / "val.txt").write_text(str(images / "small-c.jpg"), encoding="utf-8")
    return root


def test_build_mirrors_every_image_and_validates_clean(harmonised: Path, tmp_path: Path):
    out = tmp_path / "out" / "toy"
    result = build("toy", harmonised, out, workers=2)

    assert result.total == 3
    assert result.resized == 2  # the two oversized
    assert result.linked == 1  # the one already within target
    assert result.failed == []
    assert validate("toy", harmonised, out) == []


def test_labels_are_byte_identical_after_the_resize(harmonised: Path, tmp_path: Path):
    """The core claim. If a resize ever rewrote a label, every downstream metric is wrong."""
    out = tmp_path / "out" / "toy"
    build("toy", harmonised, out, workers=2)

    for label in (harmonised / "labels").glob("*.txt"):
        assert (out / "labels" / label.name).read_bytes() == label.read_bytes()


def test_split_membership_survives_the_mirror(harmonised: Path, tmp_path: Path):
    """Frozen splits are the reproducibility contract — the mirror repoints, never reshuffles."""
    out = tmp_path / "out" / "toy"
    build("toy", harmonised, out, workers=2)

    for split in ("train", "val"):
        before = [
            Path(line).name for line in (harmonised / f"{split}.txt").read_text().splitlines()
        ]
        after = [Path(line).name for line in (out / f"{split}.txt").read_text().splitlines()]
        assert before == after
        for line in (out / f"{split}.txt").read_text().splitlines():
            assert Path(line).exists()


def test_resized_images_are_within_the_target(harmonised: Path, tmp_path: Path):
    out = tmp_path / "out" / "toy"
    build("toy", harmonised, out, workers=2)

    for path in (out / "images").iterdir():
        image = cv2.imread(str(path))
        assert max(image.shape[:2]) <= TARGET_LONG_SIDE


def test_small_image_is_carried_across_untouched(harmonised: Path, tmp_path: Path):
    """No re-encode without cause — an in-target image must be bit-identical, not recompressed."""
    out = tmp_path / "out" / "toy"
    build("toy", harmonised, out, workers=2)

    source = (harmonised / "images" / "small-c.jpg").read_bytes()
    assert (out / "images" / "small-c.jpg").read_bytes() == source


def test_validate_catches_a_tampered_label(harmonised: Path, tmp_path: Path):
    """The validator must actually fail on the thing it exists to catch."""
    out = tmp_path / "out" / "toy"
    build("toy", harmonised, out, workers=2)

    tampered = out / "labels" / "big-a.txt"
    tampered.unlink()  # break the hard link rather than editing through it
    tampered.write_text("0 0.9 0.9 0.1 0.1", encoding="utf-8")

    problems = validate("toy", harmonised, out)
    assert any("CHANGED" in problem for problem in problems)


def test_validate_catches_a_missing_image(harmonised: Path, tmp_path: Path):
    out = tmp_path / "out" / "toy"
    build("toy", harmonised, out, workers=2)
    (out / "images" / "big-b.jpg").unlink()

    problems = validate("toy", harmonised, out)
    assert any("image set differs" in problem for problem in problems)


def test_dataset_yaml_points_at_the_resized_root(harmonised: Path, tmp_path: Path):
    out = tmp_path / "out" / "toy"
    build("toy", harmonised, out, workers=2)
    written = write_dataset_yaml("toy", out, tmp_path / "configs")

    assert written.name == "toy-640.yaml"
    body = written.read_text(encoding="utf-8")
    assert str(out.resolve()) in body
    assert "person" in body and "helmet" in body and "vest" in body
