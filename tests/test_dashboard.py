"""Tests for the dashboard's own logic and its refusals.

The page mostly delegates: detection, association, zone and dwell all live in modules with
their own tests, and that is the point of it. What is tested here is what the server decides
by itself: how it reads the weights on disk, and that it refuses a run it cannot honestly
perform rather than starting one and failing halfway.
"""

from __future__ import annotations

import pytest

from src.dashboard.app import available_models, create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(workdir=tmp_path / "work", runs=tmp_path / "runs")
    app.config["TESTING"] = True
    return app.test_client()


def test_page_serves(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"PPE Zone Compliance Monitor" in response.data


def test_status_is_empty_before_anything_happens(client):
    body = client.get("/api/status").get_json()
    assert body["running"] is False
    assert body["frames"] == 0
    assert body["violations"] == []


def test_starting_without_a_source_is_refused(client):
    response = client.post("/api/start", json={"weights": "w.pt"})
    assert response.status_code == 400
    assert "source" in response.get_json()["error"]


def test_a_source_that_cannot_be_opened_is_refused(client):
    """A wrong path, a missing camera or a dead stream must come back as a message, not a 500.

    Tested with a path that does not exist rather than an unreachable stream URL: the failure
    is the same one, and waiting out a network timeout in a unit test is a minute of nothing.
    """
    response = client.post("/api/source", json={"source": "D:/no/such/clip.mp4"})
    assert response.status_code == 400
    assert "could not open" in response.get_json()["error"]


def test_no_source_means_no_first_frame(client):
    assert client.get("/api/first-frame").status_code == 404


def _fade_in_clip(path, blank_frames=40, size=(160, 120)):
    """A clip that opens on black and fades up, the way real footage often does."""
    import cv2
    import numpy as np

    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for index in range(blank_frames + 20):
        level = 0 if index < blank_frames else 180
        writer.write(np.full((height, width, 3), level, dtype=np.uint8))
    writer.release()
    return path


def test_the_zone_editor_is_not_handed_a_black_frame(tmp_path):
    """Footage that fades in must not put the operator in front of an empty rectangle.

    Marking a safety zone means pointing at a place, so the frame the editor draws on has to
    show one. Taking whatever decodes first fails exactly on the footage most likely to be
    used, because a fade-in is a convention of edited video, not an edge case.
    """
    from src.dashboard.app import WELL_LIT_MEAN, grab_first_frame

    clip = _fade_in_clip(tmp_path / "fade.mp4")
    assert float(grab_first_frame(str(clip)).mean()) >= WELL_LIT_MEAN


def test_a_clip_that_is_black_throughout_still_returns_a_frame(tmp_path):
    """No usable frame is a reason to show the best one, never to refuse the source."""
    from src.dashboard.app import grab_first_frame

    clip = _fade_in_clip(tmp_path / "dark.mp4", blank_frames=120)
    assert grab_first_frame(str(clip)) is not None


def test_available_models_reads_run_ids(tmp_path):
    for run_id in ("X04-y8n-s0-sh17", "X04-y11n-s2-chv"):
        weights = tmp_path / run_id / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"")

    models = available_models(tmp_path)
    labels = {model["run_id"]: model["label"] for model in models}
    # Model, seed and training set, in the order the run-IDs already use. Kept short because
    # the option text carries a "recommended" marker after it and the picker is one column wide.
    assert labels["X04-y8n-s0-sh17"] == "y8n · seed 0 · SH17"
    assert labels["X04-y11n-s2-chv"] == "y11n · seed 2 · CHV"
    # The demo's agreed detector is pre-selected, so the console cannot quietly open on a
    # checkpoint the dissertation does not report.
    assert [m["run_id"] for m in models if m["recommended"]] == ["X04-y8n-s0-sh17"]


def test_available_models_is_empty_when_nothing_is_trained(tmp_path):
    assert available_models(tmp_path) == []
