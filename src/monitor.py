"""The compliance monitor: detector → tracker → association rule → violation log (O6).

At S2 this exists to prove the wiring end to end — weights load, ByteTrack keeps IDs across
frames, the association rule turns boxes into per-person compliance, and a violation reaches
a log row and a snapshot. S6 grows it into the demo proper (user-drawn zone, dwell threshold,
debouncing, webcam input); the pieces it will build on are already in place here.

Two things are deliberately shared rather than re-implemented: the **association rule**
(`src.associate`) is the same code the violation axis is scored with, so the demo cannot
quietly disagree with the results chapter, and the harmonised class ids are the same three.

A violation row is written the first time a tracked person is seen missing required PPE, and
then not again for that person — a monitor that logs every frame produces thousands of rows
for one incident. S6 replaces that with dwell + debounce proper.

Usage::

    python -m src.monitor --weights D:/runs/<run-id>/weights/best.pt --source <video|dir>
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.associate import CLASS_NAMES, THRESHOLD, Box, associate
from src.utils.logging import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TRACKER = "bytetrack.yaml"  # C9 (Zhang 2022): detector-agnostic, real-time, occlusion-robust
DEFAULT_CONF = 0.25  # Ultralytics default; the deployment operating point is chosen at S6


@dataclass
class Violation:
    """One logged non-compliance event."""

    timestamp: str
    frame: int
    track_id: int
    missing: str
    snapshot: str


def frames_from(source: Path):
    """Yield ``(index, frame)`` from a video file or a directory of images.

    Image directories are supported because the demo has to be testable without a site video,
    and because the frozen eval splits are the only PPE footage we are certain of.
    """
    import cv2

    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is not None:
                yield index, image
        return

    capture = cv2.VideoCapture(str(source))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield index, frame
        index += 1
    capture.release()


def boxes_from_result(result) -> tuple[list[Box], list[int | None]]:
    """Convert one Ultralytics tracking result into normalised boxes + their track ids."""
    boxes: list[Box] = []
    track_ids: list[int | None] = []
    if result.boxes is None:
        return boxes, track_ids

    ids = result.boxes.id
    for position, row in enumerate(result.boxes.xywhn):
        xc, yc, w, h = (float(v) for v in row)
        boxes.append(Box(int(result.boxes.cls[position]), xc, yc, w, h))
        track_ids.append(int(ids[position]) if ids is not None else None)
    return boxes, track_ids


def monitor(
    weights: Path,
    source: Path,
    out_dir: Path,
    threshold: float = THRESHOLD,
    conf: float = DEFAULT_CONF,
    tracker: str = TRACKER,
) -> list[Violation]:
    """Run the monitor over a source and log the violations it finds.

    ``conf`` is the detector's confidence gate. It is a real deployment dial — too high and
    violations are missed, too low and the log fills with noise — so it is a parameter rather
    than a constant, and S6 sets it from the evaluated operating point.
    """
    import cv2
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = out_dir / "snapshots"
    snapshots.mkdir(exist_ok=True)

    model = YOLO(str(weights))
    violations: list[Violation] = []
    already_logged: set[int] = set()
    frames_seen = 0
    people_seen = 0
    untracked = 0

    for index, frame in frames_from(source):
        frames_seen += 1
        result = model.track(frame, persist=True, tracker=tracker, conf=conf, verbose=False)[0]
        boxes, track_ids = boxes_from_result(result)
        assignment = associate(boxes, threshold=threshold)
        people_seen += len(assignment.people)

        for person in assignment.people:
            track_id = track_ids[person.index]
            if not person.violations:
                continue
            # Dedup by identity — but only when there *is* one. A missing track id means the
            # tracker could not confirm this detection, and treating every such person as the
            # same "None" identity would silently log one violation for a whole site.
            if track_id is None:
                untracked += 1
            elif track_id in already_logged:
                continue
            else:
                already_logged.add(track_id)

            snapshot = snapshots / f"frame{index:05d}-id{track_id}.jpg"
            cv2.imwrite(str(snapshot), result.plot())
            violation = Violation(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                frame=index,
                track_id=track_id if track_id is not None else -1,
                missing="+".join(CLASS_NAMES[c] for c in person.violations),
                snapshot=str(snapshot),
            )
            violations.append(violation)
            logger.info(
                "VIOLATION frame %d · track %s · missing %s",
                index,
                track_id,
                violation.missing,
            )

    logger.info(
        "%d frames · %d person detections · %d violations logged",
        frames_seen,
        people_seen,
        len(violations),
    )
    if untracked:
        logger.warning(
            "%d violating people had no track id — the tracker could not confirm them, so "
            "they are logged per detection and cannot be de-duplicated across frames",
            untracked,
        )
    return violations


def write_log(violations: list[Violation], path: Path) -> None:
    """Write the timestamped violation log (the artefact a site manager would keep)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "frame", "track_id", "missing_ppe", "snapshot"])
        for violation in violations:
            writer.writerow(
                [
                    violation.timestamp,
                    violation.frame,
                    violation.track_id,
                    violation.missing,
                    violation.snapshot,
                ]
            )
    logger.info("violation log: %s (%d rows)", path, len(violations))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compliance monitor (S2 skeleton of the O6 demo).")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="video file or image directory")
    parser.add_argument("--out", type=Path, default=Path("D:/runs/monitor"))
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="association rule τ")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="detector confidence gate")
    parser.add_argument("--tracker", default=TRACKER, help="tracker config (ByteTrack by default)")
    parser.add_argument("--limit", type=int, default=None, help="stop after N violations")
    args = parser.parse_args()

    if not args.weights.exists():
        logger.error("weights not found: %s", args.weights)
        return 1

    violations = monitor(
        args.weights, args.source, args.out, args.threshold, args.conf, args.tracker
    )
    if args.limit:
        violations = violations[: args.limit]
    write_log(violations, args.out / "violations.csv")

    if not violations:
        logger.error("no violation logged — the demo path produced no output to check")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
