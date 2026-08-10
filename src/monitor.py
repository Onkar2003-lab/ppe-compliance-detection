"""The zone compliance monitor: the O6 demonstration (S6.3 outputs, S6.4 inputs + config).

A site manager marks the area where PPE is mandatory; the monitor watches that area and
records anyone standing in it without the required equipment. That is the whole product, and
it is assembled from parts that are already evidenced rather than invented here:

* the **detector** is the trio's weights, run at the same confidence the violation axis was
  scored at, so the demo behaves like the numbers in the results chapter;
* the **tracker** is ByteTrack (C9), which gives each worker an identity so an incident can be
  one incident rather than one row per frame;
* the **association rule** is imported from :mod:`src.associate` — the same code the violation
  axis is scored with, never a copy, so the demo cannot quietly disagree with the chapter;
* the **zone** (:mod:`src.zone`) and the **dwell/debounce timer** (:mod:`src.dwell`) are pure
  and separately tested, which is what lets S6.5 evidence them exactly.

What this module adds is the wiring and the outputs a manager would actually keep: a live
banner, a timestamped CSV log, and a saved frame at the moment of each violation.

**Input is a generic video source** — a recorded file (the reproducible default), a webcam
index, an RTSP/HTTP stream, or a directory of stills. One interface covers all four, so the
demo makes no claim about the camera it is attached to.

**Time comes from the source, not the clock, whenever the source has its own.** A recorded
file timestamps frames at ``index / fps``, so replaying a clip produces byte-identical alerts;
a live camera has no such thing and uses the wall clock.

**Nothing here re-measures detection accuracy.** That is the violation axis (F27), scored on
Pictor with ground truth. This module is evaluated on the layer that is new: latency, and the
zone/dwell logic its tests pin exactly.

Usage::

    python -m src.monitor --config configs/demo.yaml
    python -m src.monitor --config configs/demo.yaml --source 0 --display
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.associate import CLASS_NAMES, PERSON, THRESHOLD, Box, PersonState, associate
from src.dwell import DEFAULT_DWELL_SECONDS, DEFAULT_GRACE_SECONDS, Alert, DwellTracker
from src.utils.logging import get_logger
from src.zone import Zone, load_zone

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")
TRACKER = "bytetrack.yaml"  # C9 (Zhang 2022): detector-agnostic, real-time, occlusion-robust

# The operating point the violation axis was scored at (S5). Sharing it means the demo's
# behaviour is the behaviour the results chapter reports, rather than an unmeasured one.
DEFAULT_CONF = 0.25
DEFAULT_FPS = 25.0  # only used when a file declines to report its own frame rate
WARMUP_FRAMES = 15  # discarded before latency is summarised, matching src.efficiency

BY_NAME = {name: cls for cls, name in CLASS_NAMES.items()}
COMPLIANT_COLOUR = (0, 200, 0)
VIOLATION_COLOUR = (0, 0, 235)
ZONE_COLOUR = (0, 220, 220)


# ---------------------------------------------------------------------------------- config


@dataclass
class DemoConfig:
    """Every dial the demo has, read from YAML so no path or threshold is hardcoded."""

    weights: Path
    source: str
    zone: Path | None = None
    required_ppe: tuple[int, ...] = (BY_NAME["helmet"], BY_NAME["vest"])
    conf: float = DEFAULT_CONF
    association_threshold: float = THRESHOLD
    tracker: str = TRACKER
    track: bool | None = None  # None = track video and live sources, not directories of stills
    dwell_seconds: float = DEFAULT_DWELL_SECONDS
    grace_seconds: float = DEFAULT_GRACE_SECONDS
    out: Path = Path("D:/runs/X06-demo")
    display: bool = True
    save_snapshots: bool = True
    fps: float | None = None  # override the source's declared frame rate
    limit_frames: int | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> DemoConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> DemoConfig:
        data = dict(data)
        required = data.pop("required_ppe", None)
        zone = data.pop("zone", None)
        config = cls(
            weights=Path(data.pop("weights")),
            source=str(data.pop("source")),
            zone=Path(zone) if zone else None,
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
        )
        if required is not None:
            config.required_ppe = parse_required(required)
        config.out = Path(config.out)
        return config


def parse_required(names) -> tuple[int, ...]:
    """Turn ``["helmet", "vest"]`` into class ids, rejecting anything not trained.

    A typo here would silently mean "this PPE is never required", so an unknown name is an
    error rather than a shrug.
    """
    ids = []
    for name in names:
        key = str(name).strip().lower()
        if key not in BY_NAME or BY_NAME[key] == PERSON:
            raise ValueError(
                f"required PPE {name!r} is not in the trained label space "
                f"{sorted(n for n, c in BY_NAME.items() if c != PERSON)}"
            )
        ids.append(BY_NAME[key])
    return tuple(dict.fromkeys(ids))  # de-duplicated, order kept


# ---------------------------------------------------------------------------------- source


@dataclass
class Source:
    """A resolved video source: what it is, how to read it, and whether it has its own clock."""

    kind: str  # "file" | "images" | "camera" | "stream"
    handle: object  # a path, a directory, a camera index or a URL
    live: bool

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.handle}"

    @property
    def stills(self) -> bool:
        """Whether the frames are unrelated images rather than a continuous sequence."""
        return self.kind == "images"


def resolve_source(spec: str) -> Source:
    """Decide what a source string means, without touching the device.

    Kept separate from reading so the rule is testable: a bare integer is a camera index, a
    URL is a stream, a directory is stills, and anything else is a file on disk.
    """
    text = str(spec).strip()
    if text.isdigit():
        return Source(kind="camera", handle=int(text), live=True)
    if text.lower().startswith(STREAM_PREFIXES):
        return Source(kind="stream", handle=text, live=True)
    path = Path(text)
    if path.is_dir():
        return Source(kind="images", handle=path, live=False)
    return Source(kind="file", handle=path, live=False)


def frames_from(source: Source, limit: int | None = None):
    """Yield ``(index, frame)`` from any resolved source."""
    import cv2

    if source.kind == "images":
        paths = sorted(p for p in source.handle.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for index, path in enumerate(paths):
            if limit is not None and index >= limit:
                return
            image = cv2.imread(str(path))
            if image is not None:
                yield index, image
        return

    handle = source.handle if source.kind == "camera" else str(source.handle)
    capture = cv2.VideoCapture(handle)
    index = 0
    try:
        while True:
            if limit is not None and index >= limit:
                return
            ok, frame = capture.read()
            if not ok:
                return
            yield index, frame
            index += 1
    finally:
        capture.release()


def source_fps(source: Source, override: float | None = None) -> float:
    """Frames per second of a recorded source, for turning frame indices into seconds."""
    if override:
        return float(override)
    if source.kind in ("file",):
        import cv2

        capture = cv2.VideoCapture(str(source.handle))
        declared = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if declared and declared > 0:
            return float(declared)
    return DEFAULT_FPS


# ------------------------------------------------------------------------------ per-frame


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


def still_ids(index: int, count: int) -> list[int]:
    """Identities for an untracked frame of stills: unique per image, so nothing debounces.

    Each photograph is its own incident. Giving its people ids that cannot recur means every
    violating worker is alerted once and none is suppressed as a repeat of someone in an
    unrelated picture.
    """
    return [index * 1000 + position for position in range(count)]


def missing_for(person: PersonState, required: tuple[int, ...]) -> tuple[int, ...]:
    """Required PPE this person is not wearing.

    Deliberately not :attr:`PersonState.violations`: that property answers against the frozen
    requirement the violation axis is scored with, while a zone may require less (a helmet-only
    area) or the same. The association itself is untouched — only what counts as required.
    """
    return tuple(cls for cls in required if not person.wears(cls))


def violations_in_zone(
    people: list[PersonState],
    track_ids: list[int | None],
    zone: Zone | None,
    required: tuple[int, ...],
) -> tuple[dict[int, tuple[int, ...]], int]:
    """Who, of the tracked people standing in the zone, is missing required PPE.

    Returns the mapping the dwell timer consumes, plus a count of violating people the tracker
    could not give an identity to. Those are reported rather than folded in: without an id
    there is nothing to debounce against, and treating them all as one person would collapse a
    whole site into a single incident — the defect the S2 skeleton exposed.
    """
    violating: dict[int, tuple[int, ...]] = {}
    untracked = 0
    for person in people:
        if zone is not None and not zone.contains_box(person.box):
            continue
        missing = missing_for(person, required)
        if not missing:
            continue
        track_id = track_ids[person.index]
        if track_id is None:
            untracked += 1
            continue
        violating[track_id] = missing
    return violating, untracked


# ---------------------------------------------------------------------------------- output


@dataclass
class Violation:
    """One logged incident — the row a site manager keeps."""

    timestamp: str
    frame: int
    seconds: float
    track_id: int
    zone: str
    missing_ppe: str
    dwell_seconds: float
    snapshot: str


LOG_COLUMNS = (
    "timestamp",
    "frame",
    "seconds",
    "track_id",
    "zone",
    "missing_ppe",
    "dwell_seconds",
    "snapshot",
)


def write_log(violations: list[Violation], path: Path) -> None:
    """Write the timestamped violation log (CSV, so it opens in whatever the site uses)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for violation in violations:
            writer.writerow(asdict(violation))
    logger.info("violation log: %s (%d rows)", path, len(violations))


def draw_overlay(
    frame,
    zone: Zone | None,
    people: list[PersonState],
    track_ids: list[int | None],
    required: tuple[int, ...],
    banner: str,
):
    """Draw the zone, each person's compliance state, and the alert banner onto a frame."""
    import cv2
    import numpy as np

    canvas = frame.copy()
    height, width = canvas.shape[:2]

    if zone is not None:
        points = np.array(zone.to_pixels(width, height), dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [points], ZONE_COLOUR)
        canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
        cv2.polylines(canvas, [points], True, ZONE_COLOUR, 2)

    for person in people:
        inside = zone is None or zone.contains_box(person.box)
        missing = missing_for(person, required)
        colour = VIOLATION_COLOUR if (inside and missing) else COMPLIANT_COLOUR
        x1, y1, x2, y2 = person.box.corners
        top_left = (int(x1 * width), int(y1 * height))
        bottom_right = (int(x2 * width), int(y2 * height))
        cv2.rectangle(canvas, top_left, bottom_right, colour, 2)
        if not inside:
            continue
        track_id = track_ids[person.index]
        label = "+".join(CLASS_NAMES[c] for c in missing) if missing else "compliant"
        cv2.putText(
            canvas,
            f"#{track_id if track_id is not None else '?'} {label}",
            (top_left[0], max(16, top_left[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2,
        )

    if banner:
        cv2.rectangle(canvas, (0, 0), (width, 34), (0, 0, 0), -1)
        cv2.putText(canvas, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, VIOLATION_COLOUR, 2)
    return canvas


# ------------------------------------------------------------------------------------- run


@dataclass
class Summary:
    """What one demo run measured — the input to S6.5 and to Methodology 4.9."""

    frames: int = 0
    people_detections: int = 0
    alerts: int = 0
    untracked_violations: int = 0
    latencies_ms: list[float] = field(default_factory=list, repr=False)

    def stats(self) -> dict:
        """Latency summary over the frames after warm-up.

        The first frames pay for CUDA context creation and kernel selection — 88 ms against a
        17 ms median in the first smoke run — so they are discarded exactly as the efficiency
        axis discards its warm-up, and for the same reason: they measure starting the program,
        not running it. Median leads, so one stalled frame does not set the figure.
        """
        measured = self.latencies_ms[WARMUP_FRAMES:] or self.latencies_ms
        if not measured:
            return {}
        ordered = sorted(measured)
        mean = sum(ordered) / len(ordered)
        median = ordered[len(ordered) // 2]
        return {
            "frames": self.frames,
            "measured_frames": len(ordered),
            "mean_ms": round(mean, 2),
            "median_ms": round(median, 2),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
            "mean_fps": round(1000.0 / mean, 2),
            "median_fps": round(1000.0 / median, 2),
        }


def run(config: DemoConfig) -> tuple[list[Violation], Summary]:
    """Drive the monitor over its source and return the violations and the run summary."""
    import cv2
    from ultralytics import YOLO

    source = resolve_source(config.source)
    zone = load_zone(config.zone) if config.zone else None
    if zone is None:
        logger.warning("no zone configured — every person in frame is treated as in scope")

    fps = source_fps(source, config.fps)
    config.out.mkdir(parents=True, exist_ok=True)
    snapshots = config.out / "snapshots"
    snapshots.mkdir(exist_ok=True)

    model = YOLO(str(config.weights))
    tracking = config.track if config.track is not None else not source.stills
    dwell_seconds = config.dwell_seconds
    if not tracking:
        # A tracker over unrelated stills is worse than no tracker: ByteTrack only returns
        # detections it has confirmed across consecutive frames, so on a directory of separate
        # images it silently withholds most boxes for a frame — and a withheld helmet reads as
        # a bare head. Each image is therefore scored on its own, which also makes dwell
        # meaningless, since there is no elapsed time between two unrelated photographs.
        logger.info("stills source: detecting per image, no tracking and no dwell")
        dwell_seconds = 0.0
    timer = DwellTracker(
        dwell_seconds=dwell_seconds,
        grace_seconds=config.grace_seconds,
        zone=zone.name if zone else "frame",
    )
    violations: list[Violation] = []
    summary = Summary()
    started = time.perf_counter()
    banner = ""

    for index, frame in frames_from(source, config.limit_frames):
        began = time.perf_counter()
        now = (time.perf_counter() - started) if source.live else index / fps

        if tracking:
            result = model.track(
                frame, persist=True, tracker=config.tracker, conf=config.conf, verbose=False
            )[0]
            boxes, track_ids = boxes_from_result(result)
        else:
            result = model.predict(frame, conf=config.conf, verbose=False)[0]
            boxes, _ = boxes_from_result(result)
            track_ids = still_ids(index, len(boxes))

        assignment = associate(boxes, threshold=config.association_threshold)
        violating, untracked = violations_in_zone(
            assignment.people, track_ids, zone, config.required_ppe
        )
        alerts = timer.update(now, violating)

        summary.frames += 1
        summary.people_detections += len(assignment.people)
        summary.untracked_violations += untracked

        # Rebuilt every frame from who is currently in breach, so the banner shows the state
        # now rather than the last thing that happened to fire.
        banner = "  ".join(
            f"#{track_id} missing {'+'.join(CLASS_NAMES[c] for c in missing)}"
            for track_id, missing in sorted(violating.items())
        )
        annotated = draw_overlay(
            frame, zone, assignment.people, track_ids, config.required_ppe, banner
        )

        for alert in alerts:
            violations.append(record(alert, index, now, annotated, snapshots, config))
            logger.info(
                "VIOLATION frame %d · t=%.2fs · track %d · missing %s · dwell %.2fs",
                index,
                now,
                alert.track_id,
                alert.missing_names,
                alert.dwell,
            )
        summary.alerts += len(alerts)
        # Timed here so the figure covers everything a deployment pays for — detection,
        # tracking, association, zone, dwell and drawing — not the detector alone.
        summary.latencies_ms.append((time.perf_counter() - began) * 1000)

        if config.display:
            cv2.imshow("zone compliance monitor", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                logger.info("stopped by user at frame %d", index)
                break

    if config.display:
        cv2.destroyAllWindows()

    if summary.untracked_violations:
        logger.warning(
            "%d violating people had no track id — the tracker could not confirm them, so they "
            "are counted but not alerted on (there is no identity to debounce against)",
            summary.untracked_violations,
        )
    logger.info(
        "%d frames · %d person detections · %d alerts · %s",
        summary.frames,
        summary.people_detections,
        summary.alerts,
        summary.stats(),
    )
    return violations, summary


def record(
    alert: Alert,
    index: int,
    seconds: float,
    annotated,
    snapshots: Path,
    config: DemoConfig,
) -> Violation:
    """Save the evidence frame and build the log row for one alert."""
    import cv2

    path = snapshots / f"frame{index:06d}-track{alert.track_id}.jpg"
    if config.save_snapshots:
        cv2.imwrite(str(path), annotated)
    return Violation(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        frame=index,
        seconds=round(seconds, 3),
        track_id=alert.track_id,
        zone=alert.zone,
        missing_ppe=alert.missing_names,
        dwell_seconds=round(alert.dwell, 3),
        snapshot=str(path) if config.save_snapshots else "",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Zone compliance monitor (O6 demonstration).")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--source", default=None, help="override: file | 0 | rtsp://… | dir")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--zone", type=Path, default=None)
    parser.add_argument("--dwell", type=float, default=None, help="seconds before an alert")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="stop after N frames")
    parser.add_argument("--display", dest="display", action="store_true", default=None)
    parser.add_argument("--no-display", dest="display", action="store_false")
    args = parser.parse_args()

    if not args.config.exists():
        logger.error("no config at %s — copy configs/demo.yaml and edit it", args.config)
        return 1
    config = DemoConfig.from_yaml(args.config)
    for name, value in (
        ("source", args.source),
        ("weights", args.weights),
        ("zone", args.zone),
        ("dwell_seconds", args.dwell),
        ("out", args.out),
        ("limit_frames", args.limit),
        ("display", args.display),
    ):
        if value is not None:
            setattr(config, name, value)

    if not config.weights.exists():
        logger.error("weights not found: %s", config.weights)
        return 1
    if config.zone and not Path(config.zone).exists():
        logger.error("zone not found: %s — draw one with `python -m src.zone`", config.zone)
        return 1

    violations, summary = run(config)
    write_log(violations, config.out / "violations.csv")
    (config.out / "demo-metrics.json").write_text(
        json.dumps(
            {
                "source": resolve_source(config.source).label,
                "weights": str(config.weights),
                "zone": str(config.zone) if config.zone else None,
                "required_ppe": [CLASS_NAMES[c] for c in config.required_ppe],
                "conf": config.conf,
                "dwell_seconds": config.dwell_seconds,
                "alerts": summary.alerts,
                "people_detections": summary.people_detections,
                "untracked_violations": summary.untracked_violations,
                "latency": summary.stats(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("metrics: %s", config.out / "demo-metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
