"""S6.1 the safety zone: drawing it, storing it, and deciding who is standing in it (O6).

The demonstration's premise is that a site manager marks the area where PPE is mandatory and
the system watches *that area*, not the whole frame. Everything downstream, the dwell timer
(S6.2), the alert and its log row (S6.3), is gated on one question this module answers:
**is this person inside the zone?**

Three decisions are worth stating, because each one changes the answer.

**Coordinates are normalised, never pixels.** A zone is stored as fractions of frame width and
height, so the same file governs a 4K clip and a 640-pixel still, and so a zone drawn once
survives a change of source resolution. It also lets the S6.5 evaluation place a zone over
Pictor images, which vary in size image to image; pixel coordinates would silently mean a
different region in every one of them. A file whose points fall outside ``[0, 1]`` is rejected
on load rather than trusted, because pixel coordinates read as normalised ones put every
worker outside the zone and would look like a quiet, plausible "no violations".

**Membership is decided by the feet, not the centre.** A person is in the zone if the bottom
centre of their box (where they stand) is inside the polygon. Using the box centre would put
a worker leaning over a barrier inside the restricted area while both feet are outside it, and
would drag tall foreground figures in from the edge of the frame. The feet point is the
standard ground-contact proxy for a monocular view; it is wrong when a worker is occluded from
the knees down, which is a stated limit rather than a hidden one.

**The boundary is inside.** A point exactly on a drawn edge belongs to the zone. Ray casting
is undefined on the boundary and would otherwise decide such points by floating-point accident,
which is precisely the case a synthetic test hits first.

Usage::

    python -m src.zone --source clip.mp4 --name loading-bay      # draw it, save it, preview it
    python -m src.zone --source clip.mp4 --name loading-bay --show   # re-open an existing zone
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.associate import Box
from src.utils.logging import get_logger

logger = get_logger(__name__)

Point = tuple[float, float]

DEFAULT_ZONE_DIR = Path("configs/zones")
# Slack allowed when asking whether a point sits *on* an edge, in normalised units: half a
# pixel of a 640-wide frame. Tight enough not to widen the zone, loose enough that a point
# read back from a rounded file still lands on the boundary it was drawn on.
EDGE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Zone:
    """A polygon in normalised frame coordinates, plus where it came from.

    Attributes:
        name: Human label used in the alert banner and the log row ("loading-bay").
        points: Vertices in draw order, each ``(x, y)`` in ``[0, 1]``. Closing the ring is
            implicit: the last vertex joins the first.
        source: The video or image the polygon was drawn on, kept so a zone can be traced
            back to the frame that justified it.
        created: UTC timestamp of the drawing session.
    """

    name: str
    points: tuple[Point, ...]
    source: str | None = None
    created: str | None = None

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"a zone needs at least 3 points, got {len(self.points)}")
        for x, y in self.points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"zone point ({x}, {y}) is outside [0, 1]; points are normalised "
                    "fractions of frame size, not pixels"
                )

    def contains(self, point: Point) -> bool:
        """Whether a normalised point lies inside the polygon (boundary counts as inside)."""
        return point_in_polygon(point, self.points)

    def contains_box(self, box: Box) -> bool:
        """Whether the person this box describes is standing in the zone.

        Judged on :func:`feet_point`, so the answer is about where they stand rather than
        where the middle of their bounding box happens to fall.
        """
        return self.contains(feet_point(box))

    def to_pixels(self, width: int, height: int) -> list[tuple[int, int]]:
        """The polygon in pixel coordinates, for drawing it over a frame."""
        return [(round(x * width), round(y * height)) for x, y in self.points]


def feet_point(box: Box) -> Point:
    """The ground-contact proxy for a person box: bottom edge, horizontal centre."""
    return (box.xc, box.yc + box.h / 2)


def _on_segment(point: Point, a: Point, b: Point, tolerance: float = EDGE_TOLERANCE) -> bool:
    """Whether ``point`` lies on the segment ``a``–``b`` within ``tolerance``."""
    (px, py), (ax, ay), (bx, by) = point, a, b
    # Distance from the point to the infinite line, via the cross product, then a bounds
    # check so the rest of that line does not count.
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    if length == 0:  # a degenerate edge is just a vertex
        return abs(px - ax) <= tolerance and abs(py - ay) <= tolerance
    if abs(cross) / length > tolerance:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    return -tolerance <= dot <= length**2 + tolerance


def point_in_polygon(point: Point, polygon: tuple[Point, ...] | list[Point]) -> bool:
    """Even-odd ray casting, with the boundary treated as inside.

    A horizontal ray is cast to the right and its crossings of the polygon's edges are
    counted; an odd count means inside. Each edge is treated as half-open in *y* so a ray
    passing exactly through a vertex is counted once rather than twice, the classic
    double-count that reports an interior point as outside.

    Concave polygons are handled by construction: a site zone is often an L around machinery,
    and the notch of an L must read as outside.
    """
    vertices = list(polygon)
    if len(vertices) < 3:
        return False

    for index, vertex in enumerate(vertices):
        if _on_segment(point, vertex, vertices[(index + 1) % len(vertices)]):
            return True

    x, y = point
    inside = False
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        if (y1 > y) != (y2 > y):  # the edge straddles the ray's height
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing_x > x:
                inside = not inside
    return inside


# ------------------------------------------------------------------------------ persistence


def save_zone(zone: Zone, path: Path) -> None:
    """Write a zone to YAML so the same region can be reloaded run after run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": zone.name,
                "points": [[round(x, 6), round(y, 6)] for x, y in zone.points],
                "source": zone.source,
                "created": zone.created,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger.info("zone saved: %s (%d points)", path, len(zone.points))


def load_zone(path: Path) -> Zone:
    """Read a zone back. Raises if the file is not a usable polygon."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "points" not in data:
        raise ValueError(f"{path} is not a zone file (no `points` key)")
    return Zone(
        name=data.get("name") or path.stem,
        points=tuple((float(x), float(y)) for x, y in data["points"]),
        source=data.get("source"),
        created=data.get("created"),
    )


def zone_path(name: str, directory: Path = DEFAULT_ZONE_DIR) -> Path:
    return directory / f"{name}.yaml"


# --------------------------------------------------------------------------------- drawing


def first_frame(source: Path):
    """The frame the zone is drawn on: frame 0 of a video, or the image itself.

    Read here with cv2 rather than borrowed from :mod:`src.monitor`, so that the monitor can
    depend on zones without the two modules importing each other.
    """
    import cv2

    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() != ".txt")
        if not images:
            raise FileNotFoundError(f"no images in {source}")
        return cv2.imread(str(images[0]))
    if source.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return cv2.imread(str(source))

    capture = cv2.VideoCapture(str(source))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise FileNotFoundError(f"could not read a frame from {source}")
    return frame


def draw_zone(frame, name: str, source: str | None = None) -> Zone | None:
    """Let the user click a polygon onto the frame. Returns ``None`` if they cancel.

    Left click adds a point, right click (or ``u``) removes the last, ``Enter`` accepts once
    three points exist, ``r`` restarts, ``Esc`` cancels. Interactive by nature and therefore
    not unit-tested; every decision it makes about *membership* lives in the pure functions
    above, which are.
    """
    import cv2

    height, width = frame.shape[:2]
    clicked: list[tuple[int, int]] = []
    window = f"draw zone: {name}"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and clicked:
            clicked.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    logger.info(
        "click the zone corners · Enter accepts · u/right-click undoes · r restarts · Esc cancels"
    )

    accepted = False
    while True:
        canvas = frame.copy()
        if clicked:
            import numpy as np

            points = np.array(clicked, dtype=np.int32)
            if len(clicked) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [points], (0, 180, 0))
                canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
            cv2.polylines(canvas, [points], len(clicked) >= 3, (0, 220, 0), 2)
            for point in clicked:
                cv2.circle(canvas, point, 4, (0, 220, 0), -1)
        cv2.putText(
            canvas,
            f"{name}: {len(clicked)} points | Enter=accept  u=undo  r=restart  Esc=cancel",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
        )
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc
            break
        if key == ord("u") and clicked:
            clicked.pop()
        elif key == ord("r"):
            clicked.clear()
        elif key in (13, 10) and len(clicked) >= 3:  # Enter
            accepted = True
            break
    cv2.destroyWindow(window)

    if not accepted:
        logger.warning("zone drawing cancelled; nothing saved")
        return None
    return Zone(
        name=name,
        points=tuple((x / width, y / height) for x, y in clicked),
        source=source,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def preview(frame, zone: Zone, path: Path) -> None:
    """Save the frame with the zone drawn on it: the evidence that it landed where intended."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    points = np.array(zone.to_pixels(width, height), dtype=np.int32)
    canvas = frame.copy()
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [points], (0, 180, 0))
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
    cv2.polylines(canvas, [points], True, (0, 220, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    logger.info("zone preview: %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw, save and preview a safety zone (S6.1).")
    parser.add_argument("--source", type=Path, required=True, help="video, image or image dir")
    parser.add_argument("--name", default="zone", help="zone name (also the file stem)")
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONE_DIR)
    parser.add_argument("--out", type=Path, default=None, help="where to write the preview image")
    parser.add_argument(
        "--show",
        action="store_true",
        help="preview the saved zone instead of drawing a new one",
    )
    args = parser.parse_args()

    if not args.source.exists():
        logger.error("source not found: %s", args.source)
        return 1

    frame = first_frame(args.source)
    path = zone_path(args.name, args.zones)

    if args.show:
        if not path.exists():
            logger.error("no zone at %s; draw one first (drop --show)", path)
            return 1
        zone = load_zone(path)
    else:
        zone = draw_zone(frame, args.name, source=str(args.source))
        if zone is None:
            return 1
        save_zone(zone, path)
        # A zone is only usable if it survives the round trip, so prove it here rather than
        # discovering at S6.5 that the saved file means something else.
        reloaded = load_zone(path)
        if reloaded.points != zone.points:
            logger.error("zone did not survive the round trip: %s", path)
            return 1

    preview(frame, zone, args.out or Path("D:/runs/X06-demo") / f"zone-{args.name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
