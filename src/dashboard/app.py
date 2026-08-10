"""The dashboard server: choose a source, draw the zone, watch the monitor work (O6).

The demonstration answers to a supervisor, not to a command line, so it needs a console: drop
in footage or point at a camera, mark the area where PPE is mandatory, and watch violations
appear with the evidence attached. That is what this serves — a local web page, no cloud, no
account, nothing leaving the machine.

**It drives the same loop as everything else.** The page does not re-implement detection,
association, zone membership or dwell; it consumes :func:`src.monitor.iterate` frame by frame.
A second copy of that loop is precisely how a dashboard ends up quietly disagreeing with the
numbers the dissertation reports, so there isn't one.

**Three sources, one abstraction.** An uploaded video file (reproducible, the default), a local
webcam by index, or a stream URL — an RTSP/MJPEG address, which is how a phone becomes a camera
and how a real site's IP cameras would be reached. They all resolve through
:func:`src.monitor.resolve_source`, so the demonstration claims nothing about the hardware
behind the lens.

**One run at a time**, deliberately: this is a single-operator console on one GPU, and pretending
otherwise would invite a queue nobody needs.

Usage::

    python -m src.dashboard                    # http://127.0.0.1:8000
    python -m src.dashboard --port 8080 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

from src.associate import CLASS_NAMES
from src.monitor import (
    DemoConfig,
    Summary,
    Violation,
    iterate,
    parse_required,
    resolve_source,
    source_fps,
    write_log,
)
from src.utils.logging import get_logger
from src.zone import Zone, save_zone

logger = get_logger(__name__)

DEFAULT_WORKDIR = Path("D:/runs/X06-dashboard")
DEFAULT_RUNS = Path("D:/runs")
MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # a demo clip, not an archive
STREAM_FPS = 25  # how often the browser is fed a frame; the monitor runs at its own pace


@dataclass
class Session:
    """Everything one console session holds. Guarded by :attr:`lock`."""

    workdir: Path
    source: str | None = None
    source_kind: str | None = None
    source_info: dict = field(default_factory=dict)
    first_frame_jpeg: bytes | None = None
    latest_jpeg: bytes | None = None
    running: bool = False
    stop_requested: bool = False
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    config: DemoConfig | None = None
    summary: Summary = field(default_factory=Summary)
    violations: list[Violation] = field(default_factory=list)
    in_breach: int = 0
    run_dir: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def out(self) -> Path:
        """Where this run writes. Each run gets its own folder.

        Runs are never overwritten and nothing is deleted: a console that showed the previous
        run's evidence frames beside this run's log would be worse than useless, and clearing
        the folder instead would throw away a record somebody may still want.
        """
        return self.run_dir or (self.workdir / "run")


def encode_jpeg(frame, quality: int = 80) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else b""


def grab_first_frame(source_spec: str):
    """Read one frame from any source, for the zone editor to draw on."""
    import cv2

    source = resolve_source(source_spec)
    if source.kind == "images":
        from src.zone import first_frame

        return first_frame(Path(source_spec))

    handle = source.handle if source.kind == "camera" else str(source.handle)
    capture = cv2.VideoCapture(handle)
    # A webcam needs a moment to expose; the first read is often black or fails outright.
    frame = None
    for _ in range(10):
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        time.sleep(0.1)
    capture.release()
    if frame is None:
        raise RuntimeError(f"could not read a frame from {source_spec}")
    return frame


def describe_source(source_spec: str, frame) -> dict:
    """Resolution, frame rate and length, for the header — and for honest timing later."""
    import cv2

    source = resolve_source(source_spec)
    height, width = frame.shape[:2]
    info = {
        "kind": source.kind,
        "label": source.label,
        "live": source.live,
        "width": int(width),
        "height": int(height),
        "fps": round(source_fps(source), 2),
        "frames": None,
        "duration": None,
    }
    if source.kind == "file":
        capture = cv2.VideoCapture(str(source.handle))
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        if count and count > 0:
            info["frames"] = int(count)
            info["duration"] = round(count / info["fps"], 1) if info["fps"] else None
    return info


def available_models(runs: Path = DEFAULT_RUNS) -> list[dict]:
    """Trained weights on this machine, newest run-ID first.

    Listed rather than typed, because pointing the demo at the wrong checkpoint is the easiest
    way to show numbers that do not match the chapter.
    """
    found = []
    for weights in sorted(runs.glob("X04-*/weights/best.pt")):
        run_id = weights.parent.parent.name
        model, _, dataset = run_id.replace("X04-", "").partition("-s")
        seed, _, trained_on = dataset.partition("-")
        found.append(
            {
                "value": str(weights),
                "run_id": run_id,
                "label": f"{model} · seed {seed} · trained on {trained_on.upper()}",
                "recommended": run_id == "X04-y8n-s0-sh17",
            }
        )
    return found


def run_session(session: Session) -> None:
    """Drive the monitor for one session, feeding the page as it goes."""
    try:
        for result in iterate(session.config, session.summary):
            jpeg = encode_jpeg(result.annotated)
            with session.lock:
                session.latest_jpeg = jpeg
                session.in_breach = result.in_breach
                session.violations.extend(result.violations)
                if session.stop_requested:
                    logger.info("stopped by the console at frame %d", result.index)
                    break
    except Exception as exc:  # an unplugged camera must not kill the server
        logger.exception("run failed")
        with session.lock:
            session.error = str(exc)
    finally:
        with session.lock:
            session.running = False
            session.finished_at = time.time()
        if session.violations:
            write_log(session.violations, session.out / "violations.csv")
        (session.out / "demo-metrics.json").write_text(
            json.dumps(
                {
                    "source": session.source_info,
                    "weights": str(session.config.weights),
                    "required_ppe": [CLASS_NAMES[c] for c in session.config.required_ppe],
                    "conf": session.config.conf,
                    "dwell_seconds": session.config.dwell_seconds,
                    "memory_seconds": session.config.memory_seconds,
                    "device": session.config.device or "auto",
                    "alerts": session.summary.alerts,
                    "people_detections": session.summary.people_detections,
                    "untracked_violations": session.summary.untracked_violations,
                    "latency": session.summary.stats(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("run finished: %s", session.summary.stats())


def create_app(workdir: Path = DEFAULT_WORKDIR, runs: Path = DEFAULT_RUNS) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "uploads").mkdir(exist_ok=True)
    session = Session(workdir=workdir)
    app.extensions["session"] = session

    @app.get("/")
    def index():
        return render_template("index.html", models=available_models(runs))

    @app.post("/api/source")
    def set_source():
        """Accept an uploaded file, a camera index or a stream URL, and return frame one."""
        if session.running:
            return jsonify({"error": "a run is in progress — stop it first"}), 409

        if "file" in request.files:
            upload = request.files["file"]
            if not upload.filename:
                return jsonify({"error": "no file chosen"}), 400
            target = workdir / "uploads" / Path(upload.filename).name
            upload.save(target)
            spec = str(target)
        else:
            payload = request.get_json(silent=True) or {}
            spec = str(payload.get("source", "")).strip()
            if not spec:
                return jsonify({"error": "no source given"}), 400

        try:
            frame = grab_first_frame(spec)
            info = describe_source(spec, frame)
        except Exception as exc:  # noqa: BLE001 — a bad URL is the user's mistake, not a 500
            logger.warning("source rejected: %s (%s)", spec, exc)
            return jsonify({"error": f"could not open that source: {exc}"}), 400

        with session.lock:
            session.source = spec
            session.source_kind = info["kind"]
            session.source_info = info
            session.first_frame_jpeg = encode_jpeg(frame, quality=88)
            session.latest_jpeg = None
            session.violations = []
            session.summary = Summary()
            session.error = None
        return jsonify(info)

    @app.get("/api/first-frame")
    def first_frame_image():
        if not session.first_frame_jpeg:
            return jsonify({"error": "no source yet"}), 404
        return Response(session.first_frame_jpeg, mimetype="image/jpeg")

    @app.post("/api/start")
    def start():
        if session.running:
            return jsonify({"error": "already running"}), 409
        if not session.source:
            return jsonify({"error": "choose a source first"}), 400

        payload = request.get_json(silent=True) or {}
        session.run_dir = workdir / "runs" / time.strftime("%Y%m%d-%H%M%S")
        points = payload.get("zone") or []
        try:
            required = parse_required(payload.get("required") or ["helmet", "vest"])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        zone_path = None
        if points:
            try:
                zone = Zone(
                    name=str(payload.get("zone_name") or "zone"),
                    points=tuple((float(x), float(y)) for x, y in points),
                    source=session.source,
                )
            except ValueError as exc:
                return jsonify({"error": f"zone rejected: {exc}"}), 400
            # Saved rather than held in memory: the zone is part of the record of a run, and
            # the same file re-runs it from the command line.
            zone_path = session.out / "zone.yaml"
            save_zone(zone, zone_path)

        config = DemoConfig(
            weights=Path(payload["weights"]),
            source=session.source,
            zone=zone_path,
            required_ppe=required,
            conf=float(payload.get("conf", 0.25)),
            dwell_seconds=float(payload.get("dwell", DemoConfig.dwell_seconds)),
            memory_seconds=float(payload.get("memory", DemoConfig.memory_seconds)),
            device=payload.get("device") or None,
            out=session.out,
            display=False,
            limit_frames=payload.get("limit") or None,
        )
        if not config.weights.exists():
            return jsonify({"error": f"weights not found: {config.weights}"}), 400

        with session.lock:
            session.config = config
            session.summary = Summary()
            session.violations = []
            session.latest_jpeg = None
            session.running = True
            session.stop_requested = False
            session.error = None
            session.started_at = time.time()
            session.finished_at = None
        threading.Thread(target=run_session, args=(session,), daemon=True).start()
        return jsonify({"started": True, "zone": str(zone_path) if zone_path else None})

    @app.post("/api/stop")
    def stop():
        with session.lock:
            session.stop_requested = True
        return jsonify({"stopping": True})

    @app.get("/api/status")
    def status():
        with session.lock:
            elapsed = (session.finished_at or time.time()) - (session.started_at or time.time())
            rows = [asdict(v) for v in session.violations[-200:]]
            return jsonify(
                {
                    "running": session.running,
                    "error": session.error,
                    "source": session.source_info,
                    "elapsed": round(elapsed, 1),
                    "frames": session.summary.frames,
                    "people": session.summary.people_detections,
                    "alerts": session.summary.alerts,
                    "in_breach": session.in_breach,
                    "untracked": session.summary.untracked_violations,
                    "latency": session.summary.stats(),
                    "violations": rows,
                    "has_frame": session.latest_jpeg is not None,
                }
            )

    @app.get("/api/stream")
    def stream():
        """MJPEG of the annotated frames — the live view."""

        def frames():
            blank = session.first_frame_jpeg or b""
            while True:
                with session.lock:
                    payload = session.latest_jpeg or blank
                    running = session.running
                if payload:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                if not running and session.latest_jpeg is None:
                    return
                time.sleep(1 / STREAM_FPS)

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/snapshot/<path:name>")
    def snapshot(name: str):
        path = session.out / "snapshots" / Path(name).name
        if not path.exists():
            return jsonify({"error": "no such snapshot"}), 404
        return send_file(path, mimetype="image/jpeg")

    @app.get("/api/violations.csv")
    def download_log():
        path = session.out / "violations.csv"
        if not path.exists():
            write_log(session.violations, path)
        return send_file(path, as_attachment=True, download_name="violations.csv")

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Zone compliance monitor — dashboard (O6).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()

    app = create_app(args.workdir, args.runs)
    logger.info("dashboard on http://%s:%d — press Ctrl+C to stop", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
