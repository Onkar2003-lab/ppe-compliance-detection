"""Render a video from still images, so the demo's latency can be measured on the video path.

The demo's headline speed figure has to come from the path a deployment actually runs:
decode a frame, detect, track, associate, gate on the zone, time the dwell, draw. Two of
those (decoding and tracking) do no meaningful work on a directory of unrelated stills, so
measuring there would understate the cost. There is no site video in this project (no labelled
clip exists, which is why S6.5's alert accuracy is scored on Pictor instead), so the sequence
is rendered from images we already hold.

This is a **timing fixture, not evidence about accuracy**. Consecutive frames are unrelated
photographs, so the tracker's matches are meaningless, but its work per frame is real, which
is the only thing being measured. Any accuracy claim comes from `src.demo_eval`, never here.

Usage::

    python scripts/make_demo_clip.py --images D:/Dissertation/harmonised/chv/images \
        --out D:/runs/X06-demo/demo-clip.mp4 --frames 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render stills into a clip for demo timing.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in SUFFIXES)
    if not paths:
        print(f"no images in {args.images}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    written = 0
    for path in paths[: args.frames]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        writer.write(cv2.resize(image, (args.width, args.height)))
        written += 1
    writer.release()
    print(f"{written} frames -> {args.out} ({args.width}x{args.height} @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
