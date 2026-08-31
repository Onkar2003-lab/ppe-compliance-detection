"""X07-roc-metrics: ROC / PR / calibration data for the per-worker helmet-compliance decision.

Post-processing only. Nothing here retrains, re-tunes, or touches mAP: it re-scores the
**frozen S5.4 violation-axis decision** across the detection-confidence threshold tau, so the
write-up can show an ROC / PR curve and a calibration plot for a decision where those curves
are actually defined.

**Why the curves are valid here and not on raw detection.** Object detection has no true
negatives (an unbounded number of boxes are correctly not emitted), so ROC on detections is
meaningless. The per-worker compliance decision does have them: every labelled Pictor worker
is one sample, the positive class is *helmet violation* (~35 % of 2,497), and a compliant
worker the system correctly stayed silent about is a true negative. Helmet only: vest is
saturated (41 vest-wearers, 1.6 %) and its curve would describe the label skew, not the model.

**What moves and what does not.** Only tau moves. The association rule
(:func:`src.associate.associate`, containment 0.80) and the person match (IoU 0.50) are the
frozen S5.4 settings, imported rather than reimplemented; the eval set is the same whole
Pictor release (774 images / 2,497 workers); the strict scoring rule of S5.4 decision 1 is
kept: an undetected violator is a false negative, an undetected compliant worker is a true
negative. The tau=0.25 column of the sweep therefore has to reproduce the frozen X05 helmet
counts exactly, and :func:`verify` asserts it against the cached X05 JSON.

Inference runs once per model at the sweep floor and is cached, so every later invocation is
pure arithmetic. Filtering a cached conf>=0.01 prediction set at tau is exactly equivalent to
re-running the detector at conf=tau: Ultralytics applies the confidence filter before NMS, and
NMS is greedy from the highest score down, so a box below tau can never suppress one above it.

Usage::

    python -m scripts.roc_sweep --limit 20    # smoke check
    python -m scripts.roc_sweep               # full sweep, 9 SH17 runs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src.associate import HELMET, PERSON, Box, associate
from src.utils.logging import get_logger
from src.violation import (
    MATCH_IOU,
    Confusion,
    Worker,
    load_ground_truth,
    match,
    to_normalised,
)

logger = get_logger(__name__)

DEFAULT_PICTOR = Path("D:/Dissertation/pictor-ppe_dataset")
DEFAULT_RUNS = Path("D:/runs")
DEFAULT_CACHE = Path("D:/runs/X07-roc/cache")
DEFAULT_OUT = Path("D:/runs/X07-roc")  # pass --out to write the CSVs somewhere else

# SH17-trained only: the transfer direction S5.4 reports. Seeds 0-2 all included; inference is
# well under a minute a run, so the extra seeds cost nothing and let the curve carry a spread.
RUNS = [f"X04-{m}-s{s}-sh17" for s in (0, 1, 2) for m in ("y8n", "y11n", "y26n")]
MODEL_OF = {"y8n": "YOLOv8n", "y11n": "YOLO11n", "y26n": "YOLO26n"}

SWEEP_FLOOR = 0.01  # inference confidence; also the first tau
TAUS = [round(0.01 * i, 2) for i in range(1, 100)]  # 0.01 -> 0.99
SCORES_TAU = 0.25  # the frozen S5.4 operating point, for worker_scores.csv


# ------------------------------------------------------------------------------ inference


def cached_predictions(weights: Path, images_dir: Path, names: list[str], cache: Path) -> dict:
    """Every detection at conf >= :data:`SWEEP_FLOOR` for one run, cached to disk.

    Returns the per-image ``(M, 6)`` arrays of ``(cls, xc, yc, w, h, conf)`` in normalised
    coordinates, plus each image's pixel size, needed to normalise the ground truth the same
    way :mod:`src.violation` does, from the detector's own view of the image.
    """
    if cache.is_file():
        blob = np.load(cache, allow_pickle=False)
        order = [str(n) for n in blob["names"]]
        edges = np.concatenate([[0], np.cumsum(blob["counts"])])
        flat, sizes = blob["boxes"], blob["sizes"]
        logger.info("cache hit: %s (%d images)", cache.name, len(order))
        return {
            "boxes": {n: flat[edges[i] : edges[i + 1]] for i, n in enumerate(order)},
            "sizes": {n: tuple(sizes[i]) for i, n in enumerate(order)},
            "reran": False,
        }

    from ultralytics import YOLO

    model = YOLO(str(weights))
    run_id = weights.parent.parent.name
    boxes, sizes, counts = {}, {}, []
    started = time.time()
    for i, name in enumerate(names, start=1):
        path = images_dir / name
        if not path.is_file():
            logger.warning("missing image, skipped: %s", path)
            boxes[name], sizes[name] = np.zeros((0, 6), np.float32), (0, 0)
            counts.append(0)
            continue
        result = model.predict(source=str(path), conf=SWEEP_FLOOR, verbose=False, device=None)[0]
        height, width = result.orig_shape
        xywhn = result.boxes.xywhn.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().reshape(-1, 1)
        conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
        boxes[name] = np.hstack([cls, xywhn, conf]).astype(np.float32)
        sizes[name] = (width, height)
        counts.append(len(boxes[name]))
        if i % 200 == 0:
            logger.info("  %s: %d/%d images", run_id, i, len(names))

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        names=np.array(names),
        counts=np.array(counts),
        sizes=np.array([sizes[n] for n in names]),
        boxes=np.vstack([boxes[n] for n in names]) if names else np.zeros((0, 6), np.float32),
    )
    logger.info("inference %s: %.1f s -> %s", run_id, time.time() - started, cache.name)
    return {"boxes": boxes, "sizes": sizes, "reran": True}


# ---------------------------------------------------------------------- per-worker scoring


def workers_for(rows: list, size: tuple[int, int]) -> list[Worker]:
    width, height = size
    return [
        Worker(box=to_normalised(pixels, width, height), helmet=h, vest=v, split=s)
        for pixels, h, v, s in rows
    ]


def score_image_rows(raw: np.ndarray, workers: list[Worker], tau: float) -> list[dict]:
    """One row per labelled worker: the helmet decision this image's detections produce.

    Mirrors :func:`src.violation.score_image` exactly: same association call, same greedy
    IoU person match, same strict treatment of an undetected worker, but emits the decision
    per worker instead of folding it straight into a confusion table, so one pass produces
    both the sweep tallies and the calibration rows.
    """
    kept = raw[raw[:, 5] >= tau] if len(raw) else raw
    boxes = [
        Box(cls=int(r[0]), xc=float(r[1]), yc=float(r[2]), w=float(r[3]), h=float(r[4]))
        for r in kept
    ]
    confidences = kept[:, 5] if len(kept) else np.zeros(0, np.float32)

    people = [i for i, b in enumerate(boxes) if b.cls == PERSON]
    state_of = {p.index: p for p in associate(boxes).people}
    pairs = match([boxes[i] for i in people], workers, MATCH_IOU)

    rows = []
    for w, worker in enumerate(workers):
        detected = w in pairs
        person = state_of.get(people[pairs[w]]) if detected else None
        helmet_index = person.bound.get(HELMET) if person is not None else None
        rows.append(
            {
                "worker_idx": w,
                "true_label": int(not worker.helmet),
                "person_detected": int(detected),
                "helmet_conf_bound": (
                    float(confidences[helmet_index]) if helmet_index is not None else None
                ),
                # No detection means no alert: the pipeline stayed silent about this worker.
                "pred_violation": int(person is not None and not person.wears(HELMET)),
            }
        )
    return rows


def tally(rows: list[dict]) -> Confusion:
    """Fold per-worker helmet decisions into the confusion table (positive = violation)."""
    counts = Confusion()
    for row in rows:
        violation, flagged = bool(row["true_label"]), bool(row["pred_violation"])
        if violation and flagged:
            counts.tp += 1
        elif violation:
            counts.fn += 1
            if not row["person_detected"]:
                counts.missed_people += 1
        elif flagged:
            counts.fp += 1
        else:
            counts.tn += 1
    return counts


# ------------------------------------------------------------------------------------ main


def verify(run_id: str, counts: Confusion, runs_dir: Path, partial: bool) -> str:
    """Assert the tau=0.25 column reproduces the frozen X05 helmet counts.

    Only meaningful over the whole eval set; a ``--limit`` smoke run scores a subset of the
    774 images and cannot match, so it reports rather than asserts.
    """
    cached = runs_dir / "X05-violation" / f"{run_id}-violation.json"
    if partial:
        return f"{run_id}: smoke subset, equivalence check skipped"
    if not cached.is_file():
        return f"{run_id}: no cached X05 JSON to check against"
    frozen = json.loads(cached.read_text(encoding="utf-8"))["axes"]["helmet"]["counts"]
    ours = {"tp": counts.tp, "fp": counts.fp, "fn": counts.fn, "tn": counts.tn}
    if ours != frozen:
        raise AssertionError(f"{run_id}: tau=0.25 gives {ours}, frozen X05 says {frozen}")
    return f"{run_id}: tau=0.25 reproduces frozen X05 helmet counts {ours}"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d rows)", path, len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ROC/PR/calibration sweep for the helmet decision."
    )
    parser.add_argument("--pictor", type=Path, default=DEFAULT_PICTOR)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="first N images (smoke check)")
    args = parser.parse_args()

    truth = load_ground_truth(args.pictor)
    names = sorted(truth)[: args.limit] if args.limit else sorted(truth)
    images_dir = args.pictor / "Images"
    args.out.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict] = []
    checks: list[str] = []
    provenance: list[dict] = []
    scores_rows: list[dict] | None = None

    for run_id in RUNS:
        weights = args.runs / run_id / "weights" / "best.pt"
        if not weights.is_file():
            logger.warning("no weights for %s, skipped", run_id)
            continue
        family, seed = run_id.split("-")[1], int(run_id.split("-")[2][1:])
        cached = cached_predictions(weights, images_dir, names, args.cache / f"{run_id}.npz")
        provenance.append(
            {
                "run_id": run_id,
                "weights": str(weights),
                "config": f"configs/X04-{family}-s{seed}-sh17.yaml",
                "predictions": "reran" if cached["reran"] else "cached",
            }
        )

        workers_of = {n: workers_for(truth[n], cached["sizes"][n]) for n in names}
        started = time.time()
        for tau in TAUS:
            rows = [
                (n, r)
                for n in names
                for r in score_image_rows(cached["boxes"][n], workers_of[n], tau)
            ]
            counts = tally([r for _, r in rows])
            sweep_rows.append(
                {
                    "model": MODEL_OF[family],
                    "seed": seed,
                    "train_set": "sh17",
                    "tau": f"{tau:.2f}",
                    "TP": counts.tp,
                    "FP": counts.fp,
                    "FN": counts.fn,
                    "TN": counts.tn,
                }
            )
            if abs(tau - SCORES_TAU) < 1e-9:
                checks.append(verify(run_id, counts, args.runs, partial=bool(args.limit)))
                if run_id == "X04-y8n-s0-sh17":
                    scores_rows = [dict(image_id=n, **r) for n, r in rows]
        logger.info("%s: swept %d thresholds in %.1f s", run_id, len(TAUS), time.time() - started)

    write_csv(
        args.out / "roc_sweep.csv",
        ["model", "seed", "train_set", "tau", "TP", "FP", "FN", "TN"],
        sweep_rows,
    )

    if scores_rows:
        for row in scores_rows:
            bound = row["helmet_conf_bound"]
            row["helmet_conf_bound"] = "NA" if bound is None else f"{bound:.6f}"
        write_csv(
            args.out / "worker_scores.csv",
            [
                "image_id",
                "worker_idx",
                "true_label",
                "person_detected",
                "helmet_conf_bound",
                "pred_violation",
            ],
            scores_rows,
        )

    for line in checks:
        logger.info("check: %s", line)
    (args.out / "_provenance.json").write_text(
        json.dumps({"runs": provenance, "checks": checks}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
