"""The violation-recall axis: zero-shot compliance scoring on Pictor-PPE (S5).

This is the axis that asks the question the whole project exists for: *does the system
catch a worker who is not wearing their PPE?* It is the only one of the four axes
that no earlier stage touched.

**What makes the number meaningful is that nothing here was trained.** Pictor-PPE is
evaluation-only under a binding supervisor condition (2026-07-26), it shares zero images
with SH17 or CHV (0 dHash pairs, X01/S1.1b F11), and `no-helmet` is not a class any model
was ever shown. The detector emits `{person, helmet, vest}`; the calibrated association
rule (:mod:`src.associate`, threshold fixed on SH17/CHV train splits and never on Pictor)
derives each person's compliance state; Pictor's per-worker `W`/`WH`/`WV`/`WHV` labels then
score that state. So the axis measures the deployed *pipeline*, not a classifier head, and
it shares its logic with the O6 demo, which is why the rule lives in one module imported by
both.

Four scoring decisions carry the number, and each is a choice that could reasonably have
gone the other way. They are stated here rather than buried in the code:

1. **A worker the detector never finds counts as a violation missed.** An undetected person
   raises no alert, and on a real site that is indistinguishable from a person the system
   decided was compliant. Scoring only the workers we happened to detect would quietly
   discard the pipeline's worst failures and inflate recall. Unmatched *compliant* workers
   are counted as correct silence, not as false alarms.
2. **The whole crowd-sourced release is scored (774 images, ~2,497 workers), not its test
   split.** Pictor's own train/valid/test partition exists to serve models trained on
   Pictor; ours never was, so every image is equally held out and using all of them buys
   support the axis badly needs. The split each worker came from is recorded anyway, so the
   test-split-only figure remains recoverable without re-running.
3. **Predicted people are matched to labelled workers at IoU >= 0.5**, greedily and
   one-to-one. Person-to-person is a like-for-like comparison, so ordinary IoU is right
   here, unlike the PPE binding inside the association rule, where a helmet against a
   full-body box forced containment instead (F17).
4. **The detection confidence threshold is frozen at the Ultralytics default (0.25).** The
   improvement protocol's L1 lever (threshold tuning for recall) is deliberately deferred
   (user, 2026-08-06), so tuning it here would quietly run a lever we agreed not to pull.
   ``--sweep`` reports sensitivity across thresholds for the write-up without changing the
   headline.

**Helmet leads; vest is reported but never alone.** The available Pictor release holds only
41 vest-wearing workers (1.6 %), so a model that never predicts a vest scores near-perfect
vest-violation recall. Every vest figure is therefore emitted beside the always-flag trivial
baseline and its support count, in the same row, by construction; the report cannot be
generated without them (user decision A, X01/S1.1b F13).

Usage::

    python -m src.violation --runs D:/runs --limit 20     # smoke check on a few images
    python -m src.violation --runs D:/runs                # score all 18 runs
    python -m src.violation --runs D:/runs --sweep        # + confidence sensitivity
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.associate import HELMET, PERSON, VEST, Box, associate
from src.audit_pictor import COMPLIANCE_DECODING, load_approach
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PICTOR = Path("D:/Dissertation/pictor-ppe_dataset")
DEFAULT_RUNS = Path("D:/runs")
DEFAULT_OUT = Path("D:/runs/X05-violation")

# Approach 2 is the compliance-labelled view: one box per worker, carrying the W/WH/WV/WHV
# class. Approaches 1 and 3 label PPE as separate objects and are not the ground truth here.
COMPLIANCE_APPROACH = "02"

CONFIDENCE = 0.25  # frozen: see decision 4 in the module docstring
MATCH_IOU = 0.50  # predicted person <-> labelled worker
SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

# The two classes the axis scores. Helmet is the headline; vest rides along with its
# baseline attached (see the module docstring).
SCORED = {"helmet": HELMET, "vest": VEST}


# --------------------------------------------------------------------------- ground truth


@dataclass(frozen=True)
class Worker:
    """One labelled worker: where they are, and what they are actually wearing."""

    box: Box  # normalised, cls=PERSON so it can share the geometry helpers
    helmet: bool
    vest: bool
    split: str  # Pictor's own partition, recorded so a test-only figure stays recoverable

    def wearing(self, cls: int) -> bool:
        return self.helmet if cls == HELMET else self.vest


def to_normalised(pixels: tuple[int, int, int, int, int], width: int, height: int) -> Box:
    """Convert one Pictor ``(x1, y1, x2, y2, cls)`` pixel box to a normalised :class:`Box`.

    The class is forced to ``PERSON``: a Pictor row's class is the worker's *compliance
    state*, not an object class, and letting it through as a class id would silently feed
    the association rule a box it would read as a helmet.
    """
    x1, y1, x2, y2, _cls = pixels
    return Box(
        cls=PERSON,
        xc=((x1 + x2) / 2) / width,
        yc=((y1 + y2) / 2) / height,
        w=abs(x2 - x1) / width,
        h=abs(y2 - y1) / height,
    )


def load_ground_truth(root: Path) -> dict[str, list[tuple[tuple[int, ...], bool, bool, str]]]:
    """Read Pictor's compliance labels as raw pixel boxes plus decoded PPE state.

    Kept in pixels at this stage because normalising needs each image's dimensions, which
    are read once from the detector's own view of the image rather than by opening every
    file a second time.
    """
    labels = load_approach(root, COMPLIANCE_APPROACH)
    truth: dict[str, list[tuple[tuple[int, ...], bool, bool, str]]] = {}
    for image, boxes in labels.by_image.items():
        split = labels.split_of[image]
        rows = []
        for box in boxes:
            decoded = COMPLIANCE_DECODING.get(box[4])
            if decoded is None:
                logger.warning("%s: unknown compliance class %s, skipped", image, box[4])
                continue
            rows.append((box, decoded["helmet"], decoded["vest"], split))
        truth[image] = rows
    logger.info(
        "ground truth: %d images, %d workers (approach %s)",
        len(truth),
        sum(len(v) for v in truth.values()),
        COMPLIANCE_APPROACH,
    )
    return truth


# ------------------------------------------------------------------------------- matching


def iou(a: Box, b: Box) -> float:
    """Symmetric IoU: the right measure for person-against-person (decision 3)."""
    ax1, ay1, ax2, ay2 = a.corners
    bx1, by1, bx2, by2 = b.corners
    width = min(ax2, bx2) - max(ax1, bx1)
    height = min(ay2, by2) - max(ay1, by1)
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    union = a.area + b.area - overlap
    return overlap / union if union > 0 else 0.0


def match(
    predicted: list[Box], workers: list[Worker], threshold: float = MATCH_IOU
) -> dict[int, int]:
    """Greedily pair predicted people to labelled workers, best overlap first.

    One-to-one and strongest-first, so a single confident detection cannot be credited with
    covering two workers standing close together.

    Returns:
        ``{worker index: predicted index}`` for pairs above ``threshold``. Workers absent
        from the mapping were never detected, which decision 1 scores as a missed violation.
    """
    candidates = [
        (iou(predicted[p], worker.box), w, p)
        for w, worker in enumerate(workers)
        for p in range(len(predicted))
        if iou(predicted[p], worker.box) >= threshold
    ]
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    pairs: dict[int, int] = {}
    used: set[int] = set()
    for _score, w, p in candidates:
        if w in pairs or p in used:
            continue
        pairs[w] = p
        used.add(p)
    return pairs


# -------------------------------------------------------------------------------- scoring


@dataclass
class Confusion:
    """Counts for one PPE class, where the positive class is *the violation*."""

    tp: int = 0  # not wearing it, and we said so
    fp: int = 0  # wearing it, but we raised an alert
    fn: int = 0  # not wearing it, and we stayed silent (incl. never detecting them)
    tn: int = 0  # wearing it, and we stayed silent
    missed_people: int = 0  # violations inside `fn` that came from an undetected worker

    @property
    def support(self) -> int:
        """Labelled workers actually in violation: the denominator of recall."""
        return self.tp + self.fn

    @property
    def recall(self) -> float:
        return self.tp / self.support if self.support else 0.0

    @property
    def precision(self) -> float:
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def base_rate(self) -> float:
        """Share of workers in violation, and the precision of always flagging."""
        total = self.tp + self.fp + self.fn + self.tn
        return self.support / total if total else 0.0

    def summary(self) -> dict:
        """Metrics *with* the trivial baseline attached, so it cannot be reported without it.

        The always-flag baseline is the honest comparator for a class as skewed as vest:
        it scores recall 1.0 by construction, and its precision is just the base rate. A
        model that cannot beat it has learned nothing about that class.
        """
        return {
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "support_violations": self.support,
            "base_rate": round(self.base_rate, 4),
            "trivial_always_flag": {
                "recall": 1.0,
                "precision": round(self.base_rate, 4),
                "note": "flags every worker; beating this is the minimum bar for the class",
            },
            "counts": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "violations_missed_because_person_undetected": self.missed_people,
        }


@dataclass
class Score:
    """One model's violation-axis result."""

    run_id: str
    weights: str
    confidence: float
    match_iou: float
    images: int = 0
    workers: int = 0
    workers_detected: int = 0
    classes: dict[str, Confusion] = field(default_factory=lambda: {k: Confusion() for k in SCORED})
    by_split: Counter = field(default_factory=Counter)

    def report(self) -> dict:
        detection_rate = self.workers_detected / self.workers if self.workers else 0.0
        return {
            "run_id": self.run_id,
            "weights": self.weights,
            "settings": {
                "confidence": self.confidence,
                "match_iou": self.match_iou,
                "association_threshold": "src.associate.THRESHOLD (calibrated on SH17/CHV train)",
                "scored": "whole crowd-sourced release, all splits",
            },
            "images": self.images,
            "workers": self.workers,
            "person_detection_rate": round(detection_rate, 4),
            "headline_helmet_violation_recall": round(self.classes["helmet"].recall, 4),
            "axes": {name: c.summary() for name, c in self.classes.items()},
            "workers_by_pictor_split": dict(self.by_split),
        }


def score_image(
    predictions: list[Box],
    workers: list[Worker],
    score: Score,
    match_iou: float = MATCH_IOU,
) -> None:
    """Fold one image's detections into ``score``.

    The association rule runs over the *detections* exactly as the demo would run it: the
    labelled worker boxes are never fed in, because at deployment nobody hands the system
    the ground-truth person boxes.
    """
    people = [i for i, b in enumerate(predictions) if b.cls == PERSON]
    association = associate(predictions)
    state_of = {p.index: p for p in association.people}

    pairs = match([predictions[i] for i in people], workers, match_iou)
    score.images += 1
    score.workers += len(workers)
    score.workers_detected += len(pairs)

    for w, worker in enumerate(workers):
        score.by_split[worker.split] += 1
        detected = w in pairs
        person = state_of.get(people[pairs[w]]) if detected else None

        for name, cls in SCORED.items():
            confusion = score.classes[name]
            in_violation = not worker.wearing(cls)
            # No detection means no alert: the pipeline stayed silent about this worker.
            flagged = bool(person is not None and not person.wears(cls))

            if in_violation and flagged:
                confusion.tp += 1
            elif in_violation:
                confusion.fn += 1
                if not detected:
                    confusion.missed_people += 1
            elif flagged:
                confusion.fp += 1
            else:
                confusion.tn += 1


# ------------------------------------------------------------------------------ inference


def predictions_for(model, image_path: Path, confidence: float) -> tuple[list[Box], int, int]:
    """Detections for one image as normalised boxes, plus the image's pixel size."""
    result = model.predict(source=str(image_path), conf=confidence, verbose=False, device=None)[0]
    height, width = result.orig_shape
    boxes = [
        Box(cls=int(c), xc=float(x), yc=float(y), w=float(bw), h=float(bh))
        for (x, y, bw, bh), c in zip(result.boxes.xywhn.tolist(), result.boxes.cls.tolist())
    ]
    return boxes, width, height


def score_run(
    weights: Path,
    truth: dict,
    images_dir: Path,
    confidence: float = CONFIDENCE,
    match_iou: float = MATCH_IOU,
    limit: int | None = None,
) -> Score:
    """Score one trained model on the violation axis."""
    from ultralytics import YOLO

    run_id = weights.parent.parent.name
    model = YOLO(str(weights))
    score = Score(run_id=run_id, weights=str(weights), confidence=confidence, match_iou=match_iou)

    names = sorted(truth)[:limit] if limit else sorted(truth)
    for n, image in enumerate(names, start=1):
        path = images_dir / image
        if not path.is_file():
            logger.warning("missing image, skipped: %s", path)
            continue
        boxes, width, height = predictions_for(model, path, confidence)
        workers = [
            Worker(box=to_normalised(pixels, width, height), helmet=h, vest=v, split=s)
            for pixels, h, v, s in truth[image]
        ]
        score_image(boxes, workers, score, match_iou)
        if n % 200 == 0:
            logger.info("  %s: %d/%d images", run_id, n, len(names))

    helmet = score.classes["helmet"]
    logger.info(
        "%s: helmet-violation recall %.4f (support %d) · person detection %.3f",
        run_id,
        helmet.recall,
        helmet.support,
        score.workers_detected / score.workers if score.workers else 0.0,
    )
    return score


# -------------------------------------------------------------------------------- reports


def find_runs(runs_dir: Path, pattern: str = "X04-*") -> list[Path]:
    """Every graded run's best checkpoint, in run-ID order."""
    found = sorted(runs_dir.glob(f"{pattern}/weights/best.pt"))
    if not found:
        raise FileNotFoundError(f"no weights matching {pattern}/weights/best.pt under {runs_dir}")
    return found


def build_report(scores: list[Score], out: Path) -> Path:
    """Write the per-run results plus a table ordered by the headline metric."""
    out.mkdir(parents=True, exist_ok=True)
    reports = [s.report() for s in scores]
    for report in reports:
        (out / f"{report['run_id']}-violation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    ranked = sorted(reports, key=lambda r: -r["headline_helmet_violation_recall"])
    lines = [
        "# X05: violation-recall axis (zero-shot, Pictor-PPE)",
        "",
        "Per-run numbers only. **Nothing here is a result until S5 aggregates it across seeds",
        "with a 95 % BCa CI and the named test**; no 'X beats Y' from this table.",
        "",
        "Helmet is the headline. Vest is shown beside its always-flag trivial baseline and its",
        "support count, because only 1.6 % of the available workers wear one.",
        "",
        "| run | helmet recall | helmet prec | helmet F1 | vest recall | vest trivial prec | vest support | person det. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked:
        h, v = r["axes"]["helmet"], r["axes"]["vest"]
        lines.append(
            f"| {r['run_id']} | {h['recall']:.4f} | {h['precision']:.4f} | {h['f1']:.4f} "
            f"| {v['recall']:.4f} | {v['trivial_always_flag']['precision']:.4f} "
            f"| {v['support_violations']} | {r['person_detection_rate']:.3f} |"
        )

    first = reports[0]
    settings = (
        f"Settings: confidence {first['settings']['confidence']}, "
        f"person match IoU {first['settings']['match_iou']}, "
        "association threshold calibrated on SH17/CHV train splits only."
    )
    scored_on = (
        f"Scored on {first['images']} images / {first['workers']} labelled workers "
        "(whole crowd-sourced release; Pictor was never trained on)."
    )
    lines += [
        "",
        settings,
        scored_on,
        "",
        "An undetected worker counts as a violation missed, not as an absent sample; see",
        "`violations_missed_because_person_undetected` in each run's JSON.",
    ]
    path = out / "X05-violation-summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("report: %s", path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the violation-recall axis on Pictor.")
    parser.add_argument("--pictor", type=Path, default=DEFAULT_PICTOR)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pattern", default="X04-*", help="which run directories to score")
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--match-iou", type=float, default=MATCH_IOU)
    parser.add_argument("--limit", type=int, default=None, help="first N images (smoke check)")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also report confidence sensitivity (reporting only; does not move the headline)",
    )
    args = parser.parse_args()

    truth = load_ground_truth(args.pictor)
    images_dir = args.pictor / "Images"
    weights = find_runs(args.runs, args.pattern)
    logger.info("scoring %d runs at confidence %.2f", len(weights), args.conf)

    scores = [
        score_run(w, truth, images_dir, args.conf, args.match_iou, args.limit) for w in weights
    ]
    build_report(scores, args.out)

    if args.sweep:
        sensitivity = {}
        for conf in SWEEP:
            s = score_run(weights[0], truth, images_dir, conf, args.match_iou, args.limit)
            sensitivity[f"{conf:.2f}"] = s.classes["helmet"].summary()
            logger.info("sweep conf %.2f -> helmet recall %.4f", conf, s.classes["helmet"].recall)
        (args.out / "X05-confidence-sensitivity.json").write_text(
            json.dumps({"run_id": scores[0].run_id, "sweep": sensitivity}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
