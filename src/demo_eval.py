"""S6.5: evaluating the demo. Does it alert on the right people, inside the zone?

The build plan originally asked for "alert accuracy against a labelled test clip". No such
clip exists. There is no site video annotated with a zone and per-worker compliance, hand
labelling one costs days the calendar has not got, and inventing one would be fabricated
evidence. The evaluation is therefore split into the three things that *can* be measured
honestly, and this module is the third:

1. **Latency**: measured by the demo itself (``src.monitor`` writes ``demo-metrics.json``).
2. **Zone, dwell and debounce logic**: proven *exactly* by ``tests/test_zone.py`` and
   ``tests/test_dwell.py`` on synthetic geometry and scripted tracks. Pure logic with known
   answers is settled by test, not sampled; that is stronger than a small labelled clip.
3. **Alert accuracy**: this module. Pictor-PPE labels every worker's position *and* their
   compliance state, so once a zone is drawn over the frame the correct alert set follows by
   construction: a worker standing inside it without the required PPE should raise an alert,
   and anyone else should not. Real ground truth, no hand labelling.

**This is not a second measurement of detection accuracy.** That is the violation axis (F27),
and it is already reported. What is new here is the *zone gate*: whether restricting attention
to a marked area alerts on the right subset. The scoring machinery is deliberately imported
from :mod:`src.violation`: the same ground truth, the same one-to-one IoU 0.5 matching, the
same frozen confidence, so the two numbers are comparable rather than merely similar.

**Several zones are scored, not one.** A single hand-placed polygon invites the suspicion that
it was moved until the number improved, so the geometries are fixed in :data:`ZONES` before any
of them is run, defined without reference to the labels, and reported together, including the
whole frame, which is the no-zone control. Detections are computed once per image and reused
for every zone, so the comparison is exactly like-for-like as well as cheap.

**Alerting is scored per image, without the tracker.** Pictor's images are unrelated stills;
ByteTrack only returns detections it has confirmed across consecutive frames, so running it
here would withhold boxes and manufacture violations. Dwell and debounce are likewise
meaningless between two unrelated photographs; they are what the unit tests cover.

**Helmet leads.** Only 41 of 2,497 workers wear a vest (1.6 %, X01/S1.1b F13), so a vest
figure would say more about the base rate than the system. The required-PPE set is a flag, and
the default is helmet.

Usage::

    python -m src.demo_eval --weights D:/runs/X04-y8n-s0-sh17/weights/best.pt
    python -m src.demo_eval --weights ... --limit 50        # smoke check
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.associate import CLASS_NAMES, PERSON, Box, associate
from src.utils.logging import get_logger
from src.violation import (
    CONFIDENCE,
    DEFAULT_PICTOR,
    MATCH_IOU,
    Worker,
    load_ground_truth,
    match,
    predictions_for,
    to_normalised,
)
from src.zone import Zone, feet_point

logger = get_logger(__name__)

DEFAULT_OUT = Path("D:/runs/X06-demo-eval")

# Fixed before any of them was scored, and defined by frame geometry alone, never by where
# the workers happen to be. The whole frame is the control: it is the same pipeline with the
# zone gate open, so the difference between it and the rest is what the zone layer does.
ZONES: dict[str, Zone] = {
    "central-half": Zone(
        name="central-half", points=((0.25, 0.0), (0.75, 0.0), (0.75, 1.0), (0.25, 1.0))
    ),
    "lower-half": Zone(name="lower-half", points=((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0))),
    "left-half": Zone(name="left-half", points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))),
    "whole-frame": Zone(
        name="whole-frame", points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    ),
}


@dataclass
class AlertScore:
    """Alert-level confusion for one zone, where the positive class is *an alert*."""

    zone: str
    workers: int = 0
    workers_in_zone: int = 0
    should_alert: int = 0  # in the zone and missing required PPE
    tp: int = 0  # alerted, and they should have been
    fn: int = 0  # should have alerted and did not
    missed_people: int = 0  # of those misses, how many were never detected at all
    # False alerts are split by cause, because the two say different things about the system.
    # One is the pipeline judging a real worker wrongly (the PPE was there and was not bound)
    # and is squarely the demo's fault. The other is an alert on a detection that matches no
    # labelled worker at all, which is a person-detection artefact and would be a false alarm
    # on site too, but is not the zone or association layer misbehaving.
    fp_compliant: int = 0
    fp_unmatched: int = 0

    @property
    def fp(self) -> int:
        return self.fp_compliant + self.fp_unmatched

    @property
    def precision(self) -> float:
        """Operational precision: of every alert raised, how many a supervisor should act on."""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def precision_worker_level(self) -> float:
        """Precision counting only alerts on labelled workers.

        The violation axis scores workers, not detections; a prediction matching no labelled
        worker is invisible to it. This figure applies the same rule, so it is directly
        comparable with F27's helmet-violation precision, while :attr:`precision` above stays
        the stricter one a site would actually experience.
        """
        return self.tp / (self.tp + self.fp_compliant) if (self.tp + self.fp_compliant) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def summary(self) -> dict:
        return {
            "zone": self.zone,
            "workers": self.workers,
            "workers_in_zone": self.workers_in_zone,
            "should_alert": self.should_alert,
            "tp": self.tp,
            "fp": self.fp,
            "fp_on_a_compliant_worker": self.fp_compliant,
            "fp_on_an_unmatched_detection": self.fp_unmatched,
            "fn": self.fn,
            "misses_that_were_undetected_people": self.missed_people,
            "precision": round(self.precision, 4),
            "precision_worker_level": round(self.precision_worker_level, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class Evaluation:
    """Every zone's score for one set of weights."""

    weights: str
    confidence: float
    match_iou: float
    required: tuple[int, ...]
    images: int = 0
    scores: dict[str, AlertScore] = field(default_factory=dict)
    suppress_duplicates: bool = False


def alerts_for(
    predictions: list[Box], zone: Zone, required: tuple[int, ...]
) -> tuple[list[int], list[int]]:
    """Which detected people the demo would alert on inside this zone.

    Returns ``(person indices, alerting subset)``, both indices into ``predictions``, so the
    caller can match every detected person to a worker and still know which ones alerted.
    This runs the association rule over the *detections*, exactly as the demo does; the
    labelled worker boxes are never fed in, because a deployment is not handed them.
    """
    people = [i for i, box in enumerate(predictions) if box.cls == PERSON]
    association = associate(predictions)
    state_of = {person.index: person for person in association.people}

    alerting = []
    for index in people:
        person = state_of.get(index)
        if person is None or not zone.contains(feet_point(person.box)):
            continue
        if any(not person.wears(cls) for cls in required):
            alerting.append(index)
    return people, alerting


def score_image(
    predictions: list[Box],
    workers: list[Worker],
    zone: Zone,
    required: tuple[int, ...],
    score: AlertScore,
    match_iou: float = MATCH_IOU,
) -> None:
    """Fold one image's alerts into ``score``.

    A worker who should have been alerted on but was never detected counts as a miss, for the
    same reason the violation axis counts them: on a site, an undetected worker and a worker
    the system judged compliant produce the same silence.
    """
    people, alerting = alerts_for(predictions, zone, required)
    pairs = match([predictions[i] for i in people], workers, match_iou)  # worker -> person slot
    alerting_slots = {people.index(i) for i in alerting}

    score.workers += len(workers)
    matched_alerting = set()

    for w, worker in enumerate(workers):
        in_zone = zone.contains(feet_point(worker.box))
        should = in_zone and any(not worker.wearing(cls) for cls in required)
        score.workers_in_zone += int(in_zone)
        score.should_alert += int(should)

        detected = w in pairs
        alerted = detected and pairs[w] in alerting_slots
        if alerted:
            matched_alerting.add(pairs[w])

        if should and alerted:
            score.tp += 1
        elif should:
            score.fn += 1
            score.missed_people += int(not detected)
        elif alerted:
            score.fp_compliant += 1  # a real worker, wearing it, alerted on anyway

    # Alerts on detections that matched no labelled worker: a duplicated or spurious person
    # box, or somebody the labels do not carry. The site still gets an alert nobody should
    # answer, so it counts against precision, but it is a detection artefact, not the zone
    # or association layer, which is why it is counted separately.
    score.fp_unmatched += len(alerting_slots - matched_alerting)


def evaluate(
    weights: Path,
    pictor: Path,
    required: tuple[int, ...],
    confidence: float = CONFIDENCE,
    match_iou: float = MATCH_IOU,
    limit: int | None = None,
    suppress_duplicates: bool = False,
) -> Evaluation:
    """Score every zone in :data:`ZONES` for one set of weights.

    ``suppress_duplicates`` runs :func:`src.monitor.suppress_duplicate_people` over the
    detections before scoring, matching what the live demonstration does. It defaults to
    **off** because the S6.5 numbers were produced before the suppression existed (F37), and
    silently changing the default would move a recorded result without anybody deciding to.
    Turning it on is the sensitivity analysis: it should shift the "alert on an unmatched
    detection" column, which is the category duplicate person boxes create.
    """
    from ultralytics import YOLO

    from src.monitor import suppress_duplicate_people

    truth = load_ground_truth(pictor)
    images_dir = pictor / "Images"
    model = YOLO(str(weights))

    evaluation = Evaluation(
        weights=str(weights),
        confidence=confidence,
        match_iou=match_iou,
        required=required,
        scores={name: AlertScore(zone=name) for name in ZONES},
        suppress_duplicates=suppress_duplicates,
    )

    names = sorted(truth)[:limit] if limit else sorted(truth)
    for position, image in enumerate(names, start=1):
        path = next((p for p in images_dir.glob(f"{Path(image).stem}.*")), None)
        if path is None:
            logger.warning("image not found for %s, skipped", image)
            continue

        # Detected once, scored by every zone: the zones then differ only in geometry, which
        # is the whole point of comparing them.
        predictions, width, height = predictions_for(model, path, confidence)
        if suppress_duplicates:
            keep = suppress_duplicate_people(predictions)
            predictions = [predictions[index] for index in keep]
        workers = [
            Worker(box=to_normalised(box, width, height), helmet=helmet, vest=vest, split=split)
            for box, helmet, vest, split in truth[image]
        ]
        evaluation.images += 1
        for name, zone in ZONES.items():
            score_image(predictions, workers, zone, required, evaluation.scores[name], match_iou)

        if position % 100 == 0:
            logger.info("%d/%d images scored", position, len(names))

    return evaluation


def build_report(evaluation: Evaluation) -> str:
    """Markdown record of what the zone gate did (the ledger stays canonical for findings)."""
    required = "+".join(CLASS_NAMES[cls] for cls in evaluation.required)
    control = evaluation.scores["whole-frame"]
    lines = [
        "# X06 / S6.5: demo alert accuracy on Pictor-PPE, by zone",
        "",
        (
            f"Weights `{evaluation.weights}` · confidence {evaluation.confidence} · "
            f"person match IoU {evaluation.match_iou} · required PPE **{required}** · "
            f"{evaluation.images} images."
        ),
        "",
        (
            "An alert is correct when a worker standing inside the zone without the required "
            "PPE raises one. Ground truth comes from Pictor's per-worker compliance labels and "
            "positions, so the correct alert set is derived, not judged. Every zone below was "
            "fixed before scoring and defined by frame geometry alone; **whole-frame is the "
            "control**, the same pipeline with the zone gate open. Detections are computed once "
            "per image and shared across zones, so the rows differ only in geometry."
        ),
        "",
        "| zone | workers in zone | should alert | TP | FP | FN | precision (all alerts) | precision (labelled workers) | recall | F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ZONES:
        s = evaluation.scores[name]
        lines.append(
            f"| {name} | {s.workers_in_zone} | {s.should_alert} | {s.tp} | {s.fp} | {s.fn} | "
            f"{s.precision:.3f} | {s.precision_worker_level:.3f} | {s.recall:.3f} | {s.f1:.3f} |"
        )

    lines += [
        "",
        "### Where the errors come from",
        "",
        "| zone | false alerts on a compliant worker | false alerts on an unmatched detection | misses that were undetected people |",
        "|---|---|---|---|",
    ]
    for name in ZONES:
        s = evaluation.scores[name]
        missed = f"{s.missed_people} of {s.fn}" if s.fn else "-"
        lines.append(f"| {name} | {s.fp_compliant} | {s.fp_unmatched} | {missed} |")

    lines += [
        "",
        "## What this does and does not evidence",
        "",
        (
            "It evidences the **zone gate**: given the detections the violation axis already "
            "reports, restricting attention to a marked area alerts on the right subset of "
            "workers. It is **not** a second measurement of detection accuracy; the "
            f"whole-frame control's recall of {control.recall:.3f} is governed by whether the "
            "detector finds the person at all, which F27 already established as this "
            "pipeline's weak link."
        ),
        "",
        (
            "It does **not** evidence end-to-end accuracy on real site footage. No labelled "
            "site video exists here, and none is claimed; a Discussion limitation stated "
            "plainly, alongside the nano-only and under-supervised-person limits. Dwell and "
            "debounce are not exercised here either: Pictor's images are unrelated stills with "
            "no elapsed time between them, so that logic is evidenced by unit test instead."
        ),
        "",
        "_Generated by `python -m src.demo_eval`._",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="S6.5 demo alert accuracy, scored by zone.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pictor", type=Path, default=DEFAULT_PICTOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--match-iou", type=float, default=MATCH_IOU)
    parser.add_argument(
        "--required",
        nargs="+",
        default=["helmet"],
        help="required PPE (default helmet: only 1.6 %% of Pictor workers wear a vest)",
    )
    parser.add_argument("--limit", type=int, default=None, help="score only the first N images")
    parser.add_argument(
        "--suppress-duplicates",
        action="store_true",
        help="drop person boxes contained in a larger one, as the live demo does (F37)",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        logger.error("weights not found: %s", args.weights)
        return 1

    required = tuple(dict.fromkeys(_class_of(name) for name in args.required))
    evaluation = evaluate(
        args.weights,
        args.pictor,
        required,
        args.conf,
        args.match_iou,
        args.limit,
        args.suppress_duplicates,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "demo-alert-accuracy.md").write_text(build_report(evaluation), encoding="utf-8")
    (args.out / "demo-alert-accuracy.json").write_text(
        json.dumps(
            {
                "weights": evaluation.weights,
                "images": evaluation.images,
                "settings": {
                    "confidence": evaluation.confidence,
                    "match_iou": evaluation.match_iou,
                    "required_ppe": [CLASS_NAMES[c] for c in evaluation.required],
                    "association_threshold": "src.associate.THRESHOLD (SH17/CHV train only)",
                    "tracking": "off: Pictor images are unrelated stills",
                    "duplicate_suppression": "on" if evaluation.suppress_duplicates else "off",
                },
                "zones": {
                    name: {
                        "points": [list(p) for p in ZONES[name].points],
                        **score.summary(),
                    }
                    for name, score in evaluation.scores.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, score in evaluation.scores.items():
        logger.info(
            "%s: precision %.3f · recall %.3f · F1 %.3f (%d should alert)",
            name,
            score.precision,
            score.recall,
            score.f1,
            score.should_alert,
        )
    logger.info("report: %s", args.out / "demo-alert-accuracy.md")
    return 0


def _class_of(name: str) -> int:
    """Class id for a PPE name, rejecting anything outside the trained space."""
    lookup = {value: key for key, value in CLASS_NAMES.items()}
    key = name.strip().lower()
    if key not in lookup or lookup[key] == PERSON:
        raise ValueError(f"{name!r} is not trained PPE")
    return lookup[key]


if __name__ == "__main__":
    raise SystemExit(main())
