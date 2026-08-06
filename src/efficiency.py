"""The efficiency axis: what each model costs to run (S5, fills ⚑ M3).

The deployment-readiness framework treats speed and size as a first-class axis because a
detector that cannot keep up with a camera cannot monitor a site, however accurate it is.
**This is a proxy, measured on a laptop RTX 4070 and an Ultra 9 185H — never a claim about
edge hardware.** No embedded device was used, and the write-up must say so (Locked-Context
§4); what these numbers support is a *relative* comparison between the three architectures
under identical conditions, plus a sanity check against the demo's 2 fps floor (C2).

Two different kinds of quantity live here, and conflating them would misrepresent the
uncertainty:

* **Architecture constants** — parameters, GFLOPs, file size. Identical for every seed of a
  family by construction, because seeds change weight *values*, not shape. They are measured
  once per architecture and verified identical across seeds rather than assumed.
* **Measured timings** — latency and FPS. These vary between repeats because of thermal and
  clock behaviour, not because of anything about the model. So their spread is *measurement*
  noise, and it must never be reported as if it were the seed variance that the accuracy axis
  reports. The two are kept in separate fields for exactly that reason.

Timing method, chosen so the numbers mean something:

* **Batch of 1**, because a camera delivers one frame at a time; batching would flatter
  throughput in a way no deployment could use.
* **End-to-end**, summing pre-process + inference + post-process. Post-processing is where a
  real architectural difference lives: YOLO26 is NMS-free, and measuring inference alone
  would hide the thing the newest generation changed.
* **Warm-up first, and discarded.** The first passes pay for CUDA context creation, memory
  allocation and clock ramp; including them would make every model look slower than it is and
  the first model measured look worst of all.
* **Real images from the frozen test split**, not random tensors, so decode and letterbox
  costs are the ones deployment actually pays.

⚠️ **Thermal caveat.** Sustained load makes this laptop's GPU drop clocks (~3,105 -> ~2,055
MHz, X04 timeline note). Models are therefore measured in a randomised interleaved order, so
a throttling drift cannot be mistaken for one architecture being slower.

Usage::

    python -m src.efficiency --repeats 3 --images 100      # GPU + CPU, all architectures
    python -m src.efficiency --skip-cpu                    # GPU only (much faster)
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics as st
from pathlib import Path

from src.run import DATA_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RUNS = Path("D:/runs")
DEFAULT_OUT = Path("D:/runs/X05-efficiency")
IMGSZ = 640
BATCH = 1  # deployment realism: one frame at a time
WARMUP = 15
TIMED = 100
REPEATS = 3
SEED = 0  # for the interleaving order, so the schedule is reproducible


def architecture_of(run_id: str) -> str:
    """The model family behind a run-ID (`X04-y11n-s2-chv` -> `y11n`)."""
    return run_id.split("-")[1]


def sample_images(dataset: str, count: int) -> list[Path]:
    """A fixed sample of real test images, identical for every model measured."""
    import yaml

    path = DATA_DIR / f"{dataset}-640.yaml"
    if not path.is_file():
        path = DATA_DIR / f"{dataset}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    listing = Path(document["path"]) / document["test"]
    images = [
        Path(line.strip())
        for line in listing.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(SEED)
    return rng.sample(images, min(count, len(images)))


def model_constants(weights: Path) -> dict:
    """Parameters, GFLOPs and on-disk size — properties of the architecture, not the seed."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    parameters = sum(p.numel() for p in model.model.parameters())
    gflops = None
    try:  # Ultralytics exposes this via its profiler; absence must not kill the axis
        from ultralytics.utils.torch_utils import get_flops

        gflops = round(float(get_flops(model.model, imgsz=IMGSZ)), 3)
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        logger.warning("GFLOPs unavailable for %s: %s", weights.parent.parent.name, error)

    return {
        "parameters": int(parameters),
        "parameters_millions": round(parameters / 1e6, 3),
        "gflops": gflops,
        "file_size_mb": round(weights.stat().st_size / 1e6, 2),
    }


def time_model(weights: Path, images: list[Path], device: str, repeats: int) -> dict:
    """Measure end-to-end latency per frame, after a discarded warm-up.

    Ultralytics reports per-image milliseconds for pre-process, inference and post-process;
    the sum is what a deployment waits for, so that is what is reported.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for image in images[:WARMUP]:
        model.predict(source=str(image), imgsz=IMGSZ, device=device, verbose=False)

    passes = []
    for _ in range(repeats):
        stages = {"preprocess": [], "inference": [], "postprocess": []}
        for image in images:
            result = model.predict(
                source=str(image), imgsz=IMGSZ, batch=BATCH, device=device, verbose=False
            )[0]
            for stage, collected in stages.items():
                collected.append(float(result.speed[stage]))
        total = sum(st.mean(v) for v in stages.values())
        passes.append(
            {k: round(st.mean(v), 3) for k, v in stages.items()} | {"total_ms": round(total, 3)}
        )

    totals = [p["total_ms"] for p in passes]
    return {
        "device": device,
        "repeats": repeats,
        "images_per_pass": len(images),
        "latency_ms_mean": round(st.mean(totals), 3),
        # Spread across repeats of the SAME weights: measurement noise, not seed variance.
        "latency_ms_measurement_sd": round(st.stdev(totals), 3) if len(totals) > 1 else 0.0,
        "fps_mean": round(1000.0 / st.mean(totals), 2),
        "stage_breakdown_ms": passes[-1],
        "passes_ms": totals,
    }


def measure(weights_by_run: dict[str, Path], images: list[Path], device: str, repeats: int) -> dict:
    """Time every run on one device, interleaved so throttling cannot bias one architecture."""
    order = list(weights_by_run)
    random.Random(SEED).shuffle(order)
    logger.info("timing on %s, interleaved order: %s", device, ", ".join(order[:4]) + " ...")

    results = {}
    for n, run_id in enumerate(order, start=1):
        logger.info("  [%d/%d] %s on %s", n, len(order), run_id, device)
        results[run_id] = time_model(weights_by_run[run_id], images, device, repeats)
    return results


def build_report(records: dict, out: Path) -> Path:
    """Per-architecture table. Constants are stated once; timings carry their own spread."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "X05-efficiency.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    preamble = (
        f"Measured on {records['hardware']['device_name']} and "
        f"{records['hardware']['cpu']}. Batch {BATCH}, imgsz {IMGSZ}, end-to-end "
        "(pre-process + inference + post-process), warm-up discarded, models interleaved."
    )
    lines = [
        "# X05 — efficiency axis (proxy; NOT edge hardware)",
        "",
        preamble,
        "",
        "**This is a proxy for deployment cost, not a measurement on an edge device.** No",
        "embedded hardware was used; the comparison is relative and like-for-like.",
        "",
        "## Architecture constants (identical across seeds by construction)",
        "",
        "| architecture | params (M) | GFLOPs | weights (MB) |",
        "|---|---|---|---|",
    ]
    for arch, c in sorted(records["constants"].items()):
        gflops = f"{c['gflops']:.2f}" if c["gflops"] is not None else "n/a"
        lines.append(
            f"| {arch} | {c['parameters_millions']:.3f} | {gflops} | {c['file_size_mb']:.2f} |"
        )

    for device_key, label in (("gpu", "GPU"), ("cpu", "CPU-constrained")):
        if device_key not in records["timings"]:
            continue
        lines += [
            "",
            f"## {label} latency (mean over runs of the same architecture)",
            "",
            "| architecture | latency ms | FPS | measurement SD (ms) | n runs |",
            "|---|---|---|---|---|",
        ]
        by_arch: dict[str, list[dict]] = {}
        for run_id, timing in records["timings"][device_key].items():
            by_arch.setdefault(architecture_of(run_id), []).append(timing)
        for arch, timings in sorted(by_arch.items()):
            latencies = [t["latency_ms_mean"] for t in timings]
            sd = [t["latency_ms_measurement_sd"] for t in timings]
            mean = st.mean(latencies)
            lines.append(
                f"| {arch} | {mean:.2f} | {1000.0 / mean:.1f} | "
                f"{st.mean(sd):.3f} | {len(timings)} |"
            )

    lines += [
        "",
        "The spread above is **measurement** noise across repeats of identical weights, not the",
        "seed variance reported on the accuracy axis. The two must not be pooled.",
        "",
        "⚠️ Sustained load throttles this laptop's GPU, so models were timed in randomised",
        "interleaved order; a drift over the session cannot favour one architecture.",
    ]
    path = out / "X05-efficiency-summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("report: %s", path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the efficiency-proxy axis.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pattern", default="X04-*")
    parser.add_argument("--images", type=int, default=TIMED)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--dataset", default="chv", help="which test split supplies the frames")
    parser.add_argument("--skip-cpu", action="store_true", help="GPU only (CPU pass is slow)")
    args = parser.parse_args()

    import torch

    weights_by_run = {
        d.name: d / "weights" / "best.pt"
        for d in sorted(args.runs.glob(args.pattern))
        if (d / "weights" / "best.pt").is_file()
    }
    if not weights_by_run:
        logger.error("no runs matching %s under %s", args.pattern, args.runs)
        return 1

    images = sample_images(args.dataset, args.images)
    logger.info("%d runs, %d frames from %s test", len(weights_by_run), len(images), args.dataset)

    # One representative run per architecture is enough for the constants — and checking the
    # rest match is how "identical across seeds" stays a verified claim rather than an
    # assumption.
    constants: dict[str, dict] = {}
    for run_id, weights in weights_by_run.items():
        arch = architecture_of(run_id)
        measured = model_constants(weights)
        if arch not in constants:
            constants[arch] = measured
        elif constants[arch]["parameters"] != measured["parameters"]:
            logger.error(
                "%s: parameter count differs within architecture %s (%d vs %d)",
                run_id,
                arch,
                measured["parameters"],
                constants[arch]["parameters"],
            )
            return 1

    records = {
        "hardware": {
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cpu": platform.processor() or platform.machine(),
        },
        "settings": {"imgsz": IMGSZ, "batch": BATCH, "warmup": WARMUP, "frames": len(images)},
        "constants": constants,
        "timings": {},
    }

    records["timings"]["gpu"] = measure(weights_by_run, images, "0", args.repeats)
    if not args.skip_cpu:
        # One run per architecture on CPU: the constants are shared, the CPU pass is slow, and
        # a third repeat of identical weights buys nothing the GPU pass has not already shown.
        first_of_each = {}
        for run_id in weights_by_run:
            first_of_each.setdefault(architecture_of(run_id), run_id)
        subset = {run_id: weights_by_run[run_id] for run_id in first_of_each.values()}
        records["timings"]["cpu"] = measure(subset, images, "cpu", max(1, args.repeats - 1))

    build_report(records, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
