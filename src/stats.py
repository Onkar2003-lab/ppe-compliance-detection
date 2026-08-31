"""Seed aggregation and significance testing: the graduation gate (S5).

Every number in the dissertation passes through here. A per-run score is raw material; it
becomes a *result* only once it is aggregated across seeds with an interval and tested. That
rule exists because the closest prior work (C3) reports single-seed numbers, and beating it
on rigour is one of the two originality levers.

**The statistics are deliberately CI-first, not p-first** (decision P2, 2026-07-23). An
interval says how large an effect is and how well it is pinned down; a p-value only says
whether a null can be rejected, which at three seeds it usually cannot regardless of the
truth. So the interval is the headline and the tests support it.

What is implemented, and why each was chosen:

* **95 % BCa bootstrap CI** (Efron & Tibshirani 1993), the primary quantity. Bias-corrected
  and accelerated because a plain percentile interval is visibly wrong for small, skewed
  samples. Du et al. (2025) is the precedent for using it at three seeds.
* **Paired permutation / sign-flip test** for pairwise comparisons. Exact at this sample
  size rather than asymptotic, and paired because the models share seeds and datasets; an
  unpaired test would throw away the pairing that gives the comparison what little power it
  has.
* **Friedman + Nemenyi** across all three architectures (Demšar 2006), the standard
  multi-model comparison, with the critical-difference values a CD diagram needs.

⚠️ **Power is the honest limitation, and it is reported rather than hidden.** Three seeds per
cell is the cited floor, not a comfortable sample. A paired sign-flip test on n=3 pairs has a
minimum attainable two-sided p of 0.25, so **it cannot return significance at n=3 however
large the effect is**: the pairing is pooled across both directions to reach n=6 where the
comparison allows it, and every test reports its own minimum attainable p so a null result is
never mistaken for evidence of equivalence. This is exactly the ledger's expectation: seed
noise is the same order as the between-model difference, so "no significant difference" is
the likely and honest outcome.

Usage::

    python -m src.stats                    # aggregate every axis, write the report
    python -m src.stats --metric map50     # one metric only
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy import stats as sps

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ACCURACY = Path("D:/runs/X05-accuracy/X05-accuracy-per-run.json")
DEFAULT_VIOLATION = Path("D:/runs/X05-violation")
DEFAULT_OUT = Path("D:/runs/X05-stats")

BOOTSTRAP = 10000
ALPHA = 0.05
RNG_SEED = 0  # the bootstrap is itself an experiment: it gets a fixed seed like any other


# ------------------------------------------------------------------ interval estimation


@dataclass(frozen=True)
class Interval:
    """A point estimate with its uncertainty, and the honesty flags that go with it."""

    mean: float
    low: float
    high: float
    n: int
    method: str

    def as_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "ci95_low": round(self.low, 4),
            "ci95_high": round(self.high, 4),
            "ci95_width": round(self.high - self.low, 4),
            "n_seeds": self.n,
            "method": self.method,
        }


def bca_interval(values: list[float], alpha: float = ALPHA, draws: int = BOOTSTRAP) -> Interval:
    """95 % bias-corrected and accelerated bootstrap CI for the mean.

    Falls back to the percentile interval when the acceleration term is undefined, which
    happens when every jackknife replicate is identical, i.e. the sample has no variance.
    Reporting *which* method produced an interval matters: a silently-degraded BCa is a
    misdescribed statistic.
    """
    sample = np.asarray(values, dtype=float)
    n = len(sample)
    if n < 2:
        value = float(sample[0]) if n else float("nan")
        return Interval(value, value, value, n, "degenerate (n<2, no interval)")

    rng = np.random.default_rng(RNG_SEED)
    observed = float(sample.mean())
    replicates = rng.choice(sample, size=(draws, n), replace=True).mean(axis=1)

    # Bias correction: where the observed statistic sits in the bootstrap distribution.
    proportion = float((replicates < observed).mean())
    if proportion in (0.0, 1.0):
        low, high = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return Interval(observed, float(low), float(high), n, "percentile (BCa undefined)")
    z0 = sps.norm.ppf(proportion)

    # Acceleration: jackknife skewness of the statistic.
    jackknife = np.array([np.delete(sample, i).mean() for i in range(n)])
    deviations = jackknife.mean() - jackknife
    denominator = 6.0 * (float((deviations**2).sum()) ** 1.5)
    if denominator == 0:
        low, high = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return Interval(observed, float(low), float(high), n, "percentile (zero variance)")
    acceleration = float((deviations**3).sum()) / denominator

    z_low, z_high = sps.norm.ppf(alpha / 2), sps.norm.ppf(1 - alpha / 2)
    adjusted = []
    for z in (z_low, z_high):
        numerator = z0 + z
        adjusted.append(sps.norm.cdf(z0 + numerator / (1 - acceleration * numerator)))
    low, high = np.percentile(replicates, [100 * adjusted[0], 100 * adjusted[1]])
    return Interval(observed, float(low), float(high), n, f"BCa ({draws} draws)")


# --------------------------------------------------------------------------- hypothesis


def min_attainable_p(n: int) -> float:
    """Smallest two-sided p a sign-flip test can produce with ``n`` pairs.

    Reported beside every test so an unsurprising null is read as low power rather than as
    evidence that two models perform identically.
    """
    return min(1.0, 2.0 / (2**n)) if n else 1.0


def paired_permutation(a: list[float], b: list[float]) -> dict:
    """Exact paired sign-flip test on the per-pair differences.

    Enumerates all 2^n sign assignments while that is cheap, so the p-value is exact rather
    than sampled. Pairing is by (seed, direction): the models saw the same data under the
    same protocol, so the difference is the meaningful unit.
    """
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(differences)
    if n == 0:
        return {"error": "no paired observations"}

    observed = float(abs(differences.mean()))
    signs = np.array(list(product([1, -1], repeat=n)))
    null = np.abs((signs * differences).mean(axis=1))
    p = float((null >= observed - 1e-12).mean())

    return {
        "mean_difference": round(float(differences.mean()), 4),
        "p_value": round(p, 4),
        "n_pairs": n,
        "min_attainable_p": round(min_attainable_p(n), 4),
        "significant_at_05": bool(p < ALPHA),
        "test": "exact paired sign-flip (permutation)",
    }


def friedman_nemenyi(by_model: dict[str, list[float]]) -> dict:
    """Friedman across architectures, with Nemenyi post-hoc and the critical difference.

    Blocks are (seed, direction) pairs: the conditions every model was measured under.
    Requires at least three models and two blocks; below that the test is not defined and
    saying so is better than emitting a number.
    """
    models = sorted(by_model)
    if len(models) < 3:
        return {"error": f"Friedman needs >=3 models, got {len(models)}"}
    lengths = {len(v) for v in by_model.values()}
    if len(lengths) != 1:
        return {"error": f"unequal block counts across models: {lengths}"}
    blocks = lengths.pop()
    if blocks < 2:
        return {"error": f"Friedman needs >=2 blocks, got {blocks}"}

    matrix = np.array([by_model[m] for m in models])  # models x blocks
    statistic, p = sps.friedmanchisquare(*matrix)

    # Mean rank per model (rank 1 = best within a block; higher score is better).
    ranks = np.apply_along_axis(lambda column: sps.rankdata(-column), 0, matrix)
    mean_ranks = {m: round(float(ranks[i].mean()), 3) for i, m in enumerate(models)}

    result = {
        "test": "Friedman (Demsar 2006)",
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p), 4),
        "models": models,
        "blocks": blocks,
        "mean_ranks": mean_ranks,
        "significant_at_05": bool(p < ALPHA),
    }

    # Critical difference for the CD diagram (Nemenyi, alpha=0.05).
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}.get(len(models))
    if q_alpha:
        k = len(models)
        result["critical_difference"] = round(q_alpha * math.sqrt(k * (k + 1) / (6.0 * blocks)), 3)

    try:
        import scikit_posthocs as sp

        posthoc = sp.posthoc_nemenyi_friedman(matrix.T)
        result["nemenyi_p"] = {
            f"{models[i]} vs {models[j]}": round(float(posthoc.iloc[i, j]), 4)
            for i, j in combinations(range(len(models)), 2)
        }
    except Exception as error:  # noqa: BLE001 - absence is reported, not fatal
        logger.warning("Nemenyi post-hoc unavailable: %s", error)

    return result


# ------------------------------------------------------------------------ data loading


def load_accuracy(path: Path) -> list[dict]:
    """Per-run accuracy records, flattened to the fields the aggregation needs."""
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in records:
        architecture = r["run_id"].split("-")[1]
        rows.append(
            {
                "run_id": r["run_id"],
                "architecture": architecture,
                "trained_on": r["trained_on"],
                "seed": r["seed"],
                "map50_in": r["in_domain"]["map50"],
                "map50_cross": r["cross_domain"]["map50"],
                "map50_95_in": r["in_domain"]["map50_95"],
                "map50_95_cross": r["cross_domain"]["map50_95"],
                "transfer_delta": r["transfer_delta_map50"],
            }
        )
    return rows


def load_violation(directory: Path) -> list[dict]:
    """Per-run violation-axis records."""
    rows = []
    for path in sorted(directory.glob("X04-*-violation.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        parts = r["run_id"].split("-")
        rows.append(
            {
                "run_id": r["run_id"],
                "architecture": parts[1],
                "trained_on": parts[3],
                "seed": int(parts[2][1:]),
                "helmet_violation_recall": r["axes"]["helmet"]["recall"],
                "helmet_violation_precision": r["axes"]["helmet"]["precision"],
                "person_detection_rate": r["person_detection_rate"],
            }
        )
    return rows


# ------------------------------------------------------------------------- aggregation


def aggregate(rows: list[dict], metric: str) -> dict:
    """Aggregate one metric per (architecture, training set), then test between models."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    paired: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        cells[(row["architecture"], row["trained_on"])].append(row[metric])
    # Pair by (trained_on, seed) so every model contributes under identical conditions.
    keyed = {(r["architecture"], r["trained_on"], r["seed"]): r[metric] for r in rows}
    conditions = sorted({(r["trained_on"], r["seed"]) for r in rows})
    architectures = sorted({r["architecture"] for r in rows})
    for architecture in architectures:
        for trained_on, seed in conditions:
            value = keyed.get((architecture, trained_on, seed))
            if value is not None:
                paired[architecture].append(value)

    per_cell = {
        f"{arch} | trained on {dataset}": bca_interval(values).as_dict()
        for (arch, dataset), values in sorted(cells.items())
    }
    per_model = {arch: bca_interval(values).as_dict() for arch, values in sorted(paired.items())}

    complete = {a: v for a, v in paired.items() if len(v) == len(conditions)}
    pairwise = {
        f"{a} vs {b}": paired_permutation(complete[a], complete[b])
        for a, b in combinations(sorted(complete), 2)
    }

    return {
        "metric": metric,
        "per_cell": per_cell,
        "per_model_pooled": per_model,
        "pooling_note": (
            "per_model_pooled pools both training directions to reach n="
            f"{len(conditions)} pairs; per_cell keeps them separate at n=3."
        ),
        "pairwise_permutation": pairwise,
        "friedman_nemenyi": friedman_nemenyi(complete),
    }


def build_report(results: dict, out: Path) -> Path:
    """Human-readable summary. Intervals lead; tests support; power is stated."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "X05-stats.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# X05: seed aggregation + significance (the graduation gate)",
        "",
        "Mean +/- 95 % BCa bootstrap CI over seeds, with paired sign-flip tests and",
        "Friedman/Nemenyi across architectures. **CI-first:** the interval is the claim; the",
        "tests support it. Numbers here are cleared to enter the dissertation.",
        "",
        "> **Power warning.** Three seeds is the cited floor, not a comfortable sample. An",
        "> exact paired sign-flip on n=3 cannot return p<0.25 however large the effect, so a",
        "> null result at that n means *underpowered*, never *equivalent*. Each test states",
        "> its own minimum attainable p.",
    ]

    for metric, result in results.items():
        lines += [
            "",
            f"## {metric}",
            "",
            "| group | mean | 95 % CI | width | n | method |",
            "|---|---|---|---|---|---|",
        ]
        for label, interval in result["per_cell"].items():
            lines.append(
                f"| {label} | {interval['mean']:.4f} | "
                f"[{interval['ci95_low']:.4f}, {interval['ci95_high']:.4f}] | "
                f"{interval['ci95_width']:.4f} | {interval['n_seeds']} | {interval['method']} |"
            )
        lines += ["", "**Pooled across both training directions:**", ""]
        lines += ["| model | mean | 95 % CI | n |", "|---|---|---|---|"]
        for label, interval in result["per_model_pooled"].items():
            lines.append(
                f"| {label} | {interval['mean']:.4f} | "
                f"[{interval['ci95_low']:.4f}, {interval['ci95_high']:.4f}] | "
                f"{interval['n_seeds']} |"
            )

        lines += ["", "**Pairwise (exact paired sign-flip):**", ""]
        lines += [
            "| comparison | mean diff | p | min attainable p | significant |",
            "|---|---|---|---|---|",
        ]
        for label, test in result["pairwise_permutation"].items():
            if "error" in test:
                lines.append(f"| {label} | - | - | - | {test['error']} |")
                continue
            lines.append(
                f"| {label} | {test['mean_difference']:+.4f} | {test['p_value']:.4f} | "
                f"{test['min_attainable_p']:.4f} | "
                f"{'yes' if test['significant_at_05'] else 'no'} |"
            )

        friedman = result["friedman_nemenyi"]
        if "error" in friedman:
            lines += ["", f"Friedman: {friedman['error']}"]
        else:
            verdict = "significant" if friedman["significant_at_05"] else "not significant"
            headline = (
                f"**Friedman:** chi2 = {friedman['statistic']:.4f}, "
                f"p = {friedman['p_value']:.4f}, {friedman['blocks']} blocks, "
                f"{verdict} at 0.05."
            )
            lines += [
                "",
                headline,
                f"Mean ranks: {friedman['mean_ranks']}"
                + (
                    f" · critical difference {friedman['critical_difference']}"
                    if "critical_difference" in friedman
                    else ""
                ),
            ]

    path = out / "X05-stats-summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("report: %s", path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate seeds and test significance.")
    parser.add_argument("--accuracy", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--violation", type=Path, default=DEFAULT_VIOLATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metric", default=None, help="one metric only")
    args = parser.parse_args()

    accuracy = load_accuracy(args.accuracy)
    violation = load_violation(args.violation)
    logger.info("loaded %d accuracy rows, %d violation rows", len(accuracy), len(violation))

    plan = {
        "map50_in (accuracy, in-domain)": (accuracy, "map50_in"),
        "map50_cross (cross-dataset)": (accuracy, "map50_cross"),
        "transfer_delta (in -> cross)": (accuracy, "transfer_delta"),
        "helmet_violation_recall": (violation, "helmet_violation_recall"),
        "person_detection_rate": (violation, "person_detection_rate"),
    }
    if args.metric:
        plan = {k: v for k, v in plan.items() if v[1] == args.metric}
        if not plan:
            logger.error("unknown metric %s", args.metric)
            return 1

    results = {label: aggregate(rows, metric) for label, (rows, metric) in plan.items()}
    build_report(results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
