"""Unit tests for seed aggregation and significance testing (X05/S5).

This module is the graduation gate: if it is wrong, every number in the dissertation is
wrong, and wrong in the direction that is hardest to catch by eye. So the tests use toy
inputs whose answers are known analytically rather than real run data.

The cases that matter most are the ones guarding against *overclaiming*: that a degenerate
sample cannot silently produce a confident-looking interval, that the exact permutation test
reports honest p-values at tiny n, and that a test which cannot possibly reach significance
says so rather than returning a bare null.
"""

from __future__ import annotations

import pytest

from src.stats import (
    aggregate,
    bca_interval,
    friedman_nemenyi,
    min_attainable_p,
    paired_permutation,
)

# ------------------------------------------------------------------ interval estimation


def test_interval_brackets_the_mean() -> None:
    interval = bca_interval([0.70, 0.72, 0.74])
    assert interval.low <= interval.mean <= interval.high
    assert interval.mean == pytest.approx(0.72, abs=1e-9)
    assert interval.n == 3


def test_zero_variance_sample_gives_a_zero_width_interval_and_says_why() -> None:
    """Identical seeds mean no information about spread — the method must admit it."""
    interval = bca_interval([0.5, 0.5, 0.5])
    assert interval.low == interval.high == pytest.approx(0.5)
    # Degraded to the percentile method, and the label says so rather than claiming BCa.
    assert interval.method.startswith("percentile")


def test_single_observation_is_flagged_not_silently_intervalled() -> None:
    interval = bca_interval([0.42])
    assert interval.mean == interval.low == interval.high == pytest.approx(0.42)
    assert "degenerate" in interval.method


def test_wider_spread_gives_a_wider_interval() -> None:
    tight = bca_interval([0.700, 0.705, 0.710])
    loose = bca_interval([0.60, 0.72, 0.84])
    assert (loose.high - loose.low) > (tight.high - tight.low)


def test_interval_is_reproducible() -> None:
    """The bootstrap is an experiment; a fixed seed makes it reportable."""
    values = [0.61, 0.68, 0.73]
    assert bca_interval(values).as_dict() == bca_interval(values).as_dict()


def test_as_dict_reports_width_and_method() -> None:
    payload = bca_interval([0.1, 0.2, 0.3]).as_dict()
    assert payload["ci95_width"] == pytest.approx(payload["ci95_high"] - payload["ci95_low"])
    assert payload["n_seeds"] == 3
    assert payload["method"]


# ------------------------------------------------------------------------- power limits


def test_minimum_attainable_p_at_three_pairs_is_a_quarter() -> None:
    """The number that stops a null result being read as equivalence."""
    assert min_attainable_p(3) == pytest.approx(0.25)
    assert min_attainable_p(6) == pytest.approx(2 / 64)


def test_three_pairs_cannot_reach_significance_even_when_separated() -> None:
    """A perfectly separated comparison at n=3 still cannot clear 0.05 — by construction."""
    test = paired_permutation([0.90, 0.91, 0.92], [0.10, 0.11, 0.12])
    assert test["p_value"] >= 0.25
    assert test["significant_at_05"] is False
    assert test["min_attainable_p"] == pytest.approx(0.25)


def test_six_pairs_can_reach_significance() -> None:
    a = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
    b = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
    test = paired_permutation(a, b)
    assert test["significant_at_05"] is True
    assert test["n_pairs"] == 6


# ------------------------------------------------------------------------ paired tests


def test_identical_inputs_give_no_difference_and_p_of_one() -> None:
    test = paired_permutation([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert test["mean_difference"] == pytest.approx(0.0)
    assert test["p_value"] == pytest.approx(1.0)


def test_difference_sign_follows_argument_order() -> None:
    assert paired_permutation([0.6, 0.7], [0.5, 0.6])["mean_difference"] > 0
    assert paired_permutation([0.5, 0.6], [0.6, 0.7])["mean_difference"] < 0


def test_empty_input_is_an_error_not_a_result() -> None:
    assert "error" in paired_permutation([], [])


# ---------------------------------------------------------------------------- Friedman


def test_friedman_needs_three_models() -> None:
    result = friedman_nemenyi({"a": [1, 2, 3], "b": [2, 3, 4]})
    assert "error" in result


def test_friedman_rejects_unequal_block_counts() -> None:
    """Ragged input would silently compare models measured under different conditions."""
    result = friedman_nemenyi({"a": [1, 2, 3], "b": [2, 3], "c": [3, 4, 5]})
    assert "error" in result


def test_friedman_ranks_a_consistent_winner_first() -> None:
    result = friedman_nemenyi(
        {
            "best": [0.90, 0.91, 0.92, 0.93],
            "middle": [0.50, 0.51, 0.52, 0.53],
            "worst": [0.10, 0.11, 0.12, 0.13],
        }
    )
    assert result["mean_ranks"]["best"] < result["mean_ranks"]["worst"]
    assert result["blocks"] == 4
    assert "critical_difference" in result


# -------------------------------------------------------------------------- aggregation


def rows() -> list[dict]:
    """Two architectures x two training sets x three seeds, with a known structure."""
    out = []
    for architecture, offset in (("y8n", 0.0), ("y11n", 0.02)):
        for trained_on, base in (("sh17", 0.60), ("chv", 0.80)):
            for seed, jitter in enumerate((0.00, 0.01, 0.02)):
                out.append(
                    {
                        "run_id": f"X04-{architecture}-s{seed}-{trained_on}",
                        "architecture": architecture,
                        "trained_on": trained_on,
                        "seed": seed,
                        "score": base + offset + jitter,
                    }
                )
    return out


def test_aggregate_keeps_training_directions_separate_in_cells() -> None:
    result = aggregate(rows(), "score")
    assert "y8n | trained on sh17" in result["per_cell"]
    assert "y8n | trained on chv" in result["per_cell"]
    assert result["per_cell"]["y8n | trained on sh17"]["n_seeds"] == 3


def test_aggregate_pools_directions_to_buy_power() -> None:
    """Pooling is the only way this design reaches an n where a test can say anything."""
    result = aggregate(rows(), "score")
    assert result["per_model_pooled"]["y8n"]["n_seeds"] == 6


def test_aggregate_pairs_like_with_like() -> None:
    """The known +0.02 offset must come back exactly — proof the pairing is aligned."""
    result = aggregate(rows(), "score")
    test = result["pairwise_permutation"]["y11n vs y8n"]
    assert test["mean_difference"] == pytest.approx(0.02, abs=1e-9)
    assert test["n_pairs"] == 6


def test_aggregate_reports_the_pooling_it_did() -> None:
    assert "pools both training directions" in aggregate(rows(), "score")["pooling_note"]
