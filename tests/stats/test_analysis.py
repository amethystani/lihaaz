"""Unit tests for src.stats.analysis.

Covers Bonferroni significance flags, the seeded bootstrap percentile CI, and
the per-condition comparison DataFrame contract. The bootstrap re-seeds
``np.random.default_rng(42)`` on every call, so identical input yields
identical output -- exploited here for determinism assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stats.analysis import bonferroni, bootstrap_ci, compare_conditions


@pytest.mark.unit
def test_bonferroni_exact_logic():
    # threshold = 0.05 / 3 ~= 0.01667; only the first p clears it.
    assert bonferroni([0.01, 0.04, 0.2], alpha=0.05) == [True, False, False]


@pytest.mark.unit
def test_bonferroni_single_element():
    # threshold = 0.05 / 1 = 0.05; 0.04 < 0.05 -> significant.
    assert bonferroni([0.04]) == [True]


@pytest.mark.unit
def test_bonferroni_strict_less_than_at_threshold():
    # A p exactly equal to the threshold is NOT significant (strict ``<``).
    assert bonferroni([0.05], alpha=0.05) == [False]


@pytest.mark.unit
def test_bootstrap_ci_constant_data():
    data = np.array([5.0, 5.0, 5.0, 5.0])
    assert bootstrap_ci(data) == (5.0, 5.0)


@pytest.mark.unit
def test_bootstrap_ci_determinism_and_sanity():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    first = bootstrap_ci(data)
    second = bootstrap_ci(data)

    # (a) seed is fixed -> identical tuples across calls.
    assert first == second

    lo, hi = first
    # (b) ordered bounds.
    assert lo <= hi
    # (c) bounds lie within the data range.
    assert data.min() <= lo <= data.max()
    assert data.min() <= hi <= data.max()
    # (d) the sample mean lies within the CI.
    assert lo <= data.mean() <= hi


@pytest.mark.unit
def test_compare_conditions_contract_and_correctness():
    rng = np.random.default_rng(0)
    baseline_scores = rng.normal(0.0, 1.0, size=8)
    cond_a_scores = rng.normal(10.0, 1.0, size=8)  # clearly separated
    cond_b_scores = rng.normal(0.2, 1.0, size=8)  # near baseline

    df = pd.DataFrame(
        {
            "condition": (
                ["baseline"] * 8 + ["cond_a"] * 8 + ["cond_b"] * 8
            ),
            "score": np.concatenate(
                [baseline_scores, cond_a_scores, cond_b_scores]
            ),
        }
    )

    result = compare_conditions(
        df,
        outcome="score",
        group_col="condition",
        baseline="baseline",
        comparisons=["cond_a", "cond_b"],
    )

    # DataFrame contract.
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "condition",
        "mean",
        "ci_low",
        "ci_high",
        "t",
        "p",
        "significant_bonferroni",
    }

    # One row per comparison, in order.
    assert list(result["condition"]) == ["cond_a", "cond_b"]

    # Each row's mean equals that group's mean.
    assert result.loc[result["condition"] == "cond_a", "mean"].iloc[0] == pytest.approx(
        cond_a_scores.mean()
    )
    assert result.loc[result["condition"] == "cond_b", "mean"].iloc[0] == pytest.approx(
        cond_b_scores.mean()
    )

    # The clearly-separated condition is significant with a small p.
    cond_a_row = result.loc[result["condition"] == "cond_a"].iloc[0]
    assert bool(cond_a_row["significant_bonferroni"]) is True
    assert cond_a_row["p"] < 0.01
