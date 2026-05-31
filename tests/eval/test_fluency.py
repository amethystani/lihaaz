"""Unit tests for src.eval.fluency.

Covers the pure-numpy ``ppl_inflation_check`` G3 fluency gate (PPL inflation
ratio < 1.5x, strict ``<``) and a lazy-import guard proving ``sentence_ppl``
does not pull in torch at module import time.
"""

from __future__ import annotations

import sys

import pytest

from src.eval.fluency import ppl_inflation_check


@pytest.mark.unit
def test_passes_when_steered_close_to_baseline() -> None:
    # Arrange
    baseline = [10.0, 12.0, 11.0]
    steered = [11.0, 12.0, 10.0]

    # Act
    result = ppl_inflation_check(baseline, steered)

    # Assert
    assert result["g3_ppl_pass"]
    assert result["ratio"] < 1.5


@pytest.mark.unit
def test_fails_when_steered_at_least_1_5x_baseline() -> None:
    # Arrange
    baseline = [10.0, 10.0]
    steered = [20.0, 20.0]

    # Act
    result = ppl_inflation_check(baseline, steered)

    # Assert
    assert not result["g3_ppl_pass"]
    assert result["ratio"] == pytest.approx(2.0)


@pytest.mark.unit
def test_ratio_and_means_match_expected_values() -> None:
    # Arrange
    baseline = [10.0, 12.0, 11.0]
    steered = [11.0, 12.0, 10.0]

    # Act
    result = ppl_inflation_check(baseline, steered)

    # Assert
    assert result["baseline_ppl"] == pytest.approx(11.0)
    assert result["steered_ppl"] == pytest.approx(11.0)
    assert result["ratio"] == pytest.approx(11.0 / 11.0)


@pytest.mark.unit
def test_exact_boundary_ratio_of_1_5_does_not_pass() -> None:
    # Arrange
    baseline = [10.0, 10.0]
    steered = [15.0, 15.0]

    # Act
    result = ppl_inflation_check(baseline, steered)

    # Assert
    assert result["ratio"] == pytest.approx(1.5)
    assert not result["g3_ppl_pass"]


@pytest.mark.unit
def test_import_does_not_load_torch() -> None:
    # Act
    import src.eval.fluency  # noqa: F401

    # Assert
    assert "torch" not in sys.modules
