"""Unit tests for src.eval.rubric.

Covers the canonical rubric schema (axis names + CSV header order), the frozen
``RubricScore`` per-generation interface with its boundary validation, and the
``validate_rubric_df`` boundary check for human-entered rubric CSVs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.rubric import (
    RUBRIC_AXES,
    RUBRIC_CSV_COLUMNS,
    RubricScore,
    validate_rubric_df,
)

_EXPECTED_AXES = (
    "addressee",
    "verb_agreement",
    "register_consistency",
    "context_appropriateness",
    "fluency",
    "content_preservation",
)


def _valid_score_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"condition": "baseline", "prompt_id": "p001"}
    base.update({axis: 2 for axis in RUBRIC_AXES})
    base.update(overrides)
    return base


def _valid_df() -> pd.DataFrame:
    row = {"condition": "baseline", "prompt_id": "p001"}
    row.update({axis: 2 for axis in RUBRIC_AXES})
    return pd.DataFrame([row], columns=list(RUBRIC_CSV_COLUMNS))


@pytest.mark.unit
def test_rubric_axes_exact_names_and_order():
    assert RUBRIC_AXES == _EXPECTED_AXES


@pytest.mark.unit
def test_rubric_csv_columns_prepend_condition_and_prompt_id():
    assert RUBRIC_CSV_COLUMNS == ("condition", "prompt_id", *RUBRIC_AXES)


@pytest.mark.unit
def test_rubric_score_constructs_at_lower_boundary():
    score = RubricScore(**_valid_score_kwargs(**{axis: 0 for axis in RUBRIC_AXES}))
    assert all(getattr(score, axis) == 0 for axis in RUBRIC_AXES)


@pytest.mark.unit
def test_rubric_score_constructs_at_upper_boundary():
    score = RubricScore(**_valid_score_kwargs(**{axis: 3 for axis in RUBRIC_AXES}))
    assert all(getattr(score, axis) == 3 for axis in RUBRIC_AXES)


@pytest.mark.unit
def test_rubric_score_rejects_value_above_max():
    with pytest.raises(ValueError):
        RubricScore(**_valid_score_kwargs(fluency=4))


@pytest.mark.unit
def test_rubric_score_rejects_value_below_min():
    with pytest.raises(ValueError):
        RubricScore(**_valid_score_kwargs(addressee=-1))


@pytest.mark.unit
def test_validate_rubric_df_returns_none_for_well_formed_df():
    assert validate_rubric_df(_valid_df()) is None


@pytest.mark.unit
def test_validate_rubric_df_raises_on_missing_column():
    df = _valid_df().drop(columns=["verb_agreement"])
    with pytest.raises(ValueError):
        validate_rubric_df(df)


@pytest.mark.unit
def test_validate_rubric_df_raises_on_out_of_range_score():
    df = _valid_df()
    df.loc[0, "fluency"] = 5
    with pytest.raises(ValueError):
        validate_rubric_df(df)
