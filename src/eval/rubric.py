"""Rubric schema and per-generation scoring interface.

Canonical single source of truth for the 6-axis human-rating rubric used to
score generations. Downstream experiment scripts import :data:`RUBRIC_AXES`,
:data:`RUBRIC_CSV_COLUMNS`, :class:`RubricScore`, and :func:`validate_rubric_df`
from here. Each axis is an integer in ``[SCORE_MIN, SCORE_MAX]``. Raters save a
CSV whose header is :data:`RUBRIC_CSV_COLUMNS`, consumed downstream (e.g.
``src.stats.analysis.compare_conditions``) via pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# The 6 rubric axes, in canonical order.
RUBRIC_AXES: tuple[str, ...] = (
    "addressee",
    "verb_agreement",
    "register_consistency",
    "context_appropriateness",
    "fluency",
    "content_preservation",
)

SCORE_MIN: int = 0
SCORE_MAX: int = 3

# Full expected rubric-CSV header, in order: identifiers then the 6 axes.
RUBRIC_CSV_COLUMNS: tuple[str, ...] = ("condition", "prompt_id") + RUBRIC_AXES


@dataclass(frozen=True)
class RubricScore:
    """A single human-rated generation: identifiers plus 6 integer axis scores."""

    condition: str
    prompt_id: str
    addressee: int
    verb_agreement: int
    register_consistency: int
    context_appropriateness: int
    fluency: int
    content_preservation: int

    def __post_init__(self) -> None:
        for axis in RUBRIC_AXES:
            value = getattr(self, axis)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"rubric axis '{axis}' must be an int, got {value!r}"
                )
            if value < SCORE_MIN or value > SCORE_MAX:
                raise ValueError(
                    f"rubric axis '{axis}' must be in "
                    f"[{SCORE_MIN}, {SCORE_MAX}], got {value}"
                )


def validate_rubric_df(df: "pd.DataFrame") -> None:
    """Validate a loaded rubric CSV (human-entered data); fail fast on errors.

    Raises:
        ValueError: if any required column is missing, or any axis column
            holds a value outside ``[SCORE_MIN, SCORE_MAX]``.
    """
    missing = [col for col in RUBRIC_CSV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"rubric CSV missing required columns: {missing}")

    for axis in RUBRIC_AXES:
        column = df[axis]
        out_of_range = (column < SCORE_MIN) | (column > SCORE_MAX)
        if bool(out_of_range.any()):
            raise ValueError(
                f"rubric axis '{axis}' has values outside "
                f"[{SCORE_MIN}, {SCORE_MAX}]"
            )
