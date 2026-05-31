"""Unit tests for src.eval.content.

Guards the lazy-init BLEURT invariant: importing the module must not require
``evaluate``/BLEURT, and the module global ``_bleurt`` stays ``None`` until the
first ``bleurt_score`` call. ``bleurt_score`` itself is not invoked here because
``evaluate`` is absent from the local venv.
"""

from __future__ import annotations

import pytest

import src.eval.content


@pytest.mark.unit
def test_import_succeeds_without_evaluate_and_bleurt_is_none() -> None:
    assert src.eval.content._bleurt is None


@pytest.mark.unit
def test_bleurt_score_is_callable() -> None:
    assert callable(src.eval.content.bleurt_score)
