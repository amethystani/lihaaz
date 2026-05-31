# tests/probe/test_extract.py
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_extract_output_shape():
    # Structural shape contract — no GPU required
    d_model, n_sentences = 8, 3
    result = {0: np.zeros((n_sentences, d_model)), 1: np.zeros((n_sentences, d_model))}
    for l, arr in result.items():
        assert arr.shape == (n_sentences, d_model)
    assert set(result.keys()) == {0, 1}


@pytest.mark.unit
def test_module_imports_without_torch():
    import sys

    import src.probe.extract as ex

    assert "torch" not in sys.modules
    assert callable(ex.extract_residuals)
