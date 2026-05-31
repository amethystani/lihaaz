# tests/japanese/test_pipeline.py
from __future__ import annotations

import sys

import pytest


@pytest.mark.unit
def test_module_imports_without_torch():
    # The Japanese pipeline reuses the probe stack, all of which must stay
    # importable without torch / transformer_lens on a CPU-only box.
    import src.japanese.pipeline as pipeline

    assert "torch" not in sys.modules
    assert "transformer_lens" not in sys.modules
    assert callable(pipeline.run_japanese_probe)
