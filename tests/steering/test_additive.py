# tests/steering/test_additive.py
from __future__ import annotations

import sys

import pytest


@pytest.mark.unit
def test_module_imports_without_torch():
    # Importing the steering module must NOT pull in torch / sae_lens.
    import src.steering.additive as additive

    assert "torch" not in sys.modules
    assert callable(additive.steer_generate)
