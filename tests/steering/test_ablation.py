# tests/steering/test_ablation.py
from __future__ import annotations

import sys

import pytest


class _FakeTokens:
    def to(self, device):
        return self


class _FakeModel:
    def to_tokens(self, prompt):
        return _FakeTokens()


@pytest.mark.unit
def test_module_imports_without_torch():
    # Importing the ablation module must NOT pull in torch / sae_lens.
    import src.steering.ablation as ablation

    assert "torch" not in sys.modules
    assert callable(ablation.ablate_generate)


@pytest.mark.unit
def test_ablate_generate_rejects_unknown_mode():
    # The mode guard runs before any torch import, so this is CPU-testable.
    from src.steering.ablation import ablate_generate

    with pytest.raises(ValueError):
        ablate_generate(_FakeModel(), None, "x", 0, 1.0, 0, mode="bogus")
