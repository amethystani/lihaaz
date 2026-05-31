# tests/sae/test_loader.py
from __future__ import annotations

import sys

import pytest


@pytest.mark.unit
def test_resolve_sae_id_defaults():
    from src.sae.loader import resolve_sae_id

    assert resolve_sae_id(7) == (
        "gemma-scope-2b-pt-res",
        "layer_7/width_16k/average_l0_71",
    )


@pytest.mark.unit
def test_resolve_sae_id_9b():
    from src.sae.loader import resolve_sae_id

    assert resolve_sae_id(20, model_size="9b") == (
        "gemma-scope-9b-pt-res",
        "layer_20/width_16k/average_l0_82",
    )


@pytest.mark.unit
def test_resolve_sae_id_width_override():
    from src.sae.loader import resolve_sae_id

    release, sae_id = resolve_sae_id(3, model_size="2b", width="65k")
    assert release == "gemma-scope-2b-pt-res"
    assert sae_id == "layer_3/width_65k/average_l0_71"


@pytest.mark.unit
def test_resolve_sae_id_l0_mapping_2b_uses_71():
    from src.sae.loader import resolve_sae_id

    _, sae_id = resolve_sae_id(7, model_size="2b")
    assert "average_l0_71" in sae_id


@pytest.mark.unit
def test_resolve_sae_id_l0_mapping_9b_uses_82():
    from src.sae.loader import resolve_sae_id

    _, sae_id = resolve_sae_id(7, model_size="9b")
    assert "average_l0_82" in sae_id


@pytest.mark.unit
def test_resolve_sae_id_release_constants():
    from src.sae.loader import (
        GEMMA_2B_RELEASE,
        GEMMA_9B_RELEASE,
        resolve_sae_id,
    )

    assert resolve_sae_id(0, model_size="2b")[0] == GEMMA_2B_RELEASE
    assert resolve_sae_id(0, model_size="9b")[0] == GEMMA_9B_RELEASE


@pytest.mark.unit
def test_resolve_sae_id_layer_interpolation():
    from src.sae.loader import resolve_sae_id

    assert resolve_sae_id(0)[1] == "layer_0/width_16k/average_l0_71"
    assert resolve_sae_id(25)[1] == "layer_25/width_16k/average_l0_71"


@pytest.mark.unit
def test_import_guard_no_sae_lens_required():
    # Importing the loader module must NOT pull in sae_lens / torch.
    import src.sae.loader as loader

    assert "sae_lens" not in sys.modules
    assert callable(loader.resolve_sae_id)
    assert callable(loader.load_sae)
