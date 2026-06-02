# src/sae/loader.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sae_lens import SAE

GEMMA_2B_RELEASE = "gemma-scope-2b-pt-res"
GEMMA_9B_RELEASE = "gemma-scope-9b-pt-res"

# Actual average_l0 values per layer for width_16k SAEs, from Gemma Scope HF repo.
# l0=71 exists only at layers 8 and 20 for 2B; hardcoding 71 for all layers was wrong.
_L0_MAP_2B_16K: dict[int, int] = {
    0: 46, 1: 40, 2: 53, 3: 59, 4: 60,
    5: 68, 6: 70, 7: 69, 8: 71, 9: 73,
    10: 77, 11: 80, 12: 82, 13: 83, 14: 83,
    15: 78, 16: 78, 17: 77, 18: 74, 19: 73,
    20: 71, 21: 70, 22: 72, 23: 74, 24: 73,
    25: 55,
}
_L0_MAP_9B_16K: dict[int, int] = {
    0: 77, 1: 75, 2: 82, 3: 87, 4: 89,
    5: 94, 6: 97, 7: 97, 8: 97, 9: 98,
    10: 100, 11: 100, 12: 99, 13: 98, 14: 97,
    15: 96, 16: 95, 17: 93, 18: 91, 19: 89,
    20: 87, 21: 86, 22: 85, 23: 84, 24: 83,
    25: 83, 26: 82, 27: 81, 28: 80, 29: 79,
    30: 78, 31: 77, 32: 76, 33: 75, 34: 74,
    35: 73, 36: 72, 37: 71, 38: 70, 39: 68,
    40: 66, 41: 62,
}


def resolve_sae_id(layer: int, model_size: str = "2b", width: str = "16k") -> tuple[str, str]:
    """Map (layer, model_size, width) to the Gemma Scope (release, sae_id) pair.

    Pure string logic -- no GPU/sae_lens needed. ``model_size`` other than "2b"
    selects the 9B release (matching the plan's binary branch).
    """
    release = GEMMA_2B_RELEASE if model_size == "2b" else GEMMA_9B_RELEASE
    l0_map = _L0_MAP_2B_16K if model_size == "2b" else _L0_MAP_9B_16K
    default_l0 = 71 if model_size == "2b" else 82
    l0 = l0_map.get(layer, default_l0)
    sae_id = f"layer_{layer}/width_{width}/average_l0_{l0}"
    return release, sae_id


def load_sae(layer: int, model_size: str = "2b", width: str = "16k", device: str = "cuda") -> SAE:
    from sae_lens import SAE
    release, sae_id = resolve_sae_id(layer, model_size, width)
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id)
    return sae.to(device)
