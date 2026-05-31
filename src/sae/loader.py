# src/sae/loader.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sae_lens import SAE

GEMMA_2B_RELEASE = "gemma-scope-2b-pt-res"
GEMMA_9B_RELEASE = "gemma-scope-9b-pt-res"


def resolve_sae_id(layer: int, model_size: str = "2b", width: str = "16k") -> tuple[str, str]:
    """Map (layer, model_size, width) to the Gemma Scope (release, sae_id) pair.

    Pure string logic -- no GPU/sae_lens needed. ``model_size`` other than "2b"
    selects the 9B release (matching the plan's binary branch).
    """
    release = GEMMA_2B_RELEASE if model_size == "2b" else GEMMA_9B_RELEASE
    l0 = "71" if model_size == "2b" else "82"
    sae_id = f"layer_{layer}/width_{width}/average_l0_{l0}"
    return release, sae_id


def load_sae(layer: int, model_size: str = "2b", width: str = "16k", device: str = "cuda") -> SAE:
    from sae_lens import SAE
    release, sae_id = resolve_sae_id(layer, model_size, width)
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id)
    return sae.to(device)
