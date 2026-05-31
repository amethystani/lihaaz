# src/steering/ablation.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer
    from sae_lens import SAE


def ablate_generate(
    model: HookedTransformer, sae: SAE,
    prompt: str, feature_idx: int, alpha: float, layer: int,
    max_new_tokens: int = 80, device: str = "cuda", mode: str = "negative",
) -> str:
    if mode not in ("negative", "zero_ablation"):
        raise ValueError(f"Unknown mode: {mode}")
    import torch

    tokens = model.to_tokens(prompt).to(device)
    if mode == "negative":
        steering_vec = (-alpha * sae.W_dec[feature_idx]).to(device)

        def hook_fn(value, hook):
            return value + steering_vec.unsqueeze(0).unsqueeze(0)
    else:  # zero_ablation
        def hook_fn(value, hook):
            feat_acts = sae.encode(value)
            feat_acts[..., feature_idx] = 0.0
            return sae.decode(feat_acts)

    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, tokens.shape[1]:])
