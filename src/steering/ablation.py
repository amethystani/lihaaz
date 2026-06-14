# src/steering/ablation.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer
    from sae_lens import SAE


def ablate_generate(
    model: "HookedTransformer", sae: "SAE",
    prompt: str, feature_idx: int, alpha: float, layer: int,
    max_new_tokens: int = 80, device: str = "cuda", mode: str = "negative",
    normalize: bool = True,
) -> str:
    """Suppress a single SAE feature during generation (necessity arm).

    Two modes:

    * ``negative`` — subtract ``alpha * unit(W_dec[f])`` from the residual at the
      generated positions (the additive-steering inverse). ``normalize`` makes
      ``alpha`` comparable across features (finding C5).
    * ``zero_ablation`` — remove ONLY feature f's contribution while leaving the
      rest of the residual, including the SAE reconstruction error, intact:
      ``resid -= encode(resid)[f] * W_dec[f]``. The previous implementation
      returned ``decode(encode(resid))`` with feature f zeroed, which discards
      the (large) reconstruction-error term and collapses generation into
      garbage — this is why the committed zero-ablation outputs were unusable
      (finding A2/C5).
    """
    if mode not in ("negative", "zero_ablation"):
        raise ValueError(f"Unknown mode: {mode}")
    import torch  # noqa: F401  (local import: module import stays torch-free)

    tokens = model.to_tokens(prompt).to(device)

    if mode == "negative":
        direction = sae.W_dec[feature_idx].to(device)
        if normalize:
            direction = direction / (direction.norm() + 1e-8)
        steering_vec = -alpha * direction

        def hook_fn(value, hook):
            if value.shape[1] == 1:
                return value + steering_vec
            out = value.clone()
            out[:, -1, :] = out[:, -1, :] + steering_vec
            return out
    else:  # zero_ablation
        w_dec_i = sae.W_dec[feature_idx].to(device)

        def hook_fn(value, hook):
            feat_acts = sae.encode(value)
            contrib = feat_acts[..., feature_idx:feature_idx + 1] * w_dec_i
            return value - contrib

    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, tokens.shape[1]:])
