# src/steering/additive.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer
    from sae_lens import SAE


def steer_generate(
    model: "HookedTransformer", sae: "SAE",
    prompt: str, feature_idx: int, alpha: float, layer: int,
    max_new_tokens: int = 80, device: str = "cuda",
    normalize: bool = True,
) -> str:
    """Additive feature steering applied to generated positions only.

    Two corrections over a naive implementation (review findings C5/C6):

    1. The decoder direction is unit-normalised when ``normalize`` is True, so
       ``alpha`` is the magnitude (in residual-norm units) actually injected and
       is comparable across features and layers. Raw ``W_dec`` rows have very
       different norms; multiplying a raw row by ``alpha=20`` pushes the residual
       far off-manifold and collapses generation into degenerate loops.
    2. The vector is added to the last prefill position and to each newly
       generated token, never to the interior prompt tokens. Steering the whole
       prompt corrupts the context the model conditions on.

    Recommended sweep with ``normalize=True``: ``alpha in {2, 4, 8, 16}``.
    """
    import torch  # noqa: F401  (kept local so importing this module needs no torch)

    direction = sae.W_dec[feature_idx].to(device)
    if normalize:
        direction = direction / (direction.norm() + 1e-8)
    steering_vec = alpha * direction

    def hook_fn(value, hook):
        # Cached generation step: a single new token -> steer it.
        if value.shape[1] == 1:
            return value + steering_vec
        # Prefill: steer only the final prompt position (sets up the first
        # generated token); leave the rest of the prompt untouched.
        out = value.clone()
        out[:, -1, :] = out[:, -1, :] + steering_vec
        return out

    tokens = model.to_tokens(prompt).to(device)
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, tokens.shape[1]:])
