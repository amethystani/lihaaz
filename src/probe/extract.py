# src/probe/extract.py
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def extract_residuals(
    model: HookedTransformer,
    sentences: list[str],
    layers: list[int] | None = None,
    pooling: str = "mean",
    device: str = "cuda",
) -> dict[int, np.ndarray]:
    """Returns dict: layer_idx -> np.ndarray of shape (n_sentences, d_model)."""
    import torch
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    all_acts: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    for sentence in sentences:
        tokens = model.to_tokens(sentence).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for l in layers:
            resid = cache["resid_post", l]  # (1, seq_len, d_model)
            if pooling == "mean":
                vec = resid[0].mean(0).cpu().float().numpy()
            elif pooling == "last":
                vec = resid[0, -1].cpu().float().numpy()
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            all_acts[l].append(vec)

    return {l: np.stack(all_acts[l]) for l in layers}
