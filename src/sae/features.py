# src/sae/features.py
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sae_lens import SAE
    from transformer_lens import HookedTransformer


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*group1.std()**2 + (n2-1)*group2.std()**2) / (n1+n2-2))
    return 0.0 if pooled_std == 0 else (group1.mean() - group2.mean()) / pooled_std


def get_feature_activations(
    model: HookedTransformer, sae: SAE,
    sentences: list[str], layer: int,
    pooling: str = "mean", device: str = "cuda",
) -> np.ndarray:
    import torch
    all_acts = []
    for sentence in sentences:
        tokens = model.to_tokens(sentence).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            resid = cache["resid_post", layer]
            feat_acts = sae.encode(resid[0])
        vec = feat_acts.mean(0).cpu().float().numpy() if pooling == "mean" else feat_acts[-1].cpu().float().numpy()
        all_acts.append(vec)
    return np.stack(all_acts)


def rank_features_by_effect_size(feature_acts: np.ndarray, labels: np.ndarray, top_k: int = 20) -> list[dict]:
    high = feature_acts[labels == 1]
    low  = feature_acts[labels == 0]
    results = [
        {"feature_idx": f, "cohen_d": abs(cohen_d(high[:, f], low[:, f])),
         "cohen_d_signed": cohen_d(high[:, f], low[:, f]),
         "mean_high": high[:, f].mean(), "mean_low": low[:, f].mean()}
        for f in range(feature_acts.shape[1])
    ]
    results.sort(key=lambda x: x["cohen_d"], reverse=True)
    return results[:top_k]
