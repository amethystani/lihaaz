# src/probe/extract.py
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def _normalize_tok(tok: str) -> str:
    """Strip SentencePiece/BPE word-boundary markers and lowercase a token."""
    return tok.replace("▁", "").replace("Ġ", "").strip().lower()


def find_pronoun_position(str_tokens: list[str], pronoun_tokens: set[str]) -> int:
    """Token index of the first honorific pronoun in ``str_tokens``.

    Returns the index of the first token that matches (exactly, then by prefix)
    one of ``pronoun_tokens`` after normalising word-boundary markers. Prefix
    matching catches sub-word splits (e.g. the tokenizer emitting ``aap`` as the
    first piece of ``aapko``). Returns -1 when no pronoun token is found, so the
    caller can fall back to the last position.
    """
    norms = [_normalize_tok(t) for t in str_tokens]
    for i, norm in enumerate(norms):
        if norm and norm in pronoun_tokens:
            return i
    for i, norm in enumerate(norms):
        if len(norm) >= 2 and any(p.startswith(norm) for p in pronoun_tokens):
            return i
    return -1


def extract_residuals(
    model: HookedTransformer,
    sentences: list[str],
    layers: list[int] | None = None,
    pooling: str = "mean",
    device: str = "cuda",
    pronoun_tokens: set[str] | None = None,
) -> dict[int, np.ndarray]:
    """Returns dict: layer_idx -> np.ndarray of shape (n_sentences, d_model).

    ``pooling`` selects how the per-token residual stream is reduced to one
    vector per sentence:
      - ``"mean"``    : mean over all token positions (the default used so far).
      - ``"last"``    : the final token's residual (no positional averaging).
      - ``"pronoun"`` : the residual at the first honorific pronoun token
                        (requires ``pronoun_tokens``), falling back to the last
                        token when no pronoun is present.

    The ``"last"`` and ``"pronoun"`` modes exist to test whether the flat
    layer-wise AUROC seen with mean-pooling is an artefact of averaging away the
    positional signal, or a genuine property of the representation.
    """
    import torch
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    if pooling == "pronoun" and not pronoun_tokens:
        raise ValueError("pooling='pronoun' requires a non-empty pronoun_tokens set")

    all_acts: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    for sentence in sentences:
        tokens = model.to_tokens(sentence).to(device)
        pos = -1
        if pooling == "pronoun":
            str_tokens = model.to_str_tokens(sentence)
            found = find_pronoun_position(str_tokens, pronoun_tokens)
            pos = found if found >= 0 else -1
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for l in layers:
            resid = cache["resid_post", l]  # (1, seq_len, d_model)
            if pooling == "mean":
                vec = resid[0].mean(0).cpu().float().numpy()
            elif pooling == "last":
                vec = resid[0, -1].cpu().float().numpy()
            elif pooling == "pronoun":
                vec = resid[0, pos].cpu().float().numpy()
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            all_acts[l].append(vec)

    return {l: np.stack(all_acts[l]) for l in layers}
