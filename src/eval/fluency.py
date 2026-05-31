# src/eval/fluency.py
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def sentence_ppl(model: HookedTransformer, text: str, device: str = "cuda") -> float:
    import torch
    tokens = model.to_tokens(text).to(device)
    with torch.no_grad():
        logits = model(tokens)
    lp = torch.nn.functional.log_softmax(logits, dim=-1)
    token_lp = lp[0, :-1].gather(1, tokens[0, 1:].unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(-token_lp.mean()).item())


def ppl_inflation_check(baseline_ppls: list[float], steered_ppls: list[float]) -> dict:
    ratio = np.mean(steered_ppls) / np.mean(baseline_ppls)
    return {"baseline_ppl": np.mean(baseline_ppls), "steered_ppl": np.mean(steered_ppls),
            "ratio": ratio, "g3_ppl_pass": ratio < 1.5}
