# experiments/10_multifeature_steering.py
# Tests whether the honorific direction is recoverable by handles richer than a
# single SAE feature:
#   (a) single top feature (control, expected to fail)
#   (b) top-k combined SAE directions (k = 5, 10, 20)
#   (c) supervised difference-in-means direction (aap-class - tum/tu-class)
# All steered along a UNIT direction on generated positions only, alpha-swept,
# scored by the same hon_score proxy as 03b. Saves results_multifeature.json.
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.data.pairs import load_dataset
from src.probe.extract import extract_residuals
from src.sae.loader import load_sae

_spec = importlib.util.spec_from_file_location(
    "causal_sweep", Path(__file__).parent / "03b_causal_sweep.py")
_cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cs)
hon_score, TU_PROMPTS = _cs.hon_score, _cs.TU_PROMPTS

MODEL_NAME = "google/gemma-2-2b"
ALPHAS = [2.0, 4.0, 8.0, 16.0]


def steer_dir(model, direction, prompt, alpha, layer, device, max_new_tokens=60):
    """Steer along a fixed unit direction on generated positions only."""
    vec = (alpha * direction).to(device)

    def hook_fn(value, hook):
        if value.shape[1] == 1:
            return value + vec
        out = value.clone()
        out[:, -1, :] = out[:, -1, :] + vec
        return out

    toks = model.to_tokens(prompt).to(device)
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        out = model.generate(toks, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, toks.shape[1]:])


def unit(v):
    return v / (v.norm() + 1e-8)


def sweep(model, direction, layer, device):
    rows = []
    for a in ALPHAS:
        hs = [hon_score(steer_dir(model, direction, p, a, layer, device)) for p in TU_PROMPTS]
        rows.append({"alpha": a, "mean_hon": float(np.mean(hs))})
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat = json.load(open("experiments/results_full_features.json"))
    layer = int(feat["best_layer"])
    top = [f["feature_idx"] for f in feat["top_features"]]

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    sae = load_sae(layer=layer, model_size="2b", device=device)
    Wdec = sae.W_dec.detach()
    results = {"layer": layer}

    # (a) single feature
    results["single"] = sweep(model, unit(Wdec[top[0]]), layer, device)

    # (b) top-k combined SAE directions
    for k in (5, 10, 20):
        combo = sum(unit(Wdec[i]) for i in top[:k])
        results[f"topk_{k}"] = sweep(model, unit(combo), layer, device)

    # (c) supervised difference-in-means direction from residuals
    pairs = load_dataset("data/hindi/full_600.json")
    sents, labels = [], []
    for p in pairs:
        sents.extend([p.high, p.low]); labels.extend([1, 0])
    acts = extract_residuals(model, sents, layers=[layer], device=device)[layer]
    acts = np.asarray(acts); labels = np.asarray(labels)
    diff = acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
    diff_dir = unit(torch.tensor(diff, dtype=Wdec.dtype, device=device))
    results["supervised_diffmeans"] = sweep(model, diff_dir, layer, device)

    base = float(np.mean([hon_score(model.to_string(
        model.generate(model.to_tokens(p).to(device), max_new_tokens=60, do_sample=False)[0,
                       model.to_tokens(p).shape[1]:])) for p in TU_PROMPTS]))
    results["baseline_mean_hon"] = base

    print(f"baseline: {base:+.3f}")
    for key in ["single", "topk_5", "topk_10", "topk_20", "supervised_diffmeans"]:
        best = max(results[key], key=lambda r: r["mean_hon"])
        print(f"{key:22s} best mean_hon={best['mean_hon']:+.3f} @alpha={best['alpha']} "
              f"(delta {best['mean_hon']-base:+.3f})")

    json.dump(results, open("experiments/results_multifeature.json", "w"), indent=2)
    print("Saved results_multifeature.json")


if __name__ == "__main__":
    main()
