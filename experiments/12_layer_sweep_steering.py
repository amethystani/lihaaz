# experiments/12_layer_sweep_steering.py
# Tests whether steering at LATER layers succeeds where layer-2 fails.
# For each candidate layer, computes the supervised difference-in-means
# direction (aap - tum/tu) at that layer from residuals, steers tu-form prompts
# along it (generated positions only, alpha sweep), and scores with the hon
# proxy. Saves results_layersweep.json.
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.data.pairs import load_dataset

_spec = importlib.util.spec_from_file_location(
    "causal_sweep", Path(__file__).parent / "03b_causal_sweep.py")
_cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cs)
hon_score, TU_PROMPTS = _cs.hon_score, _cs.TU_PROMPTS

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [2, 5, 8, 12, 16, 20]
ALPHAS = [4.0, 8.0, 16.0]


def diffmeans_dir(model, sents, labels, layer, device):
    name = f"blocks.{layer}.hook_resid_post"
    vecs = []
    for s in sents:
        toks = model.to_tokens(s).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n == name, return_type=None)
        vecs.append(cache[name][0].mean(0).cpu().float().numpy())
    X = np.stack(vecs); y = np.array(labels)
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    t = torch.tensor(d, dtype=torch.bfloat16, device=device)
    return t / (t.norm() + 1e-8)


def steer(model, direction, prompt, alpha, layer, device, max_new_tokens=60):
    vec = (alpha * direction).to(device)

    def hook_fn(value, hook):
        if value.shape[1] == 1:
            return value + vec
        out = value.clone(); out[:, -1, :] = out[:, -1, :] + vec
        return out

    toks = model.to_tokens(prompt).to(device)
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        out = model.generate(toks, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, toks.shape[1]:])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype="bfloat16")
    pairs = load_dataset("data/hindi/full_600.json")
    sents, labels = [], []
    for p in pairs:
        sents += [p.high, p.low]; labels += [1, 0]

    base = float(np.mean([
        hon_score(model.to_string(model.generate(
            model.to_tokens(p).to(device), max_new_tokens=60, do_sample=False)[0,
            model.to_tokens(p).shape[1]:])) for p in TU_PROMPTS]))
    results = {"baseline_mean_hon": base, "layers": {}}
    print(f"baseline: {base:+.3f}")

    for layer in LAYERS:
        direction = diffmeans_dir(model, sents, labels, layer, device)
        rows = []
        for a in ALPHAS:
            hs = [hon_score(steer(model, direction, p, a, layer, device)) for p in TU_PROMPTS]
            rows.append({"alpha": a, "mean_hon": float(np.mean(hs))})
        best = max(rows, key=lambda r: r["mean_hon"])
        results["layers"][layer] = {"sweep": rows, "best": best}
        print(f"layer {layer:2d}: best mean_hon={best['mean_hon']:+.3f} @alpha={best['alpha']} "
              f"(delta {best['mean_hon']-base:+.3f})")

    json.dump(results, open("experiments/results_layersweep.json", "w"), indent=2)
    print("Saved results_layersweep.json")


if __name__ == "__main__":
    main()
