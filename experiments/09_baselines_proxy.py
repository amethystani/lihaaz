# experiments/09_baselines_proxy.py
# Proxy-scored baselines + null distributions, one GPU session:
#   1. zero-shot instruction prompting (tu -> aap rewrite)
#   2. 4-shot exemplar prompting (leave-one-out over the 5 contrastive prompts)
#   3. random-feature steering null (5 random features, alpha=8)
#   4. null distribution of Cohen's d and selectivity over 200 random SAE
#      features (justifies the 0.6 selectivity threshold empirically)
# Scores use the SAME hon_score proxy as 03b (imported from that file so the
# two cannot drift).
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.data.confounds import ALL_PRONOUNS
from src.data.pairs import load_dataset
from src.sae.features import get_feature_activations
from src.sae.loader import load_sae
from src.sae.selectivity import compute_selectivity
from src.steering.additive import steer_generate

_spec = importlib.util.spec_from_file_location(
    "causal_sweep", Path(__file__).parent / "03b_causal_sweep.py")
_cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cs)
hon_score, TU_PROMPTS, AAP_PROMPTS = _cs.hon_score, _cs.TU_PROMPTS, _cs.AAP_PROMPTS

MODEL_NAME = "google/gemma-2-2b"
RANDOM_STEER_ALPHA = 8.0
N_RANDOM_STEER_FEATURES = 5
N_NULL_FEATURES = 200
SEED = 0


def generate(model, prompt: str, device: str, max_new_tokens: int = 60) -> str:
    toks = model.to_tokens(prompt).to(device)
    out = model.generate(toks, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, toks.shape[1]:])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat = json.load(open("experiments/results_full_features.json"))
    layer = int(feat["best_layer"])
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    results: dict = {"layer": layer}

    print("=== 1. zero-shot instruction prompting ===")
    results["zero_shot"] = []
    for p in TU_PROMPTS:
        prompt = ("Neeche diye gaye vaakya ko aadarpurn (aap-form) Hindi mein "
                  f"likho.\nVaakya: {p}\nAap-form:")
        txt = generate(model, prompt, device)
        results["zero_shot"].append({"prompt": p, "output": txt, "hon": hon_score(txt)})

    print("=== 2. 4-shot exemplar prompting (leave-one-out) ===")
    results["four_shot"] = []
    for i, p in enumerate(TU_PROMPTS):
        shots = "".join(
            f"tu-form: {TU_PROMPTS[j]}\naap-form: {AAP_PROMPTS[j]}\n\n"
            for j in range(len(TU_PROMPTS)) if j != i)
        prompt = shots + f"tu-form: {p}\naap-form:"
        txt = generate(model, prompt, device)
        results["four_shot"].append({"prompt": p, "output": txt, "hon": hon_score(txt)})

    print("=== 3. random-feature steering null ===")
    rng = np.random.RandomState(SEED)
    sae = load_sae(layer=layer, model_size="2b", device=device)
    rand_feats = rng.randint(0, sae.W_dec.shape[0], size=N_RANDOM_STEER_FEATURES)
    results["random_steer"] = []
    for f_idx in rand_feats:
        for p in TU_PROMPTS:
            txt = steer_generate(model, sae, p, int(f_idx), RANDOM_STEER_ALPHA,
                                 layer, device=device)
            results["random_steer"].append(
                {"prompt": p, "feature_idx": int(f_idx), "output": txt, "hon": hon_score(txt)})

    print("=== 4. null distribution over random features ===")
    pairs = load_dataset("data/hindi/full_600.json")
    sentences, labels, pron = [], [], []
    for pr in pairs:
        for s, lbl in [(pr.high, 1), (pr.low, 0)]:
            sentences.append(s)
            labels.append(lbl)
            pron.append(int(any(t.lower() in ALL_PRONOUNS for t in s.split())))
    labels_arr, pron_arr = np.array(labels), np.array(pron)
    acts = get_feature_activations(model, sae, sentences, layer=layer, device=device)
    null_feats = rng.choice(acts.shape[1], size=N_NULL_FEATURES, replace=False)
    ds, sels = [], []
    for f_idx in null_feats:
        col = acts[:, f_idx]
        hi, lo = col[labels_arr == 1], col[labels_arr == 0]
        pooled = np.sqrt((hi.std() ** 2 + lo.std() ** 2) / 2)
        ds.append(abs(float(hi.mean() - lo.mean()) / pooled) if pooled > 0 else 0.0)
        sels.append(float(compute_selectivity(col, labels_arr, pron_arr)))
    results["null_dist"] = {
        "n": N_NULL_FEATURES,
        "d_mean": float(np.mean(ds)), "d_p95": float(np.percentile(ds, 95)),
        "sel_mean": float(np.mean(sels)), "sel_p95": float(np.percentile(sels, 95)),
    }

    import statistics as st
    for k in ["zero_shot", "four_shot", "random_steer"]:
        print(f"{k}: mean hon = {st.mean(r['hon'] for r in results[k]):.3f}")
    print("null:", results["null_dist"])
    json.dump(results, open("experiments/results_baselines_proxy.json", "w"),
              ensure_ascii=False, indent=2)
    print("Saved results_baselines_proxy.json")


if __name__ == "__main__":
    main()
