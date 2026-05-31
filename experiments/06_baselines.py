# experiments/06_baselines.py
import json, torch, numpy as np
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.sae.loader import load_sae
from src.steering.additive import steer_generate
from src.steering.ablation import ablate_generate
from src.baselines.prompting import zero_shot_prompt, four_shot_prompt, generate_with_prompt

MODEL_NAME = "google/gemma-2-2b"
ALPHAS     = [5.0, 10.0, 20.0, 30.0, 40.0]
N_TEST     = 40  # held-out prompts

def main():
    feats_res = json.load(open("experiments/results_full_features.json"))
    best_layer = feats_res["best_layer"]
    top3 = [f["feature_idx"] for f in feats_res["top_features"][:3]]
    print(f"Top-3 features: {top3}, best layer: {best_layer}")

    device = "cuda"
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    sae   = load_sae(layer=best_layer, model_size="2b", device=device)

    all_pairs = load_dataset("data/hindi/full_600.json")
    test_pairs = all_pairs[-120:][:N_TEST]  # held-out set

    results = {}

    # Zero-shot + 4-shot baselines
    for method, prompt_fn in [("zero_shot", zero_shot_prompt), ("four_shot", four_shot_prompt)]:
        for p in test_pairs:
            prompt = prompt_fn(p.low, "aap")
            out = generate_with_prompt(model, prompt, device=device)
            results.setdefault(method, []).append({"prompt": p.low, "output": out})

    # Alpha sweep for top-3 features
    for feat_idx in top3:
        for alpha in ALPHAS:
            for p in test_pairs:
                out = steer_generate(model, sae, p.low, feat_idx, alpha, best_layer, device=device)
                key = f"steer_feat{feat_idx}_alpha{alpha}"
                results.setdefault(key, []).append({"prompt": p.low, "output": out})

    # Ablation on aap prompts
    for feat_idx in top3:
        for p in test_pairs:
            out_neg  = ablate_generate(model, sae, p.high, feat_idx, 20.0, best_layer, mode="negative", device=device)
            out_zero = ablate_generate(model, sae, p.high, feat_idx, 20.0, best_layer, mode="zero_ablation", device=device)
            results.setdefault(f"ablate_neg_feat{feat_idx}", []).append({"prompt": p.high, "output": out_neg})
            results.setdefault(f"ablate_zero_feat{feat_idx}", []).append({"prompt": p.high, "output": out_zero})

    json.dump(results, open("experiments/results_baselines_and_steering.json", "w"), ensure_ascii=False, indent=2)
    print("Saved. [MANUAL] Score with rubric -> experiments/rubric_scores_full.csv")

if __name__ == "__main__":
    main()
