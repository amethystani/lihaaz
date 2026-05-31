# experiments/02_sae_features.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_stripped_condition, ALL_PRONOUNS
from src.sae.loader import load_sae
from src.sae.features import get_feature_activations, rank_features_by_effect_size
from src.sae.selectivity import score_top_features

def main():
    probe_res = json.load(open("experiments/results_probe_pronoun_stripped.json"))
    best_layer_idx = int(max(probe_res, key=lambda k: probe_res[k]["auroc_mean"]))
    print(f"Using layer {best_layer_idx} (AUROC={probe_res[str(best_layer_idx)]['auroc_mean']:.4f})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = load_dataset("data/hindi/pilot_30.json")
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    sae   = load_sae(layer=best_layer_idx, model_size="2b", device=device)

    sentences, labels, pronoun_flags = [], [], []
    for p in pairs:
        sentences.extend([p.high, p.low])
        labels.extend([1, 0])
        for s in [p.high, p.low]:
            pronoun_flags.append(int(any(t.lower() in ALL_PRONOUNS for t in s.split())))
    labels = np.array(labels)
    confound_labels = {"pronoun_present": np.array(pronoun_flags)}

    print(f"Extracting SAE activations for {len(sentences)} sentences...")
    feature_acts = get_feature_activations(model, sae, sentences, layer=best_layer_idx, device=device)

    top20 = rank_features_by_effect_size(feature_acts, labels, top_k=20)
    top_indices = [f["feature_idx"] for f in top20]
    sel_scores  = score_top_features(feature_acts, labels, confound_labels, top_indices)

    print("\n=== Top-20 features ===")
    for f, s in zip(top20, sel_scores):
        print(f"  Feat {f['feature_idx']:6d}: d={f['cohen_d']:.3f}, sel={s['mean_selectivity']:.3f}")

    g2_pass = any(f["cohen_d"] > 0.5 and s["mean_selectivity"] > 0.6 for f, s in zip(top20, sel_scores))
    print(f"\nG2: {'PASS' if g2_pass else 'FAIL'}")

    json.dump({"top_features": top20, "selectivity": sel_scores, "best_layer": best_layer_idx},
              open("experiments/results_sae_features.json", "w"), indent=2)

if __name__ == "__main__":
    main()
