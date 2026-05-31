# experiments/05_full_features.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_stripped_condition, ALL_PRONOUNS
from src.sae.loader import load_sae
from src.sae.features import get_feature_activations, rank_features_by_effect_size
from src.sae.selectivity import score_top_features

def main():
    probe_res = json.load(open("experiments/results_full_probe_gemma2b.json"))
    best_layer_idx = int(max(probe_res["pronoun_stripped"],
                             key=lambda k: probe_res["pronoun_stripped"][k]["auroc_mean"]))
    print(f"Best layer: {best_layer_idx}")

    device = "cuda"
    pairs = load_dataset("data/hindi/full_600.json")
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    sae   = load_sae(layer=best_layer_idx, model_size="2b", device=device)

    sentences, labels = [], []
    confound_arrays: dict[str, list[int]] = {"pronoun_present": [], "is_roman_script": [], "is_code_switched": []}
    for p in pairs:
        sentences.extend([p.high, p.low])
        labels.extend([1, 0])
        for s, lbl in [(p.high, 1), (p.low, 0)]:
            confound_arrays["pronoun_present"].append(int(any(t.lower() in ALL_PRONOUNS for t in s.split())))
            confound_arrays["is_roman_script"].append(
                int((lbl==1 and "script_roman_high" in p.confounds) or (lbl==0 and "script_roman_low" in p.confounds)))
            confound_arrays["is_code_switched"].append(
                int((lbl==1 and "code_switched_high" in p.confounds) or (lbl==0 and "code_switched_low" in p.confounds)))

    labels = np.array(labels)
    confound_labels = {k: np.array(v) for k, v in confound_arrays.items()}

    print(f"Extracting SAE activations for {len(sentences)} sentences...")
    feature_acts = get_feature_activations(model, sae, sentences, layer=best_layer_idx, device=device)

    top50 = rank_features_by_effect_size(feature_acts, labels, top_k=50)
    sel   = score_top_features(feature_acts, labels, confound_labels, [f["feature_idx"] for f in top50])

    print("\n=== Top-20 features ===")
    for f, s in zip(top50[:20], sel[:20]):
        print(f"  Feat {f['feature_idx']:6d}: d={f['cohen_d']:.3f} "
              f"sel_pronoun={s.get('sel_pronoun_present',0):.3f} "
              f"sel_roman={s.get('sel_is_roman_script',0):.3f} "
              f"mean_sel={s['mean_selectivity']:.3f}")

    json.dump({"top_features": top50, "selectivity": sel, "best_layer": best_layer_idx},
              open("experiments/results_full_features.json", "w"), indent=2)
    print("Saved results_full_features.json")
    print("\n[MANUAL] Read top-20 highest-activating examples for each feature. Record in feature_validation_notes.md")

if __name__ == "__main__":
    main()
