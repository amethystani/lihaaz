# experiments/07_japanese.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.sae.loader import load_sae
from src.sae.features import get_feature_activations, rank_features_by_effect_size
from src.japanese.pipeline import run_japanese_probe
from src.probe.probe import best_layer

MODEL_NAME = "google/gemma-2-2b"
JP_PATH    = "data/japanese/keigo_200.json"
LAYERS     = list(range(26))

def main():
    device = "cuda"
    model  = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

    jp_results  = run_japanese_probe(model, JP_PATH, LAYERS, device)
    jp_best, jp_auroc = best_layer(jp_results)

    hindi_res = json.load(open("experiments/results_full_probe_gemma2b.json"))
    hindi_ps  = hindi_res["pronoun_stripped"]
    hindi_best = int(max(hindi_ps, key=lambda k: hindi_ps[k]["auroc_mean"]))
    hindi_auroc = hindi_ps[str(hindi_best)]["auroc_mean"]

    print(f"Hindi best layer (ps): {hindi_best}, AUROC: {hindi_auroc:.4f}")
    print(f"Japanese best layer:   {jp_best}, AUROC: {jp_auroc:.4f}")
    print(f"Layer difference: {jp_best - hindi_best} (positive = Japanese later)")

    sae = load_sae(layer=jp_best, model_size="2b", device=device)
    from src.data.pairs import load_dataset
    pairs = load_dataset(JP_PATH)
    sentences = [s for p in pairs for s in [p.high, p.low]]
    labels = np.array([1, 0] * len(pairs))
    feat_acts = get_feature_activations(model, sae, sentences, layer=jp_best, device=device)
    jp_top20  = rank_features_by_effect_size(feat_acts, labels, top_k=20)

    out = {
        "hindi_best_layer": hindi_best, "hindi_auroc": hindi_auroc,
        "japanese_best_layer": jp_best, "japanese_auroc": jp_auroc,
        "jp_top_features": jp_top20,
    }
    json.dump(out, open("experiments/results_japanese.json", "w"), indent=2)
    print("Saved results_japanese.json")

if __name__ == "__main__":
    main()
