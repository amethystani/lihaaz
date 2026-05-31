# experiments/01_probe_pilot.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_stripped_condition
from src.probe.extract import extract_residuals
from src.probe.probe import run_layerwise_probes, print_probe_report

DATASET_PATH = "data/hindi/pilot_30.json"
MODEL_NAME   = "google/gemma-2-2b"
LAYERS       = list(range(26))

def build_xy(pairs, condition="raw"):
    sentences, labels = [], []
    for p in pairs:
        if condition == "raw":
            sentences.extend([p.high, p.low])
        else:
            h, l = get_stripped_condition(p)
            sentences.extend([h, l])
        labels.extend([1, 0])
    return sentences, np.array(labels)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = load_dataset(DATASET_PATH)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

    for condition in ["raw", "pronoun_stripped"]:
        sentences, labels = build_xy(pairs, condition)
        print(f"\nExtracting residuals: {condition} ({len(sentences)} sentences)")
        acts = extract_residuals(model, sentences, layers=LAYERS, device=device)
        results = run_layerwise_probes(acts, labels)
        print_probe_report(results, condition=condition)
        out = {str(l): {"auroc_mean": v["auroc_mean"], "auroc_std": v["auroc_std"]}
               for l, v in results.items()}
        json.dump(out, open(f"experiments/results_probe_{condition}.json", "w"), indent=2)

if __name__ == "__main__":
    main()
