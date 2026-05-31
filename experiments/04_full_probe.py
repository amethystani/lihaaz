# experiments/04_full_probe.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_all_conditions
from src.probe.extract import extract_residuals
from src.probe.probe import run_layerwise_probes, print_probe_report

DATASET_PATH = "data/hindi/full_600.json"
CONDITIONS   = ["raw", "pronoun_stripped", "roman_script", "code_switched"]
MODELS       = [("google/gemma-2-2b", 26, "gemma2b"), ("google/gemma-2-9b", 42, "gemma9b")]

def run_model(model_name, n_layers, tag):
    device = "cuda"
    pairs = load_dataset(DATASET_PATH)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    layers = list(range(n_layers))
    all_results = {}
    for condition in CONDITIONS:
        sentences, labels = [], []
        for p in pairs:
            conds = get_all_conditions(p)
            if condition not in conds:
                continue
            h, l = conds[condition]
            if not h or not l:
                continue
            sentences.extend([h, l])
            labels.extend([1, 0])
        labels = np.array(labels)
        acts = extract_residuals(model, sentences, layers=layers, device=device)
        results = run_layerwise_probes(acts, labels)
        print_probe_report(results, condition=f"{tag}/{condition}")
        all_results[condition] = {
            str(l): {"auroc_mean": v["auroc_mean"], "auroc_std": v["auroc_std"]}
            for l, v in results.items()
        }
    json.dump(all_results, open(f"experiments/results_full_probe_{tag}.json", "w"), indent=2)

def main():
    for model_name, n_layers, tag in MODELS:
        print(f"\n{'='*60}\nRunning {tag}\n{'='*60}")
        run_model(model_name, n_layers, tag)

if __name__ == "__main__":
    main()
