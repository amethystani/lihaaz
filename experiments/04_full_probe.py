# experiments/04_full_probe.py
import json, numpy as np, torch
from pathlib import Path
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_all_conditions
from src.probe.extract import extract_residuals
from src.probe.probe import run_layerwise_probes, print_probe_report

DATASET_PATH = "data/hindi/full_600.json"
CONDITIONS   = ["raw", "pronoun_stripped", "roman_script", "code_switched"]
MODELS       = [("google/gemma-2-2b", 26, "gemma2b")]


def run_model(model_name: str, n_layers: int, tag: str) -> None:
    device = "cuda"
    pairs = load_dataset(DATASET_PATH)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    layers = list(range(n_layers))
    out_path = Path(f"experiments/results_full_probe_{tag}.json")
    all_results: dict = json.loads(out_path.read_text()) if out_path.exists() else {}

    for condition in CONDITIONS:
        if condition in all_results:
            print(f"[{tag}/{condition}] already done, skipping")
            continue
        sentences, labels, groups = [], [], []
        for pair_idx, p in enumerate(pairs):
            conds = get_all_conditions(p)
            if condition not in conds:
                continue
            h, lo = conds[condition]
            if not h or not lo:
                continue
            sentences.extend([h, lo])
            labels.extend([1, 0])
            # Both members of a contrastive pair share a group id so grouped CV
            # never splits them across train/test (prevents leakage; finding C1).
            groups.extend([pair_idx, pair_idx])

        if not sentences:
            print(f"[{tag}/{condition}] no sentences — skipping")
            continue

        labels_arr = np.array(labels)
        groups_arr = np.array(groups)
        acts = extract_residuals(model, sentences, layers=layers, device=device)
        results = run_layerwise_probes(acts, labels_arr, groups=groups_arr)
        print_probe_report(results, condition=f"{tag}/{condition}")
        all_results[condition] = {
            str(lay): {"auroc_mean": v["auroc_mean"], "auroc_std": v["auroc_std"]}
            for lay, v in results.items()
        }
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"[{tag}/{condition}] saved → {out_path}")


def main() -> None:
    for model_name, n_layers, tag in MODELS:
        print(f"\n{'='*60}\nRunning {tag}\n{'='*60}")
        run_model(model_name, n_layers, tag)


if __name__ == "__main__":
    main()
