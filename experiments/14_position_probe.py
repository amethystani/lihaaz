# experiments/14_position_probe.py
"""Position-resolved probe: does the flat mean-pooled AUROC survive when we
read the residual at a single token position instead of averaging?

Reviewer W1: a flat mean-pooled curve is also what you'd see if (a) the metric
is saturated or (b) mean-pooling averages away a positional signal. This script
re-runs the layer-wise probe with two non-averaged poolings:

  - "last"    : final-token residual (raw and pronoun-stripped conditions)
  - "pronoun" : residual at the first honorific pronoun token (raw only)

If both stay flat across layers, the surface-morphology reading is strengthened;
if either rises with depth, the central claim needs revising. Either way we learn
the real answer before a reviewer asks.
"""
import json
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_stripped_condition, ALL_PRONOUNS
from src.probe.extract import extract_residuals
from src.probe.probe import run_layerwise_probes, print_probe_report

DATASET_PATH = "data/hindi/full_600.json"
MODEL_NAME = "google/gemma-2-2b"
N_LAYERS = 26
OUT_PATH = Path("experiments/results_position_probe.json")

# (condition tag, pooling mode, use_stripped_text)
RUNS = [
    ("raw_last", "last", False),
    ("raw_pronoun", "pronoun", False),
    ("stripped_last", "last", True),
]


def build_examples(pairs, use_stripped: bool):
    sentences, labels, groups = [], [], []
    for pair_idx, p in enumerate(pairs):
        if use_stripped:
            high, low = get_stripped_condition(p)
        else:
            high, low = p.high, p.low
        if not high or not low:
            continue
        sentences.extend([high, low])
        labels.extend([1, 0])
        # Grouped CV: both members of a pair stay on the same split side.
        groups.extend([pair_idx, pair_idx])
    return sentences, np.array(labels), np.array(groups)


def main() -> None:
    device = "cuda"
    pairs = load_dataset(DATASET_PATH)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    layers = list(range(N_LAYERS))
    all_results: dict = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}

    for tag, pooling, use_stripped in RUNS:
        if tag in all_results:
            print(f"[{tag}] already done, skipping")
            continue
        sentences, labels, groups = build_examples(pairs, use_stripped)
        if not sentences:
            print(f"[{tag}] no sentences — skipping")
            continue
        acts = extract_residuals(
            model, sentences, layers=layers, pooling=pooling,
            device=device, pronoun_tokens=ALL_PRONOUNS if pooling == "pronoun" else None,
        )
        results = run_layerwise_probes(acts, labels, groups=groups)
        print_probe_report(results, condition=f"{tag} ({pooling})")
        per_layer = {
            str(lay): {"auroc_mean": v["auroc_mean"], "auroc_std": v["auroc_std"]}
            for lay, v in results.items()
        }
        aurocs = [v["auroc_mean"] for v in per_layer.values()]
        spread = float(max(aurocs) - min(aurocs))
        all_results[tag] = {
            "pooling": pooling,
            "spread": spread,
            "max": float(max(aurocs)),
            "min": float(min(aurocs)),
            "per_layer": per_layer,
        }
        OUT_PATH.write_text(json.dumps(all_results, indent=2))
        print(f"[{tag}] spread={spread:.4f} max={max(aurocs):.4f} → {OUT_PATH}")


if __name__ == "__main__":
    main()
