"""Confirmatory probe with all contrasts from one source template in one fold.

This rerun is intentionally separate from ``04_full_probe.py`` so it cannot
overwrite the pair-grouped results reported in earlier drafts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.data.confounds import get_all_conditions
from src.data.pairs import load_dataset
from src.probe.extract import extract_residuals
from src.probe.probe import print_probe_report, run_layerwise_probes


DATASET_PATH = "data/hindi/full_600.json"
OUTPUT_PATH = Path("experiments/results_template_grouped_probe_gemma2b.json")
CONDITIONS = ["raw", "pronoun_stripped"]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This confirmatory rerun requires the project CUDA machine; "
            "Gemma-2-2B plus Gemma Scope does not fit this 8 GB Mac."
        )

    pairs = load_dataset(DATASET_PATH)
    families = [p.confounds.get("template_family") for p in pairs]
    if any(family is None for family in families):
        raise ValueError("Dataset is missing template_family metadata; regenerate it first")
    if len(set(families)) != 200:
        raise ValueError(f"Expected 200 template families, found {len(set(families))}")

    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device="cuda")
    layers = list(range(26))
    output: dict[str, dict] = {}

    for condition in CONDITIONS:
        sentences: list[str] = []
        labels: list[int] = []
        groups: list[str] = []
        for pair in pairs:
            high, low = get_all_conditions(pair)[condition]
            sentences.extend([high, low])
            labels.extend([1, 0])
            family = str(pair.confounds["template_family"])
            groups.extend([family, family])

        activations = extract_residuals(model, sentences, layers=layers, device="cuda")
        results = run_layerwise_probes(
            activations,
            np.asarray(labels),
            groups=np.asarray(groups),
        )
        print_probe_report(results, condition=f"template-grouped/{condition}")
        output[condition] = {
            str(layer): {
                "auroc_mean": values["auroc_mean"],
                "auroc_std": values["auroc_std"],
            }
            for layer, values in results.items()
        }
        OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
