# src/japanese/pipeline.py
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from src.data.pairs import load_dataset
from src.probe.extract import extract_residuals
from src.probe.probe import run_layerwise_probes, print_probe_report

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def run_japanese_probe(
    model: HookedTransformer, dataset_path: str, layers: list[int], device: str = "cuda",
) -> dict:
    pairs = load_dataset(dataset_path)
    sentences = [s for p in pairs for s in [p.high, p.low]]
    labels = np.array([1, 0] * len(pairs))
    acts = extract_residuals(model, sentences, layers=layers, device=device)
    results = run_layerwise_probes(acts, labels)
    print_probe_report(results, condition="japanese_keigo")
    return results
