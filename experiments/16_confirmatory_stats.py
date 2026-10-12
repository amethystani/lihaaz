"""Recover human-evaluation summaries and prompt-paired bootstrap intervals.

This script treats ``human_eval/ratings_long_v2.csv`` and
``human_eval/generations_v2.csv`` as the authoritative item-level records.  It
writes a compact JSON artifact used by the paper instead of copying rounded
values from prose or a hand-maintained table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RATINGS_PATH = Path("human_eval/ratings_long_v2.csv")
GENERATIONS_PATH = Path("human_eval/generations_v2.csv")
OUTPUT_PATH = Path("experiments/results_confirmatory_stats.json")

AXES = [
    "addressee_honorificity",
    "verb_agreement",
    "register_consistency",
    "context_appropriateness",
    "fluency",
    "content_preservation",
]

CONDITION_ORDER = [
    "Baseline (tu-form prompts)",
    "Zero-shot instruction",
    "Random-feature steering (null)",
    "Single SAE feature",
    "Top-5 SAE features",
    "Top-20 SAE features",
    "Supervised difference-in-means",
    "4-shot prompting",
]


def mean_pairwise_spearman(frame: pd.DataFrame, outcome: str) -> dict:
    wide = frame.pivot(index="generation_id", columns="rater_id", values=outcome)
    raters = list(wide.columns)
    pairs: list[dict] = []
    for i, left in enumerate(raters):
        for right in raters[i + 1 :]:
            left_rank = wide[left].rank(method="average").to_numpy()
            right_rank = wide[right].rank(method="average").to_numpy()
            rho = float(np.corrcoef(left_rank, right_rank)[0, 1])
            pairs.append({"raters": [str(left), str(right)], "rho": rho})
    return {"mean_rho": float(np.mean([p["rho"] for p in pairs])), "pairs": pairs}


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    lo, hi = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    ratings = pd.read_csv(RATINGS_PATH)
    generations = pd.read_csv(GENERATIONS_PATH)

    if len(ratings) != 576 or ratings["generation_id"].nunique() != 192:
        raise ValueError("Expected 576 evaluations over 192 generations")
    if ratings["rater_id"].nunique() != 3:
        raise ValueError("Expected three raters")
    if len(generations) != 192:
        raise ValueError("Expected 192 generated responses")

    human_summary: dict[str, dict] = {}
    for condition in CONDITION_ORDER:
        subset = ratings[ratings["condition_name"] == condition]
        human_summary[condition] = {
            "n_generation_rater_evaluations": int(len(subset)),
            "axes": {
                axis: {
                    "mean": float(subset[axis].mean()),
                    "std": float(subset[axis].std(ddof=1)),
                }
                for axis in AXES
            },
        }

    agreement = {
        axis: mean_pairwise_spearman(ratings, axis)
        for axis in AXES[:3]
    }

    proxy_by_prompt = generations.pivot(
        index="prompt_id", columns="condition_name", values="proxy_hon"
    )
    baseline = proxy_by_prompt[CONDITION_ORDER[0]].to_numpy()
    proxy_summary: dict[str, dict] = {}
    for condition_index, condition in enumerate(CONDITION_ORDER):
        values = proxy_by_prompt[condition].to_numpy()
        differences = values - baseline
        if condition_index == 0:
            ci_low = ci_high = 0.0
        else:
            ci_low, ci_high = paired_bootstrap_ci(
                differences, seed=42 + condition_index
            )
        proxy_summary[condition] = {
            "n_prompts": int(len(values)),
            "mean": float(values.mean()),
            "delta_vs_baseline": float(differences.mean()),
            "paired_bootstrap_95_ci": [ci_low, ci_high],
        }

    result = {
        "source_files": [str(RATINGS_PATH), str(GENERATIONS_PATH)],
        "counts": {
            "generations": 192,
            "conditions": 8,
            "prompts_per_condition": 24,
            "raters": 3,
            "generation_rater_evaluations": 576,
            "scalar_judgments": 3456,
        },
        "human_summary": human_summary,
        "agreement_mean_pairwise_spearman": agreement,
        "proxy_summary": proxy_summary,
        "bootstrap": {
            "unit": "matched prompt",
            "resamples": 10_000,
            "interval": 0.95,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
