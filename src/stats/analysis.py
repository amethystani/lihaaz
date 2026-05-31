# src/stats/analysis.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(data: np.ndarray, n: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    means = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n)]
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    threshold = alpha / len(p_values)
    return [p < threshold for p in p_values]


def compare_conditions(
    df: pd.DataFrame, outcome: str, group_col: str,
    baseline: str, comparisons: list[str],
) -> pd.DataFrame:
    baseline_data = df[df[group_col] == baseline][outcome].values
    rows, p_vals = [], []
    for cond in comparisons:
        cond_data = df[df[group_col] == cond][outcome].values
        t, p = stats.ttest_ind(cond_data, baseline_data)
        ci_lo, ci_hi = bootstrap_ci(cond_data)
        rows.append({"condition": cond, "mean": cond_data.mean(),
                     "ci_low": ci_lo, "ci_high": ci_hi, "t": t, "p": p})
        p_vals.append(p)
    bonferroni_flags = bonferroni(p_vals)
    for r, sig in zip(rows, bonferroni_flags):
        r["significant_bonferroni"] = sig
    return pd.DataFrame(rows)
