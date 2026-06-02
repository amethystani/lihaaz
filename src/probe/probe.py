# src/probe/probe.py
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    pipe.fit(X, y)
    return {"auroc_mean": scores.mean(), "auroc_std": scores.std(), "best_estimator": pipe}


def run_layerwise_probes(acts_by_layer: dict[int, np.ndarray], labels: np.ndarray, C: float = 1.0) -> dict[int, dict]:
    return {layer: train_probe(X, labels, C=C) for layer, X in acts_by_layer.items()}


def best_layer(results: dict[int, dict]) -> tuple[int, float]:
    bl = max(results.items(), key=lambda kv: kv[1]["auroc_mean"])
    return bl[0], bl[1]["auroc_mean"]


def print_probe_report(results: dict[int, dict], condition: str = "raw") -> None:
    print(f"\n=== Probe AUROC by layer ({condition}) ===")
    bl, ba = best_layer(results)
    for layer in sorted(results):
        r = results[layer]
        marker = " <- BEST" if layer == bl else ""
        print(f"  Layer {layer:3d}: {r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}{marker}")
    print(f"Best layer: {bl}, AUROC: {ba:.4f}")
    print(f"G1 (>0.70): {'PASS' if ba > 0.70 else 'FAIL'}")
