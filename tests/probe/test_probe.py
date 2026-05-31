# tests/probe/test_probe.py
import numpy as np
import pytest


@pytest.mark.unit
def test_probe_perfect_signal():
    from src.probe.probe import train_probe
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(5, 0.1, (20, 10)), rng.normal(-5, 0.1, (20, 10))])
    y = np.array([1]*20 + [0]*20)
    result = train_probe(X, y, n_splits=3)
    assert result["auroc_mean"] > 0.95


@pytest.mark.unit
def test_probe_chance_signal():
    from src.probe.probe import train_probe
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (40, 10))
    y = np.array([1]*20 + [0]*20)
    result = train_probe(X, y, n_splits=3)
    assert result["auroc_mean"] < 0.75


@pytest.mark.unit
def test_best_layer():
    from src.probe.probe import best_layer
    results = {
        0: {"auroc_mean": 0.65, "auroc_std": 0.05, "best_estimator": None},
        5: {"auroc_mean": 0.82, "auroc_std": 0.03, "best_estimator": None},
        10: {"auroc_mean": 0.71, "auroc_std": 0.04, "best_estimator": None},
    }
    layer, auroc = best_layer(results)
    assert layer == 5 and auroc == 0.82
