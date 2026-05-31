# tests/sae/test_features.py
import numpy as np
import pytest


@pytest.mark.unit
def test_cohen_d_perfect_separation():
    from src.sae.features import cohen_d
    g1 = np.array([10.0, 10.1, 9.9])
    g2 = np.array([0.0, 0.1, -0.1])
    assert cohen_d(g1, g2) > 5.0


@pytest.mark.unit
def test_cohen_d_identical():
    from src.sae.features import cohen_d
    g = np.array([1.0, 1.0, 1.0])
    assert cohen_d(g, g) == 0.0


@pytest.mark.unit
def test_rank_features_top_signal():
    from src.sae.features import rank_features_by_effect_size
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (20, 100))
    X[:10, 5] += 5.0
    labels = np.array([1]*10 + [0]*10)
    ranked = rank_features_by_effect_size(X, labels, top_k=10)
    assert len(ranked) == 10
    assert ranked[0]["feature_idx"] == 5
