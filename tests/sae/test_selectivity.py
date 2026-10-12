# tests/sae/test_selectivity.py
import numpy as np
import pytest


@pytest.mark.unit
def test_auc_pair_passes_when_confound_random():
    from src.sae.selectivity import compute_auc_pair
    rng = np.random.default_rng(0)
    n = 40
    honorific = np.array([1]*20 + [0]*20)
    feature = np.where(honorific==1, 5.0, 0.0) + rng.normal(0, 0.1, n)
    confound = rng.integers(0, 2, n)
    result = compute_auc_pair(feature, honorific, confound)
    assert result["auc_honorific"] > 0.95
    assert result["auc_confound"] <= 0.55
    assert result["passes_specificity"]


@pytest.mark.unit
def test_auc_pair_fails_when_confound_matches():
    from src.sae.selectivity import compute_auc_pair
    rng = np.random.default_rng(0)
    n = 40
    labels = np.array([1]*20 + [0]*20)
    feature = np.where(labels==1, 5.0, 0.0) + rng.normal(0, 0.1, n)
    result = compute_auc_pair(feature, labels, labels.copy())
    assert result["auc_honorific"] > 0.95
    assert result["auc_confound"] > 0.95
    assert not result["passes_specificity"]


@pytest.mark.unit
def test_orientation_invariant_auc_counts_reverse_prediction():
    from src.sae.selectivity import orientation_invariant_auc
    labels = np.array([0, 0, 0, 1, 1, 1])
    reverse_feature = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    assert orientation_invariant_auc(reverse_feature, labels) == pytest.approx(1.0)
