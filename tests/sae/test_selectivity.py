# tests/sae/test_selectivity.py
import numpy as np
import pytest


@pytest.mark.unit
def test_selectivity_high_when_confound_random():
    from src.sae.selectivity import compute_selectivity
    rng = np.random.default_rng(0)
    n = 40
    honorific = np.array([1]*20 + [0]*20)
    feature = np.where(honorific==1, 5.0, 0.0) + rng.normal(0, 0.1, n)
    confound = rng.integers(0, 2, n)
    # A random confound's finite-sample AUROC scatters around 0.5, so selectivity
    # (1 - auroc_c/auroc_h) centers on 0.5 exactly. Assert clearly above the
    # matched-confound regime (0.0) rather than on the fragile 0.5 knife-edge.
    assert compute_selectivity(feature, honorific, confound) > 0.4


@pytest.mark.unit
def test_selectivity_low_when_confound_matches():
    from src.sae.selectivity import compute_selectivity
    rng = np.random.default_rng(0)
    n = 40
    labels = np.array([1]*20 + [0]*20)
    feature = np.where(labels==1, 5.0, 0.0) + rng.normal(0, 0.1, n)
    assert compute_selectivity(feature, labels, labels.copy()) < 0.5
