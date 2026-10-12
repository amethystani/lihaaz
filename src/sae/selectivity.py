# src/sae/selectivity.py
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _auroc_single(col: np.ndarray, labels: np.ndarray) -> float:
    # A single-class confound (e.g. a control absent from this dataset) carries
    # no information to factor out; treat it as chance rather than crashing the
    # solver, and let score_top_features drop it from the mean.
    if col.std() == 0 or len(np.unique(labels)) < 2:
        return 0.5
    col_s = StandardScaler().fit_transform(col.reshape(-1, 1))
    clf = LogisticRegression(max_iter=200).fit(col_s, labels)
    return roc_auc_score(labels, clf.predict_proba(col_s)[:, 1])


def orientation_invariant_auc(feature_col: np.ndarray, labels: np.ndarray) -> float:
    """Return single-feature AUROC with direction removed.

    A value below 0.5 is still predictive in the opposite direction, so a
    confound audit must use ``max(AUC, 1 - AUC)`` rather than treating it as
    weak prediction.
    """
    auc = _auroc_single(feature_col, labels)
    return float(max(auc, 1.0 - auc))


def compute_auc_pair(
    feature_col: np.ndarray,
    honorific_labels: np.ndarray,
    confound_labels: np.ndarray,
    *,
    honorific_threshold: float = 0.80,
    confound_ceiling: float = 0.55,
) -> dict:
    """Report honorific and confound discrimination separately."""
    auc_honorific = orientation_invariant_auc(feature_col, honorific_labels)
    auc_confound = orientation_invariant_auc(feature_col, confound_labels)
    return {
        "auc_honorific": auc_honorific,
        "auc_confound": auc_confound,
        "passes_specificity": bool(
            auc_honorific >= honorific_threshold and auc_confound <= confound_ceiling
        ),
    }


def score_top_features(
    feature_acts: np.ndarray, honorific_labels: np.ndarray,
    confound_labels_dict: dict[str, np.ndarray], top_feature_indices: list[int],
) -> list[dict]:
    # Only confounds with >=2 classes present in this dataset are informative;
    # a control that is absent (all-zero labels) cannot be factored out and is
    # dropped rather than diluting mean selectivity with a spurious score.
    usable = {n: c for n, c in confound_labels_dict.items() if len(np.unique(c)) >= 2}
    results = []
    for f_idx in top_feature_indices:
        col = feature_acts[:, f_idx]
        row: dict = {
            "feature_idx": f_idx,
            "auc_honorific": orientation_invariant_auc(col, honorific_labels),
        }
        confound_passes = []
        for name, conf_labels in usable.items():
            auc_confound = orientation_invariant_auc(col, conf_labels)
            row[f"auc_{name}"] = auc_confound
            confound_passes.append(auc_confound <= 0.55)
        row["passes_specificity"] = bool(
            row["auc_honorific"] >= 0.80
            and confound_passes
            and all(confound_passes)
        )
        results.append(row)
    return results
