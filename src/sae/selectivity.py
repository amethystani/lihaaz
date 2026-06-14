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


def compute_selectivity(feature_col: np.ndarray, honorific_labels: np.ndarray, confound_labels: np.ndarray) -> float:
    auroc_h = _auroc_single(feature_col, honorific_labels)
    auroc_c = _auroc_single(feature_col, confound_labels)
    return max(0.0, 1.0 - auroc_c / auroc_h) if auroc_h > 0.5 else 0.0


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
        row: dict = {"feature_idx": f_idx}
        for name, conf_labels in usable.items():
            row[f"sel_{name}"] = float(compute_selectivity(col, honorific_labels, conf_labels))
        sel_vals = [v for k, v in row.items() if k.startswith("sel_")]
        row["mean_selectivity"] = float(np.mean(sel_vals)) if sel_vals else 0.0
        results.append(row)
    return results
