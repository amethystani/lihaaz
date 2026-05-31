# src/data/annotate.py
from __future__ import annotations
import numpy as np

def krippendorff_alpha(data: list[list], level: str = "ordinal") -> float:
    import krippendorff
    return krippendorff.alpha(reliability_data=data, level_of_measurement=level)

def report_iaa(ratings_by_annotator: list[list]) -> dict:
    alpha = krippendorff_alpha(ratings_by_annotator)
    print(f"Krippendorff's alpha = {alpha:.4f}")
    if alpha >= 0.80:
        status = "strong"
    elif alpha >= 0.67:
        status = "moderate"
    else:
        status = "weak — review disagreements before proceeding"
    print(f"  -> {status} agreement")
    return {"alpha": alpha, "status": status}
