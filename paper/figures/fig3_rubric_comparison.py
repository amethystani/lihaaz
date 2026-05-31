# paper/figures/fig3_rubric_comparison.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUBRIC_AXES = ["addressee", "verb_agreement", "register_consistency",
               "context_appropriateness", "fluency", "content_preservation"]
FAMILY_ORDER = ["baseline", "zero_shot", "four_shot", "steer", "ablate_neg", "ablate_zero"]


def condition_family(cond: str) -> str:
    if cond.startswith("steer_"):
        return "steer"
    if cond.startswith("ablate_neg"):
        return "ablate_neg"
    if cond.startswith("ablate_zero"):
        return "ablate_zero"
    return cond


def main():
    df = pd.read_csv("experiments/rubric_scores_full.csv")
    df = df.assign(family=df["condition"].map(condition_family))
    families = [f for f in FAMILY_ORDER if f in set(df["family"])]
    means = {f: [df[df["family"] == f][ax].mean() for ax in RUBRIC_AXES] for f in families}

    x = np.arange(len(RUBRIC_AXES))
    width = 0.8 / max(len(families), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, f in enumerate(families):
        ax.bar(x + i * width, means[f], width, label=f)
    ax.set_xticks(x + width * (len(families) - 1) / 2)
    ax.set_xticklabels(RUBRIC_AXES, rotation=30, ha="right")
    ax.set_ylabel("Mean rubric score")
    ax.set_title("Rubric scores by intervention family across evaluation axes")
    ax.legend(title="Condition family")
    plt.tight_layout()
    plt.savefig("paper/figures/fig3_rubric_comparison.pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig3_rubric_comparison.png", dpi=200, bbox_inches="tight")
    print("fig3 saved")


if __name__ == "__main__":
    main()
