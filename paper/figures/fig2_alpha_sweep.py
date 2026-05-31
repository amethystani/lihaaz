# paper/figures/fig2_alpha_sweep.py
import re
import matplotlib.pyplot as plt
import pandas as pd

STEER_PATTERN = re.compile(r"steer_feat(\d+)_alpha([\d.]+)")
PRIMARY_AXIS = "addressee"


def main():
    df = pd.read_csv("experiments/rubric_scores_full.csv")
    parsed = df["condition"].str.extract(STEER_PATTERN)
    df = df.assign(feature=parsed[0], alpha=parsed[1].astype(float))
    steer = df.dropna(subset=["feature", "alpha"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for feat, grp in steer.groupby("feature"):
        means = grp.groupby("alpha")[PRIMARY_AXIS].mean().sort_index()
        ax.plot(means.index, means.values, marker="o", label=f"Feature {feat}")
    ax.set_xlabel("Steering coefficient α")
    ax.set_ylabel(f"Mean {PRIMARY_AXIS} rubric score")
    ax.set_title("Steering dose–response across the α sweep")
    ax.legend(title="SAE feature")
    plt.tight_layout()
    plt.savefig("paper/figures/fig2_alpha_sweep.pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig2_alpha_sweep.png", dpi=200, bbox_inches="tight")
    print("fig2 saved")


if __name__ == "__main__":
    main()
