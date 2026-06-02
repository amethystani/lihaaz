# paper/figures/fig2_alpha_sweep_pilot.py
# Pilot version: shows top-20 feature Cohen's d as a proxy for the alpha sweep.
# Replace with rubric_scores_full.csv-based fig2_alpha_sweep.py once full data exists.
import json, matplotlib.pyplot as plt, numpy as np


def main():
    res = json.load(open("experiments/results_sae_features.json"))
    top20 = res["top_features"][:20]
    indices = [f["feature_idx"] for f in top20]
    cohens  = [f["cohen_d"] for f in top20]
    sels    = [s["mean_selectivity"] for s in res["selectivity"][:20]]

    x = np.arange(len(indices))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width/2, cohens, width, label="Cohen's d", color="steelblue")
    ax.bar(x + width/2, sels,   width, label="Mean selectivity", color="darkorange")
    ax.axhline(0.5, color="red",   linestyle="--", linewidth=1, label="G2 d threshold (0.5)")
    ax.axhline(0.6, color="green", linestyle=":",  linewidth=1, label="G2 sel threshold (0.6)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in indices], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("SAE feature index (layer 5, top-20 by Cohen's d)")
    ax.set_ylabel("Score")
    ax.set_title(f"Top-20 SAE features — pilot n=30, layer {res['best_layer']}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("paper/figures/fig2_alpha_sweep.pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig2_alpha_sweep.png", dpi=200, bbox_inches="tight")
    print("fig2 saved")


if __name__ == "__main__":
    main()
