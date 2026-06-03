# paper/figures/fig1_probe_auroc.py
import json, matplotlib.pyplot as plt, numpy as np

COLORS = {"raw": "steelblue", "pronoun_stripped": "darkorange"}
LABELS = {"raw": "Raw (n=600)", "pronoun_stripped": "Pronoun-Stripped (n=600)"}


def main():
    res = json.load(open("experiments/results_full_probe_gemma2b.json"))
    layers = sorted(int(k) for k in res["raw"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cond in zip(axes, ["raw", "pronoun_stripped"]):
        aurocs = [res[cond][str(l)]["auroc_mean"] for l in layers]
        stds   = [res[cond][str(l)]["auroc_std"]  for l in layers]
        color  = COLORS[cond]
        ax.plot(layers, aurocs, color=color, marker="o", markersize=4, label=LABELS[cond])
        ax.fill_between(
            layers,
            np.clip(np.subtract(aurocs, stds), 0, 1),
            np.clip(np.add(aurocs, stds), 0, 1),
            alpha=0.2, color=color,
        )
        ax.axhline(0.70, color="red",   linestyle="--", linewidth=1, label="H1 threshold (0.70)")
        ax.axhline(0.50, color="gray",  linestyle=":",  alpha=0.5)
        ax.axvline(1,    color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="$L^*=1$")
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC")
        ax.set_title(LABELS[cond])
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks(range(0, 26, 2))
        ax.legend(fontsize=8)
    plt.suptitle("Linear probe AUROC by layer — Gemma-2-2B (full 600-pair dataset)", fontsize=11)
    plt.tight_layout()
    plt.savefig("paper/figures/fig1_probe_auroc.pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig1_probe_auroc.png", dpi=200, bbox_inches="tight")
    print("fig1 saved")


if __name__ == "__main__":
    main()
