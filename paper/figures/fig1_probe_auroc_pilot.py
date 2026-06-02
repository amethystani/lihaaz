# paper/figures/fig1_probe_auroc_pilot.py
# Generates fig1 from pilot (30-pair) probe results.
# Replace input files with results_full_probe_gemma2b.json once full data is available.
import json, matplotlib.pyplot as plt, numpy as np


def main():
    raw_res   = json.load(open("experiments/results_probe_raw.json"))
    strip_res = json.load(open("experiments/results_probe_pronoun_stripped.json"))

    layers = sorted(int(k) for k in raw_res)
    conditions = [
        (raw_res,   "Raw",              "steelblue"),
        (strip_res, "Pronoun-Stripped", "darkorange"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (res, label, color) in zip(axes, conditions):
        aurocs = [res[str(l)]["auroc_mean"] for l in layers]
        stds   = [res[str(l)]["auroc_std"]  for l in layers]
        ax.plot(layers, aurocs, color=color, marker="o", markersize=4, label=f"Hindi ({label})")
        ax.fill_between(layers,
                        np.clip(np.subtract(aurocs, stds), 0, 1),
                        np.clip(np.add(aurocs, stds), 0, 1),
                        alpha=0.2, color=color)
        ax.axhline(0.70, color="red",  linestyle="--", linewidth=1, label="H1 threshold (0.70)")
        ax.axhline(0.50, color="gray", linestyle=":",  alpha=0.5)
        ax.set_xlabel("Layer"); ax.set_ylabel("AUROC")
        ax.set_title(f"{label} (pilot, n=30)")
        ax.set_ylim(0.2, 1.0)
        ax.legend(fontsize=8)

    plt.suptitle("Linear probe AUROC by layer — Gemma-2-2B (pilot)", fontsize=11)
    plt.tight_layout()
    plt.savefig("paper/figures/fig1_probe_auroc.pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig1_probe_auroc.png", dpi=200, bbox_inches="tight")
    print("fig1 saved")


if __name__ == "__main__":
    main()
