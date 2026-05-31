# paper/figures/fig1_probe_auroc.py
import json, matplotlib.pyplot as plt, numpy as np


def main():
    hindi_res = json.load(open("experiments/results_full_probe_gemma2b.json"))
    layers = sorted(int(k) for k in hindi_res["raw"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (cond, label) in zip(axes, [("raw", "Raw"), ("pronoun_stripped", "Pronoun-Stripped")]):
        aurocs = [hindi_res[cond][str(l)]["auroc_mean"] for l in layers]
        stds   = [hindi_res[cond][str(l)]["auroc_std"]  for l in layers]
        ax.plot(layers, aurocs, color="steelblue", label=f"Hindi ({label})")
        ax.fill_between(layers, np.subtract(aurocs, stds), np.add(aurocs, stds), alpha=0.2, color="steelblue")
        ax.axhline(0.70, color="red", linestyle="--", label="H1 threshold (0.70)")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer"); ax.set_ylabel("AUROC"); ax.set_title(f"{label}"); ax.legend()
    plt.tight_layout()
    plt.savefig("paper/figures/fig1_probe_auroc.pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig1_probe_auroc.png", dpi=200, bbox_inches="tight")
    print("fig1 saved")


if __name__ == "__main__":
    main()
