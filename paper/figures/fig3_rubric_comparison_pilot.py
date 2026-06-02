# paper/figures/fig3_rubric_comparison_pilot.py
# Pilot version: shows word-level politeness register for baseline vs steering
# from results_causal_triangle.json (grepping for aap/tum/tu pronouns as proxy).
# Replace with rubric_scores_full.csv-based fig3 once full rubric scoring is done.
import json, matplotlib.pyplot as plt, numpy as np

AAP_MARKERS = {"aap", "aapne", "aapka", "aapki", "aapse", "aapko"}
TU_MARKERS  = {"tu", "tujhe", "tera", "tujhse", "tune"}
TUM_MARKERS = {"tum", "tumhe", "tumhara", "tumhari", "tumse"}


def politeness_score(text: str) -> float:
    words = set(text.lower().split())
    n_aap = len(words & AAP_MARKERS)
    n_low = len(words & (TU_MARKERS | TUM_MARKERS))
    total = n_aap + n_low
    return n_aap / total if total > 0 else 0.5


def main():
    d = json.load(open("experiments/results_causal_triangle.json"))

    sections = {
        "baseline":       d.get("baseline", []),
        "steering\n(+α)": d.get("steering_pos", []),
        "ablation\n(−α)": d.get("ablation_neg", []),
        "zero-ablation":  d.get("ablation_zero", []),
    }

    labels, means, stds = [], [], []
    for name, items in sections.items():
        scores = [politeness_score(it["output"]) for it in items]
        labels.append(name)
        means.append(float(np.mean(scores)) if scores else 0.0)
        stds.append(float(np.std(scores))   if scores else 0.0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, means, yerr=stds, capsize=4,
           color=["#4878cf", "#6acc65", "#d65f5f", "#b47cc7"])
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Politeness proxy score\n(fraction aap-family tokens)")
    ax.set_title("Intervention effect on surface politeness markers\n(pilot, feature 37, layer 5, α=20)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("paper/figures/fig3_rubric_comparison.pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig3_rubric_comparison.png", dpi=200, bbox_inches="tight")
    print("fig3 saved")


if __name__ == "__main__":
    main()
