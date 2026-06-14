# experiments/03b_causal_sweep.py
# Corrected causal-triangle driver. Reads the top feature/layer from
# results_full_features.json, sweeps a sane (normalised) alpha, and records an
# OBJECTIVE honorific-marker proxy for each generation. The proxy is NOT a
# substitute for native-speaker rubric scoring (still required) but lets us
# quantify whether steering moves honorific marking at all, honestly.
import json, re
import torch
from transformer_lens import HookedTransformer
from src.sae.loader import load_sae
from src.steering.additive import steer_generate
from src.steering.ablation import ablate_generate

MODEL_NAME = "google/gemma-2-2b"
ALPHAS = [2.0, 4.0, 8.0, 16.0]          # normalised units (see additive.py)
ABLATE_ALPHA = 8.0

TU_PROMPTS = [
    "tu kaisa hai? kya tune khana khaya?",
    "tu kahan se aa raha hai? kya tu thaka hua hai?",
    "tera kaam kaisa chal raha hai?",
    "tu bahut accha gaata hai.",
    "tujhse milkar accha laga.",
]
AAP_PROMPTS = [
    "aap kaise hain? kya aapne khana kha liya?",
    "aap kahan se aa rahe hain? kya aap thake hue hain?",
    "aapka kaam kaisa chal raha hai?",
    "aap bahut acche gaate hain.",
    "aapse milkar bahut accha laga.",
]

# Surface honorific markers: aap-family pronouns + plural-respectful verb endings.
_AAP_TOKENS = {"aap", "aapko", "aapke", "aapka", "aapki", "aapse", "aapne", "hain",
               "kijiye", "kariye", "ji", "the", "rahe", "karte", "gaye", "aaiye", "baithiye"}
_TU_TOKENS = {"tu", "tujhe", "tera", "teri", "tujhse", "tune", "raha", "karta", "gaya", "kaisa"}


def hon_score(text: str) -> float:
    """+1 per aap-marker, -1 per tu-marker, normalised by token count. Range ~[-1,1].
    Higher = more honorific (aap-like). A crude automatic proxy only."""
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    if not toks:
        return 0.0
    a = sum(t in _AAP_TOKENS for t in toks)
    u = sum(t in _TU_TOKENS for t in toks)
    return round((a - u) / len(toks), 4)


def main():
    feat = json.load(open("experiments/results_full_features.json"))
    layer = int(feat["best_layer"])
    feature_idx = int(feat["top_features"][0]["feature_idx"])
    print(f"Using layer={layer}, top feature={feature_idx} "
          f"(d={feat['top_features'][0].get('cohen_d')})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    sae = load_sae(layer=layer, model_size="2b", device=device)
    results = {"layer": layer, "feature_idx": feature_idx,
               "baseline": [], "steering_sweep": [], "ablation_neg": [], "ablation_zero": []}

    print("=== BASELINE (tu prompts) ===")
    for p in TU_PROMPTS:
        toks = model.to_tokens(p).to(device)
        out = model.generate(toks, max_new_tokens=80, do_sample=False)
        txt = model.to_string(out[0, toks.shape[1]:])
        results["baseline"].append({"prompt": p, "output": txt, "hon": hon_score(txt)})

    print("=== STEERING SWEEP (tu -> aap, normalised alpha) ===")
    for a in ALPHAS:
        for p in TU_PROMPTS:
            txt = steer_generate(model, sae, p, feature_idx, a, layer, device=device)
            results["steering_sweep"].append({"prompt": p, "alpha": a, "output": txt, "hon": hon_score(txt)})

    print("=== ABLATION (suppress on aap prompts) ===")
    for p in AAP_PROMPTS:
        t_neg = ablate_generate(model, sae, p, feature_idx, ABLATE_ALPHA, layer, mode="negative", device=device)
        t_zero = ablate_generate(model, sae, p, feature_idx, ABLATE_ALPHA, layer, mode="zero_ablation", device=device)
        results["ablation_neg"].append({"prompt": p, "output": t_neg, "hon": hon_score(t_neg)})
        results["ablation_zero"].append({"prompt": p, "output": t_zero, "hon": hon_score(t_zero)})

    # Objective summary (proxy only)
    import statistics as st
    base = st.mean(r["hon"] for r in results["baseline"])
    print(f"\nBaseline mean hon (tu): {base:.3f}")
    for a in ALPHAS:
        m = st.mean(r["hon"] for r in results["steering_sweep"] if r["alpha"] == a)
        print(f"Steer alpha={a:>4}: mean hon = {m:.3f}  (delta vs baseline {m-base:+.3f})")
    abl = st.mean(r["hon"] for r in results["ablation_neg"])
    ablz = st.mean(r["hon"] for r in results["ablation_zero"])
    print(f"Ablation neg (aap prompts) mean hon: {abl:.3f}; zero-ablation: {ablz:.3f}")

    json.dump(results, open("experiments/results_causal_triangle_v2.json", "w"),
              ensure_ascii=False, indent=2)
    print("Saved results_causal_triangle_v2.json")
    print("[MANUAL] Native-speaker 6-axis rubric scoring still required before any causal claim.")


if __name__ == "__main__":
    main()
