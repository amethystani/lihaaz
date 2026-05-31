# experiments/03_causal_triangle.py
import json, torch
from transformer_lens import HookedTransformer
from src.sae.loader import load_sae
from src.steering.additive import steer_generate
from src.steering.ablation import ablate_generate

MODEL_NAME  = "google/gemma-2-2b"
ALPHA       = 20.0  # initial; tune up/down if fluency degrades
# SET THESE from experiments/results_sae_features.json before running:
BEST_LAYER  = 12    # REPLACE
FEATURE_IDX = None  # REPLACE with top-1 feature index

AAP_PROMPTS = [
    "aap kaise hain? kya aapne khana kha liya?",
    "aap kahan se aa rahe hain? kya aap thake hue hain?",
    "aapka kaam kaisa chal raha hai?",
    "aap bahut acche gaate hain.",
    "aapse milkar bahut accha laga.",
]
TU_PROMPTS = [
    "tu kaisa hai? kya tune khana khaya?",
    "tu kahan se aa raha hai? kya tu thaka hua hai?",
    "tera kaam kaisa chal raha hai?",
    "tu bahut accha gaata hai.",
    "tujhse milkar accha laga.",
]

def main():
    assert FEATURE_IDX is not None, "Set FEATURE_IDX from Phase 3 results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    sae   = load_sae(layer=BEST_LAYER, model_size="2b", device=device)
    results = {}

    print("=== BASELINE ===")
    for i, p in enumerate(TU_PROMPTS):
        toks = model.to_tokens(p).to(device)
        out  = model.generate(toks, max_new_tokens=80, do_sample=False)
        text = model.to_string(out[0, toks.shape[1]:])
        results.setdefault("baseline", []).append({"prompt": p, "output": text})
        print(f"[{i}] {p}\n -> {text}\n")

    print("\n=== +alpha STEERING (tu -> aap) ===")
    for i, p in enumerate(TU_PROMPTS):
        text = steer_generate(model, sae, p, FEATURE_IDX, ALPHA, BEST_LAYER, device=device)
        results.setdefault("steering_pos", []).append({"prompt": p, "output": text, "alpha": ALPHA})
        print(f"[{i}] {p}\n -> {text}\n")

    print("\n=== -alpha ABLATION (suppress on aap prompts) ===")
    for i, p in enumerate(AAP_PROMPTS):
        text = ablate_generate(model, sae, p, FEATURE_IDX, ALPHA, BEST_LAYER, mode="negative", device=device)
        results.setdefault("ablation_neg", []).append({"prompt": p, "output": text})
        print(f"[{i}] {p}\n -> {text}\n")

    print("\n=== ZERO-ABLATION ===")
    for i, p in enumerate(AAP_PROMPTS):
        text = ablate_generate(model, sae, p, FEATURE_IDX, ALPHA, BEST_LAYER, mode="zero_ablation", device=device)
        results.setdefault("ablation_zero", []).append({"prompt": p, "output": text})
        print(f"[{i}] {p}\n -> {text}\n")

    json.dump(results, open("experiments/results_causal_triangle.json", "w"), ensure_ascii=False, indent=2)
    print("Saved to experiments/results_causal_triangle.json")
    print("\n[MANUAL] Score all outputs with the 6-axis rubric. Save to experiments/rubric_scores_mpu.csv")

if __name__ == "__main__":
    main()
