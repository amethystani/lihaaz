# experiments/00_smoke_test.py
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Gemma-2-2B...")
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    print(f"Model loaded. Layers: {model.cfg.n_layers}")

    # l0=71 exists at layer 20 for 16k width; layer 12 has l0=41/82 only
    print("Loading Gemma Scope SAE at layer 20 (width_16k, average_l0_71)...")
    sae, cfg_dict, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_20/width_16k/average_l0_71",
    )
    sae = sae.to(device)
    print(f"SAE loaded. Feature dim: {sae.cfg.d_sae}")

    test_sentence = "aap kaise hain?"
    tokens = model.to_tokens(test_sentence)
    print(f"Tokens: {tokens.shape}")

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        resid = cache["resid_post", 20]
        feature_acts = sae.encode(resid[0])

    print(f"Residual stream shape: {resid.shape}")
    print(f"Feature activations shape: {feature_acts.shape}")
    top_features = feature_acts.max(0).values.topk(5)
    print(f"Top-5 feature indices: {top_features.indices.tolist()}")
    print(f"Top-5 feature max acts: {[round(v, 3) for v in top_features.values.tolist()]}")

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"VRAM allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
    print("Smoke test PASSED" if allocated < 22 else "WARNING: VRAM tight")


if __name__ == "__main__":
    main()
