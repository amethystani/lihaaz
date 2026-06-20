# experiments/15_layer_sae.py
"""Later-layer SAE feature analysis (layers 2, 12, 20).

Reviewer W3: the SAE negative ("no feature isolates honorifics from the
pronoun") currently rests on layer 2 alone -- the stripped-probe argmax, which
is exactly the most input-anchored place to look and so begs the question. This
script re-runs the Cohen's-d ranking + pronoun-token selectivity at an early,
middle, and late layer. If a mid/late-layer SAE *also* fails to clear the 0.6
selectivity bar, the negative is robust; if a later layer produces a clean
register feature, the SAE conclusion needs revising.

It also recomputes the near-duplicate signature (how many of the top-20 share
the same Cohen's d) at each layer, since the paper uses that at layer 2 as
evidence the dictionary tiles the pronoun direction.
"""
import json
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import ALL_PRONOUNS
from src.sae.loader import load_sae
from src.sae.features import get_feature_activations, rank_features_by_effect_size
from src.sae.selectivity import score_top_features

DATASET_PATH = "data/hindi/full_600.json"
MODEL_NAME = "google/gemma-2-2b"
LAYERS = [2, 12, 20]
SELECTIVITY_THRESHOLD = 0.6
NEAR_DUP_TOL = 0.012  # matches the paper's "d = 1.922 +/- 0.012" wording
OUT_PATH = Path("experiments/results_layer_sae.json")


def near_duplicate_count(top_features: list[dict], tol: float = NEAR_DUP_TOL) -> tuple[int, float]:
    """Largest cluster of top-20 features whose Cohen's d falls within +/-tol of
    a common centre, plus that centre. Mirrors the paper's 18-of-20 claim."""
    ds = np.array([f["cohen_d"] for f in top_features[:20]])
    best_n, best_centre = 0, 0.0
    for centre in ds:
        n = int(np.sum(np.abs(ds - centre) <= tol))
        if n > best_n:
            best_n, best_centre = n, float(centre)
    return best_n, best_centre


def main() -> None:
    device = "cuda"
    pairs = load_dataset(DATASET_PATH)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

    sentences, labels, pronoun_present = [], [], []
    for p in pairs:
        for s, lbl in [(p.high, 1), (p.low, 0)]:
            sentences.append(s)
            labels.append(lbl)
            pronoun_present.append(int(any(t.lower() in ALL_PRONOUNS for t in s.split())))
    labels = np.array(labels)
    confounds = {"pronoun_present": np.array(pronoun_present)}

    all_results: dict = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}

    for layer in LAYERS:
        key = str(layer)
        if key in all_results:
            print(f"[L{layer}] already done, skipping")
            continue
        print(f"\n=== Layer {layer}: loading SAE + encoding {len(sentences)} sentences ===")
        sae = load_sae(layer=layer, model_size="2b", device=device)
        feature_acts = get_feature_activations(model, sae, sentences, layer=layer, device=device)
        top50 = rank_features_by_effect_size(feature_acts, labels, top_k=50)
        sel = score_top_features(feature_acts, labels, confounds, [f["feature_idx"] for f in top50])

        sel_by_idx = {s["feature_idx"]: s["mean_selectivity"] for s in sel}
        for f in top50:
            f["selectivity"] = sel_by_idx.get(f["feature_idx"], 0.0)

        best_sel = max((f["selectivity"] for f in top50), default=0.0)
        n_clearing = int(sum(f["selectivity"] >= SELECTIVITY_THRESHOLD for f in top50))
        nd_count, nd_centre = near_duplicate_count(top50)

        all_results[key] = {
            "layer": layer,
            "best_selectivity": float(best_sel),
            "n_clearing_threshold": n_clearing,
            "threshold": SELECTIVITY_THRESHOLD,
            "near_duplicate_count": nd_count,
            "near_duplicate_centre_d": nd_centre,
            "top_features": top50[:20],
        }
        OUT_PATH.write_text(json.dumps(all_results, indent=2))
        print(f"[L{layer}] best_sel={best_sel:.3f}  clearing_0.6={n_clearing}  "
              f"near_dup={nd_count}/20 @ d={nd_centre:.3f}  → {OUT_PATH}")

    print("\n=== Summary ===")
    for key in sorted(all_results, key=int):
        r = all_results[key]
        verdict = "CLEARS 0.6" if r["n_clearing_threshold"] > 0 else "fails 0.6"
        print(f"  Layer {r['layer']:2d}: best_sel={r['best_selectivity']:.3f} ({verdict}), "
              f"near-dup {r['near_duplicate_count']}/20 @ d={r['near_duplicate_centre_d']:.3f}")


if __name__ == "__main__":
    main()
