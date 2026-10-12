"""Corrected, orientation-invariant audit of the previously ranked SAE features.

For each raw-condition candidate from ``results_full_features.json``:

* ``A_hon`` is the feature's orientation-invariant AUC for honorific class on
  the pronoun-stripped minimal pairs.
* ``A_pron`` is its orientation-invariant AUC for raw (pronoun present) versus
  matched pronoun-stripped sentences.

This separates honorific discrimination after direct pronoun removal from
simple sensitivity to pronoun presence.  It replaces the invalid ratio score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.data.confounds import get_stripped_condition
from src.data.pairs import load_dataset
from src.sae.features import cohen_d
from src.sae.loader import load_sae
from src.sae.selectivity import orientation_invariant_auc


LAYER = 2
BATCH_SIZE = 16
SOURCE_RESULTS = Path("experiments/results_full_features.json")
OUTPUT_PATH = Path("experiments/results_corrected_feature_audit.json")


def selected_feature_activations(
    model: HookedTransformer,
    sae,
    sentences: list[str],
    feature_indices: list[int],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    pad_id = model.tokenizer.pad_token_id
    hook_name = f"blocks.{LAYER}.hook_resid_post"

    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start : start + BATCH_SIZE]
        tokens = model.to_tokens(batch, prepend_bos=True).to("cuda")
        mask = tokens.ne(pad_id)
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            residual = cache["resid_post", LAYER]
            feature_acts = sae.encode(residual).float()
            pooled = (feature_acts * mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1)
            selected = pooled[:, feature_indices].cpu().numpy()
        chunks.append(selected)
        del cache, residual, feature_acts, pooled, selected, tokens

    return np.concatenate(chunks, axis=0)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The corrected audit requires the project CUDA machine")

    prior = json.loads(SOURCE_RESULTS.read_text(encoding="utf-8"))
    candidates = prior["top_features"]
    feature_indices = [int(row["feature_idx"]) for row in candidates]

    pairs = load_dataset("data/hindi/full_600.json")
    raw_sentences: list[str] = []
    stripped_sentences: list[str] = []
    honorific_labels: list[int] = []
    for pair in pairs:
        stripped_high, stripped_low = get_stripped_condition(pair)
        raw_sentences.extend([pair.high, pair.low])
        stripped_sentences.extend([stripped_high, stripped_low])
        honorific_labels.extend([1, 0])

    labels = np.asarray(honorific_labels)
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device="cuda")
    sae = load_sae(layer=LAYER, model_size="2b", device="cuda")

    raw = selected_feature_activations(model, sae, raw_sentences, feature_indices)
    stripped = selected_feature_activations(
        model, sae, stripped_sentences, feature_indices
    )

    pronoun_values = np.concatenate([raw, stripped], axis=0)
    pronoun_labels = np.concatenate(
        [np.ones(len(raw), dtype=int), np.zeros(len(stripped), dtype=int)]
    )

    rows: list[dict] = []
    for column, candidate in enumerate(candidates):
        auc_honorific = orientation_invariant_auc(stripped[:, column], labels)
        auc_pronoun = orientation_invariant_auc(
            pronoun_values[:, column], pronoun_labels
        )
        stripped_d = cohen_d(
            stripped[labels == 1, column], stripped[labels == 0, column]
        )
        rows.append(
            {
                "raw_rank": column + 1,
                "feature_idx": feature_indices[column],
                "raw_cohen_d": float(candidate["cohen_d"]),
                "stripped_cohen_d_signed": float(stripped_d),
                "auc_honorific_stripped": auc_honorific,
                "auc_pronoun_presence": auc_pronoun,
                "passes_specificity": bool(
                    auc_honorific >= 0.80 and auc_pronoun <= 0.55
                ),
            }
        )

    result = {
        "layer": LAYER,
        "n_pairs": len(pairs),
        "n_candidates": len(rows),
        "criterion": {
            "auc_honorific_stripped_min": 0.80,
            "auc_pronoun_presence_max": 0.55,
        },
        "audit_definition": {
            "honorific": "aap-class vs tum/tu-class on pronoun-stripped pairs",
            "pronoun": "matched raw vs pronoun-stripped sentences",
            "auc_orientation": "max(AUC, 1-AUC)",
        },
        "features": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Passing candidates: {sum(row['passes_specificity'] for row in rows)}")
    for row in rows[:20]:
        print(
            f"rank={row['raw_rank']:2d} feature={row['feature_idx']:5d} "
            f"A_hon={row['auc_honorific_stripped']:.3f} "
            f"A_pron={row['auc_pronoun_presence']:.3f} "
            f"pass={row['passes_specificity']}"
        )


if __name__ == "__main__":
    main()
