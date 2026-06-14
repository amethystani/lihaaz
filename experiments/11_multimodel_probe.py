# experiments/11_multimodel_probe.py
# Cross-family / cross-scale generalisation of the decodability finding.
# Runs ONE model per invocation (argv: hf_name tag) and merges its raw +
# pronoun-stripped flat-curve summary into results_multimodel.json. Running one
# model per process isolates OOM: a crash on a big model cannot kill the others.
#
# Memory-safe load: bf16 + low_cpu_mem_usage via an explicit HF model handle,
# avoiding the fp32 CPU spike that OOMs 9B on a 62 GB host. Caches only
# resid_post so activations fit the 24 GB GPU.
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from src.data.pairs import load_dataset
from src.data.confounds import get_all_conditions
from src.probe.probe import train_probe

OUT = Path("experiments/results_multimodel.json")
CONDITIONS = ["raw", "pronoun_stripped"]


def load_model(hf_name, device):
    hf = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = HookedTransformer.from_pretrained(
        hf_name, hf_model=hf, tokenizer=tok, dtype="bfloat16", device=device)
    del hf
    gc.collect(); torch.cuda.empty_cache()
    return model


def extract_resid(model, sentences, device):
    n = model.cfg.n_layers
    names = {f"blocks.{l}.hook_resid_post" for l in range(n)}
    out = {l: [] for l in range(n)}
    for s in sentences:
        toks = model.to_tokens(s).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda x: x in names, return_type=None)
        for l in range(n):
            out[l].append(cache[f"blocks.{l}.hook_resid_post"][0].mean(0).cpu().float().numpy())
        del cache
    return {l: np.stack(v) for l, v in out.items()}


def build(condition, pairs):
    sents, labels, groups = [], [], []
    for idx, p in enumerate(pairs):
        conds = get_all_conditions(p)
        if condition not in conds:
            continue
        h, lo = conds[condition]
        if not h or not lo:
            continue
        sents += [h, lo]; labels += [1, 0]; groups += [idx, idx]
    return sents, np.array(labels), np.array(groups)


def summarise(acts, labels, groups):
    rows = {l: train_probe(acts[l], labels, groups=groups)["auroc_mean"] for l in acts}
    vals = np.array(list(rows.values()))
    return {"max": round(float(vals.max()), 4), "min": round(float(vals.min()), 4),
            "spread": round(float(vals.max() - vals.min()), 4),
            "argmax_layer": int(max(rows, key=rows.get)), "n_layers": len(rows)}


def main():
    hf_name, tag = sys.argv[1], sys.argv[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = load_dataset("data/hindi/full_600.json")
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    try:
        model = load_model(hf_name, device)
    except Exception as e:
        results[tag] = {"error": f"{type(e).__name__}: {str(e)[:140]}"}
        OUT.write_text(json.dumps(results, indent=2))
        print(f"SKIP {tag}: {type(e).__name__}: {str(e)[:140]}")
        return
    entry = {"n_layers": int(model.cfg.n_layers)}
    for cond in CONDITIONS:
        sents, labels, groups = build(cond, pairs)
        acts = extract_resid(model, sents, device)
        entry[cond] = summarise(acts, labels, groups)
        print(f"[{tag}] {cond}: {entry[cond]}")
    results[tag] = entry
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[{tag}] saved")


if __name__ == "__main__":
    main()
