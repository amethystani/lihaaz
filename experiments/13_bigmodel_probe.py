# experiments/13_bigmodel_probe.py
# Universal HF probe for models outside transformer-lens support and/or too
# large for 24 GB in bf16. Loads via AutoModelForCausalLM (bf16 or 4-bit nf4),
# with fallback to the architecture class from config (e.g. Mistral3ForConditionalGeneration).
# Extracts mean-pooled hidden states per layer with output_hidden_states, and
# runs the raw + pronoun-stripped grouped-CV probe. One model per process;
# merges its flat-curve summary into results_bigmodels.json.
#
# argv: <hf_name> <tag> <bf16|4bit>
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoTokenizer, BitsAndBytesConfig)

from src.data.pairs import load_dataset
from src.data.confounds import get_all_conditions
from src.probe.probe import train_probe

OUT = Path("experiments/results_bigmodels.json")
CONDITIONS = ["raw", "pronoun_stripped"]


def _auto_classes(hf_name):
    """Return model classes to try in order: CausalLM, any VLM Auto class available,
    then the concrete architecture class from config (handles Mistral3, etc.)."""
    classes = [AutoModelForCausalLM]
    for cls_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is not None:
            classes.append(cls)
    try:
        cfg = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
        arch = (cfg.architectures or [None])[0]
        if arch:
            arch_cls = getattr(transformers, arch, None)
            if arch_cls is not None and arch_cls not in classes:
                classes.append(arch_cls)
    except Exception:
        pass
    return classes


def load(hf_name, quant):
    tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    kw = dict(device_map="auto", low_cpu_mem_usage=True, output_hidden_states=True,
              torch_dtype=torch.bfloat16, trust_remote_code=True)
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    last_exc = None
    for cls in _auto_classes(hf_name):
        try:
            model = cls.from_pretrained(hf_name, **kw)
            model.eval()
            return model, tok
        except (ValueError, TypeError) as e:
            last_exc = e
            continue
    raise last_exc


def hidden_by_layer(model, tok, sentences):
    dev = next(model.parameters()).device
    per_layer = None
    for s in sentences:
        ids = tok(s, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            # Use keyword arg for input_ids — positional order differs across VLMs
            out = model(input_ids=ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple (n_layers+1) of [1,seq,d]
        if per_layer is None:
            per_layer = [[] for _ in hs]
        for l, h in enumerate(hs):
            per_layer[l].append(h[0].mean(0).float().cpu().numpy())
    return [np.stack(v) for v in per_layer]


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


def summarise(layers, labels, groups):
    aurocs = [train_probe(X, labels, groups=groups)["auroc_mean"] for X in layers]
    a = np.array(aurocs)
    return {"max": round(float(a.max()), 4), "min": round(float(a.min()), 4),
            "spread": round(float(a.max() - a.min()), 4), "n_layers": len(layers)}


def main():
    hf_name, tag, quant = sys.argv[1], sys.argv[2], sys.argv[3]
    pairs = load_dataset("data/hindi/full_600.json")
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    try:
        model, tok = load(hf_name, quant)
    except Exception as e:
        results[tag] = {"error": f"{type(e).__name__}: {str(e)[:160]}", "precision": quant}
        OUT.write_text(json.dumps(results, indent=2))
        print(f"SKIP {tag}: {type(e).__name__}: {str(e)[:160]}")
        return
    entry = {"precision": quant}
    for cond in CONDITIONS:
        sents, labels, groups = build(cond, pairs)
        layers = hidden_by_layer(model, tok, sents)
        entry[cond] = summarise(layers, labels, groups)
        print(f"[{tag}] {cond}: {entry[cond]}")
    results[tag] = entry
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[{tag}] saved ({quant})")


if __name__ == "__main__":
    main()
