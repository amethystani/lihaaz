# CLAUDE.md — AACL-IJCNLP 2026 SRW Project

## Project

**Title:** Where Does Politeness Live in Hindi? — A Mechanistic Interpretability Study of Honorific Encoding in Gemma Scope SAEs

**Venue target:** AACL-IJCNLP 2026 Student Research Workshop (SRW)

**Canonical proposal:** `PROBLEM_STATEMENT.md` (v4 — read this first)

## Hardware

NVIDIA RTX 4500 Ada (24GB VRAM). All experiment scripts must fit Gemma-2-9B (bf16) + Gemma Scope SAE within 24GB.

## Research plan (high level)

1. **MPU Pilot** (Week 1) — mechanistic probing of `aap`/`tum`/`tu` + verb-agreement contrastive pairs using Gemma Scope SAEs
2. **Causal validation** — activation steering + zero-ablation to confirm causal, not correlational, SAE features
3. **Cross-lingual comparison** — Japanese keigo as reference; test whether overlapping SAE features emerge
4. **Confound controls** — frequency, topic, formality, register decoupling (§8 of proposal)

## Key invariants

- Use `transformer-lens` for residual stream extraction (not raw HuggingFace hooks)
- Use `sae-lens` for Gemma Scope SAE loading
- All probes: logistic regression with L2 regularization (not MLP) to stay interpretable
- Feature selection: top-k by log-odds ratio, not raw activation magnitude

## Directory layout

```
src/          probe, SAE, steering, ablation scripts
data/         contrastive pairs + annotations (gitignored: raw/, processed/)
experiments/  pilot scripts and results
notebooks/    exploratory analysis
```

## Status

Pre-pilot. Start from §10 (MPU plan) in `PROBLEM_STATEMENT.md`.
