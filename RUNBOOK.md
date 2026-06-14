# Experiment Runbook — finishing the AACL SRW paper honestly

This is the exact path from the current (honest, partly-unrun) draft to a
submission-ready paper. Two things gate completion and **cannot** be faked:

1. **A GPU with `google/gemma-2-2b` + Gemma Scope SAEs.** The lab box is fine, but
   note its Qwen model is the *wrong* model — Gemma Scope SAEs exist only for
   Gemma-2-2B/9B. Download Gemma-2-2B onto the GPU; do not use Qwen for the SAE
   work (Qwen/Sarvam can be probe-only controls at most).
2. **At least one real second native-Hindi rater** to score generations on the
   6-axis rubric. Without this there is no honest Krippendorff's α and no honest
   rubric table. This is non-negotiable for the human-rated claims.

## 0. Environment (on the GPU box)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, transformer-lens, sae-lens, scikit-learn, evaluate
huggingface-cli login                     # Gemma is gated; accept the license once
python experiments/00_smoke_test.py       # verifies Gemma-2-2B + a Gemma Scope SAE load in 24 GB
```

## 1. Re-run the probe with the leakage fix (REQUIRED — numbers may change)
`src/probe/probe.py` and `experiments/04_full_probe.py` now use grouped CV
(pairs never split across folds). The old 0.89/0.77 numbers came from leaky
StratifiedKFold and **must be regenerated**.
```bash
python experiments/04_full_probe.py       # writes results_full_probe_gemma2b.json
```
Report the new raw + pronoun-stripped curves. Expect them to stay flat across
layers — that flatness is the paper's finding, not a bug.

## 2. Add the missing probe baselines/conditions
- **Length-only baseline** (fills Table 3): probe on sentence-length features alone.
- **Agreement-neutralised condition**: the current "pronoun-stripped" set still
  contains verb-agreement suffixes (`hain`/`ho`/`hai`), so it measures suffix
  detection, not pronoun-free honorific context. Add a condition that also
  neutralises agreement to separate the two (finding C2).
- **Script-variant / code-switch**: only build these rows if you actually
  generate the data (IndicTrans2). Otherwise delete those Table 3 rows — do not
  leave `\todo`.

## 3. Full-data SAE features at the justified layer
```bash
python experiments/05_full_features.py    # n=600, writes results_full_features.json
```
Report real Cohen's d AND selectivity. On current evidence the best selectivity
is ~0.29 < 0.6 → **no feature passes**. If that holds, the paper's RQ2 answer is
"no localized feature," which is the honest negative result.

## 4. Causal triangle with the corrected steering (only if a feature is worth it)
`additive.py` / `ablation.py` are fixed: unit-normalised direction, `alpha` in
residual-norm units, injection on generated positions only, and zero-ablation
that removes just feature f's contribution.
```bash
# edit experiments/03_causal_triangle.py: set BEST_LAYER and FEATURE_IDX from
# results_full_features.json; sweep ALPHA in {2,4,8,16} (NOT 20 raw).
python experiments/03_causal_triangle.py  # writes results_causal_triangle.json
```
Read the outputs. If steering still fails to shift honorifics at sane alpha,
report it as the negative result (that is publishable). Do not claim success
unless the generations actually become honorific Hindi.

## 5. Eval + human rubric + stats
```bash
python experiments/06_baselines.py        # zero/4-shot/LoRA baselines
# Have >=2 native raters score every generation on the 6 axes -> rubric_scores_full.csv
python experiments/08_stats.py            # mixed-effects (replace ttest, finding C7) + Krippendorff alpha
```
Regenerate `paper/figures/fig2`, `fig3` from the REAL csv (the committed PDFs are
from pilot stubs — do not ship them as-is).

## 6. Japanese comparison
```bash
python experiments/07_japanese.py         # writes results_japanese.json; fills Table 6
```

## 7. Before submitting — integrity sweep
```bash
grep -c 'todo{' paper/main.tex            # MUST be 0
```
Every remaining `\todo` must be either a real measured number or a deleted row.
Re-read the abstract/conclusion against the final tables: no claim without a
number behind it. Re-run the §8 novelty checklist in PROBLEM_STATEMENT.md.

---
**Recommended framing:** the negative/cautionary result (decodable but not
feature-localized, steering does not give clean control) is fully supported by
the data already in the repo and is a genuine first for Indic mech-interp. The
positive framing requires §4 steering to actually work at sane alpha, which the
current evidence does not support.
