# AACL-IJCNLP 2026 SRW — Hindi Honorific Mech-Interp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete, submission-ready AACL-IJCNLP 2026 SRW paper demonstrating that Gemma Scope SAEs contain identifiable, causally-manipulable features for Hindi `aap`/`tum`/`tu` honorific level.

**Architecture:** Four-stage pipeline — (1) probing to localize honorific signal layer-wise, (2) SAE feature discovery with confound controls, (3) full causal triangle (correlation + sufficiency + necessity), (4) Japanese comparison as descriptive control. Each stage produces releasable artifacts. A strict MPU gate at Phase 4 determines whether to proceed to the full study.

**Tech Stack:** Python 3.11, PyTorch 2.4+, transformer-lens, sae-lens, Gemma-2-2B/9B, Gemma Scope SAEs, scikit-learn, statsmodels, pandas, matplotlib, seaborn, BLEURT (via evaluate), indic-trans2 (transliteration).

---

## Go / No-Go Criteria (MPU Gate — end of Phase 4)

All four must be green to proceed to Phase 5+:

| # | Criterion | Threshold | Where evaluated |
|---|---|---|---|
| G1 | Pronoun-stripped probe AUROC | > 0.70 at some layer | Phase 2 |
| G2 | ≥ 1 SAE feature with effect size | d > 0.5 AND selectivity > 0.6 | Phase 3 |
| G3 | Additive steering rubric shift | p < 0.01, PPL inflation < 1.5× | Phase 4 |
| G4 | Ablation reduces honorificity | measurable drop (any p < 0.1) | Phase 4 |

**Pivot on failure:** "Hindi honorific representations in current public SAEs are distributed and resist feature-localized intervention" — still SRW-publishable.

---

## File Structure

```
AACL-IJCNLP/
├── src/
│   ├── data/
│   │   ├── pairs.py          # dataset class, loading, train/test split
│   │   ├── confounds.py      # pronoun-stripping, length-matching, script variants
│   │   └── annotate.py       # annotation schema + inter-annotator stats
│   ├── probe/
│   │   ├── extract.py        # residual stream extraction (transformer-lens)
│   │   └── probe.py          # layer-wise logistic probe, AUROC reporting
│   ├── sae/
│   │   ├── loader.py         # Gemma Scope SAE load via sae-lens
│   │   ├── features.py       # feature activation extraction + ranking
│   │   └── selectivity.py    # selectivity score vs. confound dimensions
│   ├── steering/
│   │   ├── additive.py       # +alpha * decoder_f injection
│   │   └── ablation.py       # -alpha steering + encode-side zero-ablation
│   ├── eval/
│   │   ├── rubric.py         # rubric schema, per-generation scoring interface
│   │   ├── fluency.py        # perplexity measurement
│   │   └── content.py        # BLEURT content preservation
│   ├── baselines/
│   │   ├── prompting.py      # zero-shot + 4-shot prompting baselines
│   │   ├── lora.py           # LoRA fine-tuning baseline
│   │   └── fudge.py          # FUDGE-style classifier-guided decoding
│   ├── japanese/
│   │   └── pipeline.py       # probe + feature ID pipeline reused for Japanese
│   └── stats/
│       └── analysis.py       # mixed-effects models, Bonferroni, bootstrap CIs
├── data/
│   ├── hindi/
│   │   ├── pilot_30.json     # 30-pair MPU pilot set
│   │   └── full_600.json     # final 600-pair dataset (CC-BY-SA)
│   └── japanese/
│       └── keigo_200.json    # 200-pair Japanese comparison set
├── experiments/
│   ├── 01_probe_pilot.py     # Phase 2 MPU probe run
│   ├── 02_sae_features.py    # Phase 3 feature discovery
│   ├── 03_causal_triangle.py # Phase 4 causal triangle
│   ├── 04_full_probe.py      # Phase 6 full probing
│   ├── 05_full_features.py   # Phase 7 full feature study
│   ├── 06_baselines.py       # Phase 8 baselines
│   ├── 07_japanese.py        # Phase 9 Japanese comparison
│   └── 08_stats.py           # Phase 10 statistical analysis
├── notebooks/
│   └── exploration.ipynb     # scratch analysis
└── paper/
    └── figures/              # matplotlib outputs for paper
```

---

## Phase 0: Environment Setup + Smoke Test

**Goal:** Working environment; verify GPU headroom; reproduce one steering result from Liu et al. (2025).

**Estimated time:** 1 day

### Task 0.1: Install dependencies

**Files:**
- Already exists: `requirements.txt`
- Create: `.venv/` (not tracked in git)

- [ ] **Step 1: Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- [ ] **Step 2: Verify GPU availability**

```bash
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"
```

Expected output:
```
CUDA: True
Device: NVIDIA RTX 4500 Ada Generation
VRAM: 24 GB
```

- [ ] **Step 3: Commit venv-exclusion confirmation**

```bash
git add requirements.txt
git commit -m "chore: verify requirements.txt installs cleanly"
```

---

### Task 0.2: Load Gemma-2-2B + Gemma Scope SAE smoke test

**Files:**
- Create: `experiments/00_smoke_test.py`

- [ ] **Step 1: Write smoke test script**

```python
# experiments/00_smoke_test.py
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Gemma-2-2B...")
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    print(f"Model loaded. Layers: {model.cfg.n_layers}")

    print("Loading Gemma Scope SAE at layer 12...")
    sae, cfg_dict, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_71",
    )
    sae = sae.to(device)
    print(f"SAE loaded. Feature dim: {sae.cfg.d_sae}")

    test_sentence = "aap kaise hain?"  # fallback ASCII; replace with Devanagari if tokenizer supports
    tokens = model.to_tokens(test_sentence)
    print(f"Tokens: {tokens.shape}")

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        resid = cache["resid_post", 12]
        feature_acts = sae.encode(resid[0])

    print(f"Residual stream shape: {resid.shape}")
    print(f"Feature activations shape: {feature_acts.shape}")
    top_features = feature_acts.max(0).values.topk(5)
    print(f"Top-5 feature indices: {top_features.indices.tolist()}")
    print(f"Top-5 feature max acts: {[round(v, 3) for v in top_features.values.tolist()]}")

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved() / 1024**3
    print(f"VRAM allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
    print("Smoke test PASSED" if allocated < 22 else "WARNING: VRAM tight")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test**

```bash
python experiments/00_smoke_test.py
```

Expected: prints shapes, top features, VRAM < 22 GB allocated, ends with "Smoke test PASSED".

- [ ] **Step 3: Commit**

```bash
git add experiments/00_smoke_test.py
git commit -m "feat: smoke test for Gemma-2-2B + Gemma Scope SAE"
```

---

## Phase 1: Hindi MPU Pilot Dataset — 30 Pairs

**Goal:** 30 minimally-contrastive Hindi honorific pairs, structured as JSON with per-item confound labels, validated by self (native speaker).

**Estimated time:** 1 day

### Task 1.1: Define data schema

**Files:**
- Create: `src/data/pairs.py`
- Create: `tests/data/test_pairs.py`

- [ ] **Step 1: Write failing test first**

```python
# tests/data/test_pairs.py
import json, pytest
from pathlib import Path

def test_pair_roundtrip(tmp_path):
    # This import will fail until pairs.py exists — RED
    from src.data.pairs import HonorificPair, load_dataset, save_dataset
    pair = HonorificPair(
        id="h001", high="aap kaise hain?", low="tum kaise ho?",
        high_level="aap", low_level="tum", domain="social",
        confounds={"length_ratio": 1.0},
    )
    path = tmp_path / "test.json"
    save_dataset([pair], path)
    loaded = load_dataset(path)
    assert len(loaded) == 1
    assert loaded[0].id == "h001"
    assert loaded[0].confounds["length_ratio"] == 1.0

def test_all_required_fields():
    from src.data.pairs import HonorificPair
    d = {
        "id": "h002", "high": "A", "low": "B",
        "high_level": "aap", "low_level": "tu",
        "domain": "family", "confounds": {}, "notes": "",
    }
    p = HonorificPair.from_dict(d)
    assert p.low_level == "tu"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/data/test_pairs.py -v
```

Expected: `ModuleNotFoundError` — RED (correct).

- [ ] **Step 3: Write implementation**

```python
# src/data/pairs.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import json
from pathlib import Path

HonorificLevel = Literal["aap", "tum", "tu"]

@dataclass
class HonorificPair:
    id: str
    high: str
    low: str
    high_level: HonorificLevel
    low_level: HonorificLevel
    domain: str
    confounds: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "high": self.high, "low": self.low,
            "high_level": self.high_level, "low_level": self.low_level,
            "domain": self.domain, "confounds": self.confounds, "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HonorificPair":
        return cls(**d)

def load_dataset(path: str | Path) -> list[HonorificPair]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [HonorificPair.from_dict(item) for item in data]

def save_dataset(pairs: list[HonorificPair], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([p.to_dict() for p in pairs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/data/test_pairs.py -v
```

Expected: 2 tests PASS — GREEN.

- [ ] **Step 5: Commit**

```bash
git add src/data/pairs.py tests/data/test_pairs.py
git commit -m "feat: HonorificPair data schema with roundtrip tests"
```

---

### Task 1.2: Draft 30 pilot pairs (manual annotation)

**Files:**
- Create: `experiments/draft_pilot.py`
- Create: `data/hindi/pilot_30.json`

**Distribution:** 10 `aap` vs `tum`, 10 `aap` vs `tu`, 10 `tum` vs `tu`. Domains: 8 social, 8 professional, 8 service, 6 family.

- [ ] **Step 1: Write the draft scaffold script**

```python
# experiments/draft_pilot.py
"""Run to generate pilot_30.json. Edit PAIRS list directly."""
from src.data.pairs import HonorificPair, save_dataset

PAIRS = [
    # aap vs tum (10 pairs)
    HonorificPair(id="h001", domain="social",
        high="aap kahan ja rahe hain?", low="tum kahan ja rahe ho?",
        high_level="aap", low_level="tum",
        confounds={"pronoun_stripped_high": "kahan ja rahe hain?",
                   "pronoun_stripped_low": "kahan ja rahe ho?", "length_ratio": 1.0}),
    HonorificPair(id="h002", domain="professional",
        high="aap baithiye, main abhi aata hun.",
        low="tum baitho, main abhi aata hun.",
        high_level="aap", low_level="tum", confounds={}),
    HonorificPair(id="h003", domain="service",
        high="aapka order taiyar hai.", low="tumhara order taiyar hai.",
        high_level="aap", low_level="tum", confounds={"length_ratio": 1.05}),
    HonorificPair(id="h004", domain="social",
        high="kya aap chai lenge?", low="kya tum chai loge?",
        high_level="aap", low_level="tum", confounds={}),
    HonorificPair(id="h005", domain="professional",
        high="aap is form ko bhar dijiye.", low="tum is form ko bhar do.",
        high_level="aap", low_level="tum", confounds={}),
    HonorificPair(id="h006", domain="family",
        high="aap kab aa rahe hain?", low="tum kab aa rahe ho?",
        high_level="aap", low_level="tum", confounds={}),
    HonorificPair(id="h007", domain="social",
        high="aapne bahut accha kaam kiya.", low="tumne bahut accha kaam kiya.",
        high_level="aap", low_level="tum", confounds={}),
    HonorificPair(id="h008", domain="service",
        high="aap andar aa jaiye.", low="tum andar aa jao.",
        high_level="aap", low_level="tum", confounds={}),
    HonorificPair(id="h009", domain="professional",
        high="aap mujhe bataiye, main madad kar sakta hun.",
        low="tum mujhe batao, main madad kar sakta hun.",
        high_level="aap", low_level="tum", confounds={}),
    HonorificPair(id="h010", domain="social",
        high="aap yahan rehte hain?", low="tum yahan rehte ho?",
        high_level="aap", low_level="tum", confounds={}),

    # aap vs tu (10 pairs)
    HonorificPair(id="h011", domain="family",
        high="aap theek hain?", low="tu theek hai?",
        high_level="aap", low_level="tu",
        confounds={"pronoun_stripped_high": "theek hain?",
                   "pronoun_stripped_low": "theek hai?", "length_ratio": 0.95}),
    HonorificPair(id="h012", domain="social",
        high="aap kahan se hain?", low="tu kahan se hai?",
        high_level="aap", low_level="tu", confounds={}),
    HonorificPair(id="h013", domain="professional",
        high="aapka naam kya hai?", low="tera naam kya hai?",
        high_level="aap", low_level="tu", confounds={}),
    HonorificPair(id="h014", domain="family",
        high="aap khana kha lijiye.", low="tu khana kha le.",
        high_level="aap", low_level="tu", confounds={}),
    HonorificPair(id="h015", domain="social",
        high="aap bahut acche gaate hain.", low="tu bahut accha gaata hai.",
        high_level="aap", low_level="tu", confounds={}),
    HonorificPair(id="h016", domain="service",
        high="aap apna parcha dikhiye.", low="tu apna parcha dikha.",
        high_level="aap", low_level="tu", confounds={}),
    HonorificPair(id="h017", domain="professional",
        high="aap yahan baith sakte hain.", low="tu yahan baith sakta hai.",
        high_level="aap", low_level="tu", confounds={}),
    HonorificPair(id="h018", domain="social",
        high="aap sahi keh rahe hain.", low="tu sahi keh raha hai.",
        high_level="aap", low_level="tu", confounds={}),
    HonorificPair(id="h019", domain="family",
        high="aapne mujhe bulaya?", low="tune mujhe bulaya?",
        high_level="aap", low_level="tu", confounds={}),
    HonorificPair(id="h020", domain="service",
        high="aap mere saath chaliye.", low="tu mere saath chal.",
        high_level="aap", low_level="tu", confounds={}),

    # tum vs tu (10 pairs)
    HonorificPair(id="h021", domain="social",
        high="tum kahan ja rahe ho?", low="tu kahan ja raha hai?",
        high_level="tum", low_level="tu",
        confounds={"pronoun_stripped_high": "kahan ja rahe ho?",
                   "pronoun_stripped_low": "kahan ja raha hai?", "length_ratio": 1.0}),
    HonorificPair(id="h022", domain="family",
        high="tum theek ho?", low="tu theek hai?",
        high_level="tum", low_level="tu", confounds={}),
    HonorificPair(id="h023", domain="social",
        high="tumne yeh kab seekha?", low="tune yeh kab seekha?",
        high_level="tum", low_level="tu", confounds={}),
    HonorificPair(id="h024", domain="family",
        high="tum khana khaoge?", low="tu khana khayega?",
        high_level="tum", low_level="tu", confounds={}),
    HonorificPair(id="h025", domain="social",
        high="tum bahut samajhdar ho.", low="tu bahut samajhdar hai.",
        high_level="tum", low_level="tu", confounds={}),
    HonorificPair(id="h026", domain="professional",
        high="tum yeh kaam kar sakte ho.", low="tu yeh kaam kar sakta hai.",
        high_level="tum", low_level="tu", confounds={}),
    HonorificPair(id="h027", domain="family",
        high="tum kab aaye?", low="tu kab aaya?",
        high_level="tum", low_level="tu", confounds={}),
    HonorificPair(id="h028", domain="social",
        high="tumhe yeh pasand hai?", low="tujhe yeh pasand hai?",
        high_level="tum", low_level="tu", confounds={}),
    HonorificPair(id="h029", domain="professional",
        high="tum sahi jagah aaye ho.", low="tu sahi jagah aaya hai.",
        high_level="tum", low_level="tu", confounds={}),
    HonorificPair(id="h030", domain="service",
        high="tumhara kaam ho gaya.", low="tera kaam ho gaya.",
        high_level="tum", low_level="tu", confounds={}),
]

if __name__ == "__main__":
    save_dataset(PAIRS, "data/hindi/pilot_30.json")
    aap_tum = sum(1 for p in PAIRS if p.high_level=="aap" and p.low_level=="tum")
    aap_tu  = sum(1 for p in PAIRS if p.high_level=="aap" and p.low_level=="tu")
    tum_tu  = sum(1 for p in PAIRS if p.high_level=="tum" and p.low_level=="tu")
    print(f"Saved {len(PAIRS)} pairs: aap-tum={aap_tum}, aap-tu={aap_tu}, tum-tu={tum_tu}")
```

- [ ] **Step 2: Run draft script**

```bash
python experiments/draft_pilot.py
```

Expected: `Saved 30 pairs: aap-tum=10, aap-tu=10, tum-tu=10`

Note: The pilot dataset uses romanized Hindi. Before the full study (Phase 5), replace with Devanagari script for all production runs.

- [ ] **Step 3: Manual review pass (30 min as native speaker)**

Read `data/hindi/pilot_30.json`. Verify each pair: same propositional content, only pronoun + verb agreement differs. Mark any pairs needing correction.

- [ ] **Step 4: Commit**

```bash
git add experiments/draft_pilot.py data/hindi/pilot_30.json
git commit -m "feat: 30-pair Hindi MPU pilot dataset"
```

---

### Task 1.3: Confound stripping utilities

**Files:**
- Create: `src/data/confounds.py`
- Create: `tests/data/test_confounds.py`

- [ ] **Step 1: Write failing tests first**

```python
# tests/data/test_confounds.py
def test_strip_pronouns_aap():
    from src.data.confounds import strip_pronouns
    result = strip_pronouns("aap kahan ja rahe hain?")
    assert "aap" not in result
    assert "kahan" in result

def test_strip_pronouns_no_pronoun():
    from src.data.confounds import strip_pronouns
    result = strip_pronouns("yeh kitaab acchi hai.")
    assert result == "yeh kitaab acchi hai."

def test_get_stripped_uses_annotation():
    from src.data.pairs import HonorificPair
    from src.data.confounds import get_stripped_condition
    pair = HonorificPair(id="x", high="A", low="B", high_level="aap", low_level="tum",
                         domain="social",
                         confounds={"pronoun_stripped_high": "X", "pronoun_stripped_low": "Y"})
    h, l = get_stripped_condition(pair)
    assert h == "X" and l == "Y"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/data/test_confounds.py -v
```

Expected: `ModuleNotFoundError` — RED.

- [ ] **Step 3: Write implementation**

```python
# src/data/confounds.py
from __future__ import annotations
from src.data.pairs import HonorificPair

# Romanized + Devanagari pronoun sets
AAP_TOKENS  = {"aap", "aapka", "aapki", "aapke", "aapko", "aapse", "aapne",
               "aap", "aapka", "aapki", "aapke"}
TUM_TOKENS  = {"tum", "tumhara", "tumhari", "tumhare", "tumhen", "tumne", "tumse"}
TU_TOKENS   = {"tu", "tera", "teri", "tere", "tujhe", "tune", "tujhse"}
ALL_PRONOUNS = AAP_TOKENS | TUM_TOKENS | TU_TOKENS


def strip_pronouns(text: str) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if t.lower() not in ALL_PRONOUNS)


def get_stripped_condition(pair: HonorificPair) -> tuple[str, str]:
    if "pronoun_stripped_high" in pair.confounds:
        return pair.confounds["pronoun_stripped_high"], pair.confounds["pronoun_stripped_low"]
    return strip_pronouns(pair.high), strip_pronouns(pair.low)


def get_all_conditions(pair: HonorificPair) -> dict[str, tuple[str, str]]:
    raw     = (pair.high, pair.low)
    stripped = get_stripped_condition(pair)
    conditions: dict[str, tuple[str, str]] = {"raw": raw, "pronoun_stripped": stripped}
    if pair.confounds.get("script_roman_high"):
        conditions["roman_script"] = (pair.confounds["script_roman_high"],
                                      pair.confounds["script_roman_low"])
    if pair.confounds.get("code_switched_high"):
        conditions["code_switched"] = (pair.confounds["code_switched_high"],
                                       pair.confounds["code_switched_low"])
    return conditions
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/data/test_confounds.py -v
```

Expected: 3 tests PASS — GREEN.

- [ ] **Step 5: Commit**

```bash
git add src/data/confounds.py tests/data/test_confounds.py
git commit -m "feat: pronoun-stripping confound utilities"
```

---

## Phase 2: MPU Probe Pipeline

**Goal:** Layer-wise linear probe on the 30-pair pilot; report raw and pronoun-stripped AUROC. **MPU G1: stripped AUROC > 0.70.**

**Estimated time:** 1–2 days

### Task 2.1: Residual stream extraction

**Files:**
- Create: `src/probe/extract.py`
- Create: `tests/probe/test_extract.py`

- [ ] **Step 1: Write failing test**

```python
# tests/probe/test_extract.py
import numpy as np

def test_extract_output_shape():
    # Structural shape contract — no GPU required
    d_model, n_sentences, n_layers = 8, 3, 2
    result = {0: np.zeros((n_sentences, d_model)), 1: np.zeros((n_sentences, d_model))}
    for l, arr in result.items():
        assert arr.shape == (n_sentences, d_model)
    assert set(result.keys()) == {0, 1}
```

- [ ] **Step 2: Run — PASS immediately (structural test)**

```bash
python -m pytest tests/probe/test_extract.py -v
```

- [ ] **Step 3: Write extraction module**

```python
# src/probe/extract.py
from __future__ import annotations
import torch
import numpy as np
from transformer_lens import HookedTransformer


def extract_residuals(
    model: HookedTransformer,
    sentences: list[str],
    layers: list[int] | None = None,
    pooling: str = "mean",
    device: str = "cuda",
) -> dict[int, np.ndarray]:
    """Returns dict: layer_idx -> np.ndarray of shape (n_sentences, d_model)."""
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    all_acts: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    for sentence in sentences:
        tokens = model.to_tokens(sentence).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for l in layers:
            resid = cache["resid_post", l]  # (1, seq_len, d_model)
            if pooling == "mean":
                vec = resid[0].mean(0).cpu().float().numpy()
            elif pooling == "last":
                vec = resid[0, -1].cpu().float().numpy()
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            all_acts[l].append(vec)

    return {l: np.stack(all_acts[l]) for l in layers}
```

- [ ] **Step 4: Commit**

```bash
git add src/probe/extract.py tests/probe/test_extract.py
git commit -m "feat: residual stream extraction module"
```

---

### Task 2.2: Linear probe training and AUROC reporting

**Files:**
- Create: `src/probe/probe.py`
- Create: `tests/probe/test_probe.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/probe/test_probe.py
import numpy as np

def test_probe_perfect_signal():
    from src.probe.probe import train_probe
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(5, 0.1, (20, 10)), rng.normal(-5, 0.1, (20, 10))])
    y = np.array([1]*20 + [0]*20)
    result = train_probe(X, y, n_splits=3)
    assert result["auroc_mean"] > 0.95

def test_probe_chance_signal():
    from src.probe.probe import train_probe
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (40, 10))
    y = np.array([1]*20 + [0]*20)
    result = train_probe(X, y, n_splits=3)
    assert result["auroc_mean"] < 0.75

def test_best_layer():
    from src.probe.probe import best_layer
    results = {
        0: {"auroc_mean": 0.65, "auroc_std": 0.05, "best_estimator": None},
        5: {"auroc_mean": 0.82, "auroc_std": 0.03, "best_estimator": None},
        10: {"auroc_mean": 0.71, "auroc_std": 0.04, "best_estimator": None},
    }
    layer, auroc = best_layer(results)
    assert layer == 5 and auroc == 0.82
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/probe/test_probe.py -v
```

Expected: `ModuleNotFoundError` — RED.

- [ ] **Step 3: Write implementation**

```python
# src/probe/probe.py
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty="l2", max_iter=1000, solver="lbfgs")),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    pipe.fit(X, y)
    return {"auroc_mean": scores.mean(), "auroc_std": scores.std(), "best_estimator": pipe}


def run_layerwise_probes(acts_by_layer: dict[int, np.ndarray], labels: np.ndarray, C: float = 1.0) -> dict[int, dict]:
    return {layer: train_probe(X, labels, C=C) for layer, X in acts_by_layer.items()}


def best_layer(results: dict[int, dict]) -> tuple[int, float]:
    bl = max(results.items(), key=lambda kv: kv[1]["auroc_mean"])
    return bl[0], bl[1]["auroc_mean"]


def print_probe_report(results: dict[int, dict], condition: str = "raw") -> None:
    print(f"\n=== Probe AUROC by layer ({condition}) ===")
    bl, ba = best_layer(results)
    for layer in sorted(results):
        r = results[layer]
        marker = " <- BEST" if layer == bl else ""
        print(f"  Layer {layer:3d}: {r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}{marker}")
    print(f"Best layer: {bl}, AUROC: {ba:.4f}")
    print(f"G1 (>0.70): {'PASS' if ba > 0.70 else 'FAIL'}")
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/probe/test_probe.py -v
```

Expected: 3 tests PASS — GREEN.

- [ ] **Step 5: Commit**

```bash
git add src/probe/probe.py tests/probe/test_probe.py
git commit -m "feat: layer-wise logistic probe with AUROC and G1 gate"
```

---

### Task 2.3: Run MPU probe experiment

**Files:**
- Create: `experiments/01_probe_pilot.py`

- [ ] **Step 1: Write experiment script**

```python
# experiments/01_probe_pilot.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_stripped_condition
from src.probe.extract import extract_residuals
from src.probe.probe import run_layerwise_probes, print_probe_report

DATASET_PATH = "data/hindi/pilot_30.json"
MODEL_NAME   = "google/gemma-2-2b"
LAYERS       = list(range(26))

def build_xy(pairs, condition="raw"):
    sentences, labels = [], []
    for p in pairs:
        if condition == "raw":
            sentences.extend([p.high, p.low])
        else:
            h, l = get_stripped_condition(p)
            sentences.extend([h, l])
        labels.extend([1, 0])
    return sentences, np.array(labels)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = load_dataset(DATASET_PATH)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

    for condition in ["raw", "pronoun_stripped"]:
        sentences, labels = build_xy(pairs, condition)
        print(f"\nExtracting residuals: {condition} ({len(sentences)} sentences)")
        acts = extract_residuals(model, sentences, layers=LAYERS, device=device)
        results = run_layerwise_probes(acts, labels)
        print_probe_report(results, condition=condition)
        out = {str(l): {"auroc_mean": v["auroc_mean"], "auroc_std": v["auroc_std"]}
               for l, v in results.items()}
        json.dump(out, open(f"experiments/results_probe_{condition}.json", "w"), indent=2)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run experiment**

```bash
python experiments/01_probe_pilot.py 2>&1 | tee experiments/probe_pilot_log.txt
```

Expected: AUROC tables for both conditions; G1 PASS or FAIL printed.

- [ ] **Step 3: Commit**

```bash
git add experiments/01_probe_pilot.py experiments/results_probe_*.json experiments/probe_pilot_log.txt
git commit -m "feat: MPU probe pilot — layer-wise AUROC on 30 pairs"
```

---

## Phase 3: MPU SAE Feature Discovery

**Goal:** At best probe layer, run Gemma Scope SAE; rank features by effect size + selectivity. **MPU G2: d > 0.5 AND selectivity > 0.6.**

**Estimated time:** 1–2 days

### Task 3.1: SAE loader

**Files:**
- Create: `src/sae/loader.py`

- [ ] **Step 1: Write SAE loader**

```python
# src/sae/loader.py
from __future__ import annotations
import torch
from sae_lens import SAE

GEMMA_2B_RELEASE = "gemma-scope-2b-pt-res"
GEMMA_9B_RELEASE = "gemma-scope-9b-pt-res"

def load_sae(layer: int, model_size: str = "2b", width: str = "16k", device: str = "cuda") -> SAE:
    release = GEMMA_2B_RELEASE if model_size == "2b" else GEMMA_9B_RELEASE
    l0 = "71" if model_size == "2b" else "82"
    sae_id = f"layer_{layer}/width_{width}/average_l0_{l0}"
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id)
    return sae.to(device)
```

- [ ] **Step 2: Commit**

```bash
git add src/sae/loader.py
git commit -m "feat: Gemma Scope SAE loader"
```

---

### Task 3.2: Feature extraction and ranking

**Files:**
- Create: `src/sae/features.py`
- Create: `tests/sae/test_features.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/sae/test_features.py
import numpy as np

def test_cohen_d_perfect_separation():
    from src.sae.features import cohen_d
    g1 = np.array([10.0, 10.1, 9.9])
    g2 = np.array([0.0, 0.1, -0.1])
    assert cohen_d(g1, g2) > 5.0

def test_cohen_d_identical():
    from src.sae.features import cohen_d
    g = np.array([1.0, 1.0, 1.0])
    assert cohen_d(g, g) == 0.0

def test_rank_features_top_signal():
    from src.sae.features import rank_features_by_effect_size
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (20, 100))
    X[:10, 5] += 5.0
    labels = np.array([1]*10 + [0]*10)
    ranked = rank_features_by_effect_size(X, labels, top_k=10)
    assert len(ranked) == 10
    assert ranked[0]["feature_idx"] == 5
```

- [ ] **Step 2: Run — FAIL**

```bash
python -m pytest tests/sae/test_features.py -v
```

- [ ] **Step 3: Write implementation**

```python
# src/sae/features.py
from __future__ import annotations
import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*group1.std()**2 + (n2-1)*group2.std()**2) / (n1+n2-2))
    return 0.0 if pooled_std == 0 else (group1.mean() - group2.mean()) / pooled_std


def get_feature_activations(
    model: HookedTransformer, sae: SAE,
    sentences: list[str], layer: int,
    pooling: str = "mean", device: str = "cuda",
) -> np.ndarray:
    all_acts = []
    for sentence in sentences:
        tokens = model.to_tokens(sentence).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            resid = cache["resid_post", layer]
            feat_acts = sae.encode(resid[0])
        vec = feat_acts.mean(0).cpu().float().numpy() if pooling == "mean" else feat_acts[-1].cpu().float().numpy()
        all_acts.append(vec)
    return np.stack(all_acts)


def rank_features_by_effect_size(feature_acts: np.ndarray, labels: np.ndarray, top_k: int = 20) -> list[dict]:
    high = feature_acts[labels == 1]
    low  = feature_acts[labels == 0]
    results = [
        {"feature_idx": f, "cohen_d": abs(cohen_d(high[:, f], low[:, f])),
         "cohen_d_signed": cohen_d(high[:, f], low[:, f]),
         "mean_high": high[:, f].mean(), "mean_low": low[:, f].mean()}
        for f in range(feature_acts.shape[1])
    ]
    results.sort(key=lambda x: x["cohen_d"], reverse=True)
    return results[:top_k]
```

- [ ] **Step 4: Run — PASS**

```bash
python -m pytest tests/sae/test_features.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/sae/features.py tests/sae/test_features.py
git commit -m "feat: SAE feature extraction and Cohen-d ranking"
```

---

### Task 3.3: Selectivity scoring

**Files:**
- Create: `src/sae/selectivity.py`
- Create: `tests/sae/test_selectivity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/sae/test_selectivity.py
import numpy as np

def test_selectivity_high_when_confound_random():
    from src.sae.selectivity import compute_selectivity
    rng = np.random.default_rng(0)
    n = 40
    honorific = np.array([1]*20 + [0]*20)
    feature = np.where(honorific==1, 5.0, 0.0) + rng.normal(0, 0.1, n)
    confound = rng.integers(0, 2, n)
    assert compute_selectivity(feature, honorific, confound) > 0.5

def test_selectivity_low_when_confound_matches():
    from src.sae.selectivity import compute_selectivity
    rng = np.random.default_rng(0)
    n = 40
    labels = np.array([1]*20 + [0]*20)
    feature = np.where(labels==1, 5.0, 0.0) + rng.normal(0, 0.1, n)
    assert compute_selectivity(feature, labels, labels.copy()) < 0.5
```

- [ ] **Step 2: Run — FAIL**

```bash
python -m pytest tests/sae/test_selectivity.py -v
```

- [ ] **Step 3: Write implementation**

```python
# src/sae/selectivity.py
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _auroc_single(col: np.ndarray, labels: np.ndarray) -> float:
    if col.std() == 0:
        return 0.5
    col_s = StandardScaler().fit_transform(col.reshape(-1, 1))
    clf = LogisticRegression(max_iter=200).fit(col_s, labels)
    return roc_auc_score(labels, clf.predict_proba(col_s)[:, 1])


def compute_selectivity(feature_col: np.ndarray, honorific_labels: np.ndarray, confound_labels: np.ndarray) -> float:
    auroc_h = _auroc_single(feature_col, honorific_labels)
    auroc_c = _auroc_single(feature_col, confound_labels)
    return max(0.0, 1.0 - auroc_c / auroc_h) if auroc_h > 0.5 else 0.0


def score_top_features(
    feature_acts: np.ndarray, honorific_labels: np.ndarray,
    confound_labels_dict: dict[str, np.ndarray], top_feature_indices: list[int],
) -> list[dict]:
    results = []
    for f_idx in top_feature_indices:
        col = feature_acts[:, f_idx]
        row: dict = {"feature_idx": f_idx}
        for name, conf_labels in confound_labels_dict.items():
            row[f"sel_{name}"] = compute_selectivity(col, honorific_labels, conf_labels)
        row["mean_selectivity"] = float(np.mean([v for k, v in row.items() if k.startswith("sel_")]))
        results.append(row)
    return results
```

- [ ] **Step 4: Run — PASS**

```bash
python -m pytest tests/sae/test_selectivity.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/sae/selectivity.py tests/sae/test_selectivity.py
git commit -m "feat: selectivity scoring for features vs. confound dimensions"
```

---

### Task 3.4: Run MPU SAE feature discovery

**Files:**
- Create: `experiments/02_sae_features.py`

- [ ] **Step 1: Write experiment script**

```python
# experiments/02_sae_features.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_stripped_condition, ALL_PRONOUNS
from src.sae.loader import load_sae
from src.sae.features import get_feature_activations, rank_features_by_effect_size
from src.sae.selectivity import score_top_features

def main():
    probe_res = json.load(open("experiments/results_probe_pronoun_stripped.json"))
    best_layer_idx = int(max(probe_res, key=lambda k: probe_res[k]["auroc_mean"]))
    print(f"Using layer {best_layer_idx} (AUROC={probe_res[str(best_layer_idx)]['auroc_mean']:.4f})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = load_dataset("data/hindi/pilot_30.json")
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    sae   = load_sae(layer=best_layer_idx, model_size="2b", device=device)

    sentences, labels, pronoun_flags = [], [], []
    for p in pairs:
        sentences.extend([p.high, p.low])
        labels.extend([1, 0])
        for s in [p.high, p.low]:
            pronoun_flags.append(int(any(t.lower() in ALL_PRONOUNS for t in s.split())))
    labels = np.array(labels)
    confound_labels = {"pronoun_present": np.array(pronoun_flags)}

    print(f"Extracting SAE activations for {len(sentences)} sentences...")
    feature_acts = get_feature_activations(model, sae, sentences, layer=best_layer_idx, device=device)

    top20 = rank_features_by_effect_size(feature_acts, labels, top_k=20)
    top_indices = [f["feature_idx"] for f in top20]
    sel_scores  = score_top_features(feature_acts, labels, confound_labels, top_indices)

    print("\n=== Top-20 features ===")
    for f, s in zip(top20, sel_scores):
        print(f"  Feat {f['feature_idx']:6d}: d={f['cohen_d']:.3f}, sel={s['mean_selectivity']:.3f}")

    g2_pass = any(f["cohen_d"] > 0.5 and s["mean_selectivity"] > 0.6 for f, s in zip(top20, sel_scores))
    print(f"\nG2: {'PASS' if g2_pass else 'FAIL'}")

    json.dump({"top_features": top20, "selectivity": sel_scores, "best_layer": best_layer_idx},
              open("experiments/results_sae_features.json", "w"), indent=2)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
python experiments/02_sae_features.py 2>&1 | tee experiments/sae_features_log.txt
```

- [ ] **Step 3: Commit**

```bash
git add experiments/02_sae_features.py experiments/results_sae_features.json experiments/sae_features_log.txt
git commit -m "feat: MPU SAE feature discovery — G2 check"
```

---

## Phase 4: MPU Causal Triangle

**Goal:** Additive steering (+α), ablation (−α), encode-side zero-ablation. Manual rubric scoring. **MPU G3 + G4.**

**Estimated time:** 2–3 days

### Task 4.1: Steering modules

**Files:**
- Create: `src/steering/additive.py`
- Create: `src/steering/ablation.py`

- [ ] **Step 1: Write additive steering**

```python
# src/steering/additive.py
from __future__ import annotations
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE


def steer_generate(
    model: HookedTransformer, sae: SAE,
    prompt: str, feature_idx: int, alpha: float, layer: int,
    max_new_tokens: int = 80, device: str = "cuda",
) -> str:
    steering_vec = (alpha * sae.W_dec[feature_idx]).to(device)

    def hook_fn(value, hook):
        return value + steering_vec.unsqueeze(0).unsqueeze(0)

    tokens = model.to_tokens(prompt).to(device)
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, tokens.shape[1]:])
```

- [ ] **Step 2: Write ablation module**

```python
# src/steering/ablation.py
from __future__ import annotations
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE


def ablate_generate(
    model: HookedTransformer, sae: SAE,
    prompt: str, feature_idx: int, alpha: float, layer: int,
    max_new_tokens: int = 80, device: str = "cuda",
    mode: str = "negative",  # "negative" | "zero_ablation"
) -> str:
    tokens = model.to_tokens(prompt).to(device)

    if mode == "negative":
        steering_vec = (-alpha * sae.W_dec[feature_idx]).to(device)
        def hook_fn(value, hook):
            return value + steering_vec.unsqueeze(0).unsqueeze(0)
    elif mode == "zero_ablation":
        def hook_fn(value, hook):
            feat_acts = sae.encode(value)
            feat_acts[..., feature_idx] = 0.0
            return sae.decode(feat_acts)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, tokens.shape[1]:])
```

- [ ] **Step 3: Commit**

```bash
git add src/steering/additive.py src/steering/ablation.py
git commit -m "feat: additive steering and ablation modules"
```

---

### Task 4.2: Run causal triangle experiment

**Files:**
- Create: `experiments/03_causal_triangle.py`

- [ ] **Step 1: Write experiment (set FEATURE_IDX from Phase 3 results before running)**

```python
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
```

- [ ] **Step 2: Run experiment**

```bash
python experiments/03_causal_triangle.py 2>&1 | tee experiments/causal_triangle_log.txt
```

- [ ] **Step 3: MANUAL — Score rubric**

Open `experiments/results_causal_triangle.json`. For each output, score 6 axes (0–3):
- `addressee` — Is pronoun choice appropriate?
- `verb_agreement` — Does verb agree with pronoun?
- `register_consistency` — Consistent level throughout?
- `context_appropriateness` — Pragmatically appropriate?
- `fluency` — Fluent Hindi?
- `content_preservation` — Answers the prompt?

Save to `experiments/rubric_scores_mpu.csv` with columns:
`condition,prompt_id,addressee,verb_agreement,register_consistency,context_appropriateness,fluency,content_preservation`

- [ ] **Step 4: Evaluate G3 + G4**

```bash
python -c "
import pandas as pd
from scipy import stats

df = pd.read_csv('experiments/rubric_scores_mpu.csv')
baseline = df[df.condition=='baseline']['addressee']
steered  = df[df.condition=='steering_pos']['addressee']
aap_nat  = df[df.condition=='aap_natural']['addressee']
ablated  = df[df.condition=='ablation_neg']['addressee']

t3, p3 = stats.ttest_ind(steered, baseline)
t4, p4 = stats.ttest_ind(aap_nat, ablated, alternative='greater')
print(f'G3 (steering): t={t3:.3f}, p={p3:.4f} -> {\"PASS\" if p3<0.01 else \"FAIL\"}')
print(f'G4 (ablation): t={t4:.3f}, p={p4:.4f} -> {\"PASS\" if p4<0.1 else \"FAIL\"}')
"
```

- [ ] **Step 5: Write MPU decision**

Manually create `experiments/mpu_decision.md`:
```
# MPU Decision

- G1: Layer N, AUROC X — [PASS/FAIL]
- G2: Feature M, d=X, sel=Y — [PASS/FAIL]
- G3: p=X — [PASS/FAIL]
- G4: p=X — [PASS/FAIL]

Decision: PROCEED / PIVOT TO NULL RESULT
```

- [ ] **Step 6: Commit**

```bash
git add src/steering/ experiments/03_causal_triangle.py experiments/results_causal_triangle.json experiments/mpu_decision.md
git commit -m "feat: MPU causal triangle complete — G3+G4 evaluated"
```

---

> **GATE: If mpu_decision.md says PIVOT — stop here and write null-result framing. If PROCEED, continue.**

---

## Phase 5: Full Hindi Dataset — 600 Pairs

**Goal:** Scale from 30 to ~600 pairs with full confound labels and IAA (Krippendorff's α ≥ 0.67).

**Estimated time:** 1–2 weeks (annotation-heavy)

### Task 5.1: Annotation tooling + IAA

**Files:**
- Create: `src/data/annotate.py`
- Create: `data/hindi/annotation_guidelines.md` (manual write)

- [ ] **Step 1: Write IAA module**

```python
# src/data/annotate.py
from __future__ import annotations
import numpy as np

def krippendorff_alpha(data: list[list], level: str = "ordinal") -> float:
    import krippendorff
    return krippendorff.alpha(reliability_data=data, level_of_measurement=level)

def report_iaa(ratings_by_annotator: list[list]) -> dict:
    alpha = krippendorff_alpha(ratings_by_annotator)
    print(f"Krippendorff's alpha = {alpha:.4f}")
    if alpha >= 0.80:
        status = "strong"
    elif alpha >= 0.67:
        status = "moderate"
    else:
        status = "weak — review disagreements before proceeding"
    print(f"  -> {status} agreement")
    return {"alpha": alpha, "status": status}
```

- [ ] **Step 2: Add `krippendorff` to requirements.txt**

```
krippendorff>=0.6.0
```

- [ ] **Step 3: Write annotation guidelines**

Create `data/hindi/annotation_guidelines.md` manually. Must cover:
- Minimal-contrastive definition (same propositional content, only pronoun+verb agreement changes)
- How to label domain (social/professional/service/family)
- How to fill confounds: pronoun_stripped, script_roman, code_switched, length_ratio
- Edge cases: gender-marked verbs, imperative constructions, embedded clauses
- Examples with correct and incorrect labels
- Quality control flags: "ambiguous", "non-minimal"

- [ ] **Step 4: Commit**

```bash
pip install krippendorff
git add src/data/annotate.py requirements.txt data/hindi/annotation_guidelines.md
git commit -m "feat: IAA tooling + annotation guidelines for full dataset"
```

---

### Task 5.2: Scale to 600 pairs + IAA validation

- [ ] **Step 1: Draft pairs in 6 batches of 95 (~570 + 30 pilot = 600)**

Use `experiments/draft_pilot.py` as template. Target:
- 200 `aap` vs `tum`, 200 `aap` vs `tu`, 200 `tum` vs `tu`
- All in Devanagari script (unlike the romanized pilot)
- 200 pairs with `pronoun_stripped_*` confound filled
- 150 pairs with `script_roman_*` (use indic-trans2 for transliteration)
- 100 pairs with `code_switched_*`

Save staging batches to `data/hindi/staging/batch{1..6}.json`.

- [ ] **Step 2: Share 150 items with second annotator + collect ratings**

Select 150 pairs across all three pair types. Share with a second Hindi-native speaker.
Collect their ratings on: `(a) is this a valid minimal-contrastive pair? (b) is the domain label correct?`

- [ ] **Step 3: Compute and check IAA**

```python
# run interactively
from src.data.annotate import report_iaa
annotator1 = [1, 1, 0, 1, ...]  # your labels for 150 shared items
annotator2 = [1, 0, 0, 1, ...]  # second annotator's labels
result = report_iaa([annotator1, annotator2])
assert result["alpha"] >= 0.67, "IAA too low — review disagreements"
```

- [ ] **Step 4: Merge to full_600.json**

```python
# run interactively
from src.data.pairs import load_dataset, save_dataset
from pathlib import Path
pairs = []
for batch_file in sorted(Path("data/hindi/staging").glob("*.json")):
    pairs.extend(load_dataset(batch_file))
pairs.extend(load_dataset("data/hindi/pilot_30.json"))  # include pilot
pairs = pairs[:600]
save_dataset(pairs, "data/hindi/full_600.json")
print(f"Saved {len(pairs)} pairs")
```

- [ ] **Step 5: Commit**

```bash
git add data/hindi/full_600.json
git commit -m "data: 600-pair Hindi honorific dataset, CC-BY-SA"
```

---

## Phase 6: Full Probing Study

**Goal:** Layer-wise probing on Gemma-2-2B and Gemma-2-9B with the full 600-pair dataset, all confound conditions. Confirm H1.

**Estimated time:** 1 week (mostly compute time)

### Task 6.1: Full probe experiment

**Files:**
- Create: `experiments/04_full_probe.py`

- [ ] **Step 1: Write experiment**

```python
# experiments/04_full_probe.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_all_conditions
from src.probe.extract import extract_residuals
from src.probe.probe import run_layerwise_probes, print_probe_report

DATASET_PATH = "data/hindi/full_600.json"
CONDITIONS   = ["raw", "pronoun_stripped", "roman_script", "code_switched"]
MODELS       = [("google/gemma-2-2b", 26, "gemma2b"), ("google/gemma-2-9b", 42, "gemma9b")]

def run_model(model_name, n_layers, tag):
    device = "cuda"
    pairs = load_dataset(DATASET_PATH)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    layers = list(range(n_layers))
    all_results = {}
    for condition in CONDITIONS:
        sentences, labels = [], []
        for p in pairs:
            conds = get_all_conditions(p)
            if condition not in conds:
                continue
            h, l = conds[condition]
            if not h or not l:
                continue
            sentences.extend([h, l])
            labels.extend([1, 0])
        labels = np.array(labels)
        acts = extract_residuals(model, sentences, layers=layers, device=device)
        results = run_layerwise_probes(acts, labels)
        print_probe_report(results, condition=f"{tag}/{condition}")
        all_results[condition] = {
            str(l): {"auroc_mean": v["auroc_mean"], "auroc_std": v["auroc_std"]}
            for l, v in results.items()
        }
    json.dump(all_results, open(f"experiments/results_full_probe_{tag}.json", "w"), indent=2)

def main():
    for model_name, n_layers, tag in MODELS:
        print(f"\n{'='*60}\nRunning {tag}\n{'='*60}")
        run_model(model_name, n_layers, tag)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run (will take several hours)**

```bash
python experiments/04_full_probe.py 2>&1 | tee experiments/full_probe_log.txt
```

- [ ] **Step 3: Commit**

```bash
git add experiments/04_full_probe.py experiments/results_full_probe_*.json
git commit -m "feat: full probing study on 600 pairs — Gemma-2-2B and 9B"
```

---

## Phase 7: Full SAE Feature Study

**Goal:** Comprehensive feature ranking + selectivity on all confound dimensions. Manual validation of top-20. Finalize candidate features.

**Estimated time:** 1 week

### Task 7.1: Full feature discovery

**Files:**
- Create: `experiments/05_full_features.py`

- [ ] **Step 1: Write experiment**

```python
# experiments/05_full_features.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.data.confounds import get_stripped_condition, ALL_PRONOUNS
from src.sae.loader import load_sae
from src.sae.features import get_feature_activations, rank_features_by_effect_size
from src.sae.selectivity import score_top_features

def main():
    probe_res = json.load(open("experiments/results_full_probe_gemma2b.json"))
    best_layer_idx = int(max(probe_res["pronoun_stripped"],
                             key=lambda k: probe_res["pronoun_stripped"][k]["auroc_mean"]))
    print(f"Best layer: {best_layer_idx}")

    device = "cuda"
    pairs = load_dataset("data/hindi/full_600.json")
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    sae   = load_sae(layer=best_layer_idx, model_size="2b", device=device)

    sentences, labels = [], []
    confound_arrays: dict[str, list[int]] = {"pronoun_present": [], "is_roman_script": [], "is_code_switched": []}
    for p in pairs:
        sentences.extend([p.high, p.low])
        labels.extend([1, 0])
        for s, lbl in [(p.high, 1), (p.low, 0)]:
            confound_arrays["pronoun_present"].append(int(any(t.lower() in ALL_PRONOUNS for t in s.split())))
            confound_arrays["is_roman_script"].append(
                int((lbl==1 and "script_roman_high" in p.confounds) or (lbl==0 and "script_roman_low" in p.confounds)))
            confound_arrays["is_code_switched"].append(
                int((lbl==1 and "code_switched_high" in p.confounds) or (lbl==0 and "code_switched_low" in p.confounds)))

    labels = np.array(labels)
    confound_labels = {k: np.array(v) for k, v in confound_arrays.items()}

    print(f"Extracting SAE activations for {len(sentences)} sentences...")
    feature_acts = get_feature_activations(model, sae, sentences, layer=best_layer_idx, device=device)

    top50 = rank_features_by_effect_size(feature_acts, labels, top_k=50)
    sel   = score_top_features(feature_acts, labels, confound_labels, [f["feature_idx"] for f in top50])

    print("\n=== Top-20 features ===")
    for f, s in zip(top50[:20], sel[:20]):
        print(f"  Feat {f['feature_idx']:6d}: d={f['cohen_d']:.3f} "
              f"sel_pronoun={s.get('sel_pronoun_present',0):.3f} "
              f"sel_roman={s.get('sel_is_roman_script',0):.3f} "
              f"mean_sel={s['mean_selectivity']:.3f}")

    json.dump({"top_features": top50, "selectivity": sel, "best_layer": best_layer_idx},
              open("experiments/results_full_features.json", "w"), indent=2)
    print("Saved results_full_features.json")
    print("\n[MANUAL] Read top-20 highest-activating examples for each feature. Record in feature_validation_notes.md")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
python experiments/05_full_features.py 2>&1 | tee experiments/full_features_log.txt
```

- [ ] **Step 3: Manual validation — create `experiments/feature_validation_notes.md`**

For each of the top-20 features, inspect highest-activating sentences (add a quick script to print top-10 activating sentences per feature). Write native-speaker judgment of what each feature appears to track.

- [ ] **Step 4: Commit**

```bash
git add experiments/05_full_features.py experiments/results_full_features.json experiments/feature_validation_notes.md
git commit -m "feat: full SAE feature study + manual top-20 validation"
```

---

## Phase 8: Full Causal Triangle + Baselines

**Goal:** All baselines + full causal triangle on top-3 features. Confirm H2, H3.

**Estimated time:** 2–3 weeks (compute + annotation)

### Task 8.1: Prompting baselines

**Files:**
- Create: `src/baselines/prompting.py`

- [ ] **Step 1: Write prompting module**

```python
# src/baselines/prompting.py
from __future__ import annotations
import torch
from transformer_lens import HookedTransformer

ZERO_SHOT_TEMPLATE = "Niche diye gaye vakya ko {level} mein likhiye.\nVakya: {sentence}\n{level} rupantar:"

FOUR_SHOT_EXAMPLES = {
    "aap": [
        ("tum kaise ho?", "aap kaise hain?"),
        ("tera naam kya hai?", "aapka naam kya hai?"),
        ("tum kahan ja rahe ho?", "aap kahan ja rahe hain?"),
        ("tumhara kaam ho gaya.", "aapka kaam ho gaya."),
    ],
}

def zero_shot_prompt(sentence: str, target_level: str) -> str:
    level_map = {"aap": "aap-rup (formal)", "tum": "tum-rup", "tu": "tu-rup (informal)"}
    return ZERO_SHOT_TEMPLATE.format(level=level_map[target_level], sentence=sentence)

def four_shot_prompt(sentence: str, target_level: str) -> str:
    examples = FOUR_SHOT_EXAMPLES.get(target_level, [])
    lines = [f"Original: {src}\n{target_level}-rup: {tgt}" for src, tgt in examples]
    lines.append(f"Original: {sentence}\n{target_level}-rup:")
    return "\n\n".join(lines)

def generate_with_prompt(model: HookedTransformer, prompt: str, max_new_tokens: int = 60, device: str = "cuda") -> str:
    tokens = model.to_tokens(prompt).to(device)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, tokens.shape[1]:])
```

- [ ] **Step 2: Commit**

```bash
git add src/baselines/prompting.py
git commit -m "feat: zero-shot and 4-shot prompting baselines"
```

---

### Task 8.2: Full baselines + steering experiment

**Files:**
- Create: `experiments/06_baselines.py`

- [ ] **Step 1: Write experiment (set top3_indices from Phase 7 results)**

```python
# experiments/06_baselines.py
import json, torch, numpy as np
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.sae.loader import load_sae
from src.steering.additive import steer_generate
from src.steering.ablation import ablate_generate
from src.baselines.prompting import zero_shot_prompt, four_shot_prompt, generate_with_prompt

MODEL_NAME = "google/gemma-2-2b"
ALPHAS     = [5.0, 10.0, 20.0, 30.0, 40.0]
N_TEST     = 40  # held-out prompts

def main():
    feats_res = json.load(open("experiments/results_full_features.json"))
    best_layer = feats_res["best_layer"]
    top3 = [f["feature_idx"] for f in feats_res["top_features"][:3]]
    print(f"Top-3 features: {top3}, best layer: {best_layer}")

    device = "cuda"
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    sae   = load_sae(layer=best_layer, model_size="2b", device=device)

    all_pairs = load_dataset("data/hindi/full_600.json")
    test_pairs = all_pairs[-120:][:N_TEST]  # held-out set

    results = {}

    # Zero-shot + 4-shot baselines
    for method, prompt_fn in [("zero_shot", zero_shot_prompt), ("four_shot", four_shot_prompt)]:
        for p in test_pairs:
            prompt = prompt_fn(p.low, "aap")
            out = generate_with_prompt(model, prompt, device=device)
            results.setdefault(method, []).append({"prompt": p.low, "output": out})

    # Alpha sweep for top-3 features
    for feat_idx in top3:
        for alpha in ALPHAS:
            for p in test_pairs:
                out = steer_generate(model, sae, p.low, feat_idx, alpha, best_layer, device=device)
                key = f"steer_feat{feat_idx}_alpha{alpha}"
                results.setdefault(key, []).append({"prompt": p.low, "output": out})

    # Ablation on aap prompts
    for feat_idx in top3:
        for p in test_pairs:
            out_neg  = ablate_generate(model, sae, p.high, feat_idx, 20.0, best_layer, mode="negative", device=device)
            out_zero = ablate_generate(model, sae, p.high, feat_idx, 20.0, best_layer, mode="zero_ablation", device=device)
            results.setdefault(f"ablate_neg_feat{feat_idx}", []).append({"prompt": p.high, "output": out_neg})
            results.setdefault(f"ablate_zero_feat{feat_idx}", []).append({"prompt": p.high, "output": out_zero})

    json.dump(results, open("experiments/results_baselines_and_steering.json", "w"), ensure_ascii=False, indent=2)
    print("Saved. [MANUAL] Score with rubric -> experiments/rubric_scores_full.csv")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
python experiments/06_baselines.py 2>&1 | tee experiments/baselines_log.txt
```

- [ ] **Step 3: MANUAL — Full rubric scoring**

Score all outputs (target: N ≥ 200 generations, 2–3 raters per generation). Save to `experiments/rubric_scores_full.csv`.

- [ ] **Step 4: Commit**

```bash
git add src/baselines/ experiments/06_baselines.py experiments/results_baselines_and_steering.json
git commit -m "feat: full causal triangle and all baselines"
```

---

## Phase 9: Japanese Comparison

**Goal:** Probe + feature-ID on ~200 Japanese keigo pairs. Descriptive layer + selectivity comparison (RQ4, RQ5).

**Estimated time:** 2 weeks

### Task 9.1: Japanese pipeline

**Files:**
- Create: `src/japanese/pipeline.py`
- Create: `experiments/07_japanese.py`
- Create: `data/japanese/keigo_200.json` (manual construction + JESC sourcing)

- [ ] **Step 1: Write Japanese pipeline (reuses probe modules)**

```python
# src/japanese/pipeline.py
from __future__ import annotations
import numpy as np
from transformer_lens import HookedTransformer
from src.data.pairs import load_dataset
from src.probe.extract import extract_residuals
from src.probe.probe import run_layerwise_probes, print_probe_report, best_layer


def run_japanese_probe(model: HookedTransformer, dataset_path: str, layers: list[int], device: str = "cuda") -> dict:
    pairs = load_dataset(dataset_path)
    sentences = [s for p in pairs for s in [p.high, p.low]]
    labels = np.array([1, 0] * len(pairs))
    acts = extract_residuals(model, sentences, layers=layers, device=device)
    results = run_layerwise_probes(acts, labels)
    print_probe_report(results, condition="japanese_keigo")
    return results
```

- [ ] **Step 2: Construct Japanese dataset**

Source 200 teineigo vs. plain-form pairs from JESC. Use `HonorificPair` schema where:
- `high` = teineigo form (`desu`/`masu` endings)
- `low` = plain form
- `domain`: adapt to Japanese contexts
- Get 1–2 Japanese-native raters via Prolific (~$150 budget)
- Report IAA with `src/data/annotate.report_iaa`

- [ ] **Step 3: Write Japanese comparison experiment**

```python
# experiments/07_japanese.py
import json, numpy as np, torch
from transformer_lens import HookedTransformer
from src.sae.loader import load_sae
from src.sae.features import get_feature_activations, rank_features_by_effect_size
from src.japanese.pipeline import run_japanese_probe
from src.probe.probe import best_layer

MODEL_NAME = "google/gemma-2-2b"
JP_PATH    = "data/japanese/keigo_200.json"
LAYERS     = list(range(26))

def main():
    device = "cuda"
    model  = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

    jp_results  = run_japanese_probe(model, JP_PATH, LAYERS, device)
    jp_best, jp_auroc = best_layer(jp_results)

    hindi_res = json.load(open("experiments/results_full_probe_gemma2b.json"))
    hindi_ps  = hindi_res["pronoun_stripped"]
    hindi_best = int(max(hindi_ps, key=lambda k: hindi_ps[k]["auroc_mean"]))
    hindi_auroc = hindi_ps[str(hindi_best)]["auroc_mean"]

    print(f"Hindi best layer (ps): {hindi_best}, AUROC: {hindi_auroc:.4f}")
    print(f"Japanese best layer:   {jp_best}, AUROC: {jp_auroc:.4f}")
    print(f"Layer difference: {jp_best - hindi_best} (positive = Japanese later)")

    sae = load_sae(layer=jp_best, model_size="2b", device=device)
    from src.data.pairs import load_dataset
    pairs = load_dataset(JP_PATH)
    sentences = [s for p in pairs for s in [p.high, p.low]]
    labels = np.array([1, 0] * len(pairs))
    feat_acts = get_feature_activations(model, sae, sentences, layer=jp_best, device=device)
    jp_top20  = rank_features_by_effect_size(feat_acts, labels, top_k=20)

    out = {
        "hindi_best_layer": hindi_best, "hindi_auroc": hindi_auroc,
        "japanese_best_layer": jp_best, "japanese_auroc": jp_auroc,
        "jp_top_features": jp_top20,
    }
    json.dump(out, open("experiments/results_japanese.json", "w"), indent=2)
    print("Saved results_japanese.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run + commit**

```bash
python experiments/07_japanese.py 2>&1 | tee experiments/japanese_log.txt
git add src/japanese/ experiments/07_japanese.py experiments/results_japanese.json data/japanese/keigo_200.json
git commit -m "feat: Japanese keigo comparison — probe + feature ID"
```

---

## Phase 10: Statistical Analysis + Fluency/BLEURT

**Goal:** Mixed-effects models, Bonferroni correction, bootstrap CIs, PPL inflation, BLEURT content preservation.

**Estimated time:** 1 week

### Task 10.1: Statistical analysis

**Files:**
- Create: `src/stats/analysis.py`
- Create: `src/eval/fluency.py`
- Create: `src/eval/content.py`
- Create: `experiments/08_stats.py`

- [ ] **Step 1: Write analysis module**

```python
# src/stats/analysis.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(data: np.ndarray, n: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    means = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n)]
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    threshold = alpha / len(p_values)
    return [p < threshold for p in p_values]


def compare_conditions(
    df: pd.DataFrame, outcome: str, group_col: str,
    baseline: str, comparisons: list[str],
) -> pd.DataFrame:
    baseline_data = df[df[group_col] == baseline][outcome].values
    rows, p_vals = [], []
    for cond in comparisons:
        cond_data = df[df[group_col] == cond][outcome].values
        t, p = stats.ttest_ind(cond_data, baseline_data)
        ci_lo, ci_hi = bootstrap_ci(cond_data)
        rows.append({"condition": cond, "mean": cond_data.mean(),
                     "ci_low": ci_lo, "ci_high": ci_hi, "t": t, "p": p})
        p_vals.append(p)
    bonferroni_flags = bonferroni(p_vals)
    for r, sig in zip(rows, bonferroni_flags):
        r["significant_bonferroni"] = sig
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Write fluency module**

```python
# src/eval/fluency.py
import torch
import numpy as np
from transformer_lens import HookedTransformer

def sentence_ppl(model: HookedTransformer, text: str, device: str = "cuda") -> float:
    tokens = model.to_tokens(text).to(device)
    with torch.no_grad():
        logits = model(tokens)
    lp = torch.nn.functional.log_softmax(logits, dim=-1)
    token_lp = lp[0, :-1].gather(1, tokens[0, 1:].unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(-token_lp.mean()).item())

def ppl_inflation_check(baseline_ppls: list[float], steered_ppls: list[float]) -> dict:
    ratio = np.mean(steered_ppls) / np.mean(baseline_ppls)
    return {"baseline_ppl": np.mean(baseline_ppls), "steered_ppl": np.mean(steered_ppls),
            "ratio": ratio, "g3_ppl_pass": ratio < 1.5}
```

- [ ] **Step 3: Write content preservation module**

```python
# src/eval/content.py
_bleurt = None

def bleurt_score(references: list[str], candidates: list[str]) -> list[float]:
    global _bleurt
    if _bleurt is None:
        from evaluate import load
        _bleurt = load("bleurt", module_type="metric")
    return _bleurt.compute(references=references, predictions=candidates)["scores"]
```

- [ ] **Step 4: Write stats experiment**

```python
# experiments/08_stats.py
import pandas as pd
from src.stats.analysis import compare_conditions

RUBRIC_AXES = ["addressee", "verb_agreement", "register_consistency",
               "context_appropriateness", "fluency", "content_preservation"]

def main():
    df = pd.read_csv("experiments/rubric_scores_full.csv")
    comparisons = [c for c in df["condition"].unique() if c != "baseline"]
    for axis in RUBRIC_AXES:
        print(f"\n=== {axis} ===")
        result = compare_conditions(df, outcome=axis, group_col="condition",
                                    baseline="baseline", comparisons=comparisons)
        print(result.to_string(index=False))

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run stats**

```bash
python experiments/08_stats.py 2>&1 | tee experiments/stats_log.txt
git add src/stats/ src/eval/ experiments/08_stats.py
git commit -m "feat: statistical analysis — Bonferroni, bootstrap CIs, PPL check"
```

---

## Phase 11: Paper Writing + Artifact Packaging

**Goal:** Draft all 8 paper sections, all figures, packaged releasable artifacts.

**Estimated time:** 2 weeks

### Task 11.1: Figure generation

**Files:**
- Create: `paper/figures/fig1_probe_auroc.py`
- Create: `paper/figures/fig2_alpha_sweep.py`
- Create: `paper/figures/fig3_rubric_comparison.py`

- [ ] **Step 1: Figure 1 — Layer-wise AUROC curve**

```python
# paper/figures/fig1_probe_auroc.py
import json, matplotlib.pyplot as plt, numpy as np

def main():
    hindi_res = json.load(open("experiments/results_full_probe_gemma2b.json"))
    layers = sorted(int(k) for k in hindi_res["raw"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (cond, label) in zip(axes, [("raw", "Raw"), ("pronoun_stripped", "Pronoun-Stripped")]):
        aurocs = [hindi_res[cond][str(l)]["auroc_mean"] for l in layers]
        stds   = [hindi_res[cond][str(l)]["auroc_std"]  for l in layers]
        ax.plot(layers, aurocs, color="steelblue", label=f"Hindi ({label})")
        ax.fill_between(layers, np.subtract(aurocs, stds), np.add(aurocs, stds), alpha=0.2, color="steelblue")
        ax.axhline(0.70, color="red", linestyle="--", label="H1 threshold (0.70)")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer"); ax.set_ylabel("AUROC"); ax.set_title(f"{label}"); ax.legend()
    plt.tight_layout()
    plt.savefig("paper/figures/fig1_probe_auroc.pdf", bbox_inches="tight")
    print("fig1 saved")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run figures**

```bash
mkdir -p paper/figures
python paper/figures/fig1_probe_auroc.py
```

- [ ] **Step 3: Commit figures**

```bash
git add paper/figures/
git commit -m "feat: paper figures — probe AUROC and causal triangle"
```

---

### Task 11.2: Artifact packaging + SRW mentorship

- [ ] **Step 1: Package dataset with license field**

```python
# run interactively
import json
from pathlib import Path
p = Path("data/hindi/full_600.json")
data = json.loads(p.read_text())
for item in data:
    item["license"] = "CC-BY-SA-4.0"
p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
print("License field added")
```

- [ ] **Step 2: Package feature indices artifact**

```python
import json
feats = json.load(open("experiments/results_full_features.json"))
out = {
    "model": "google/gemma-2-2b",
    "sae_release": "gemma-scope-2b-pt-res",
    "best_layer": feats["best_layer"],
    "candidate_features": feats["top_features"][:3],
    "note": "Candidate honorific-associated feature indices. See paper for selectivity scores.",
}
import os; os.makedirs("artifacts", exist_ok=True)
json.dump(out, open("artifacts/candidate_feature_indices.json", "w"), indent=2)
```

- [ ] **Step 3: Apply for AACL SRW pre-submission mentorship**

**DEADLINE: 2026-06-08** — Submit by this date. Use the abstract from PROBLEM_STATEMENT.md.
URL: https://2026.aaclnet.org/calls/srw/

- [ ] **Step 4: Final paper section checklist**

Verify draft covers:
- [ ] Abstract (update with actual results)
- [ ] §1 Introduction — problem, why now, why AACL
- [ ] §2 Background — Hindi honorifics, Japanese keigo, SAE primer
- [ ] §3 Gap Analysis — the table from PROBLEM_STATEMENT.md §3
- [ ] §4 Methodology — dataset, probe, SAE, causal triangle, baselines
- [ ] §5 Results — probe curves, feature table, causal triangle results, baselines
- [ ] §6 Failure Mode Analysis — from §9.1
- [ ] §7 Japanese Comparison — descriptive, no causal claims
- [ ] §8 Conclusion + limitations + future work

- [ ] **Step 5: Final commit**

```bash
git add artifacts/ data/ paper/
git commit -m "release: artifacts, figures, and submission-ready paper"
```

---

## Timeline Summary

| Week | Phase | Primary output | Gate |
|---|---|---|---|
| 1 | 0 + 1 | Env + 30-pair pilot | smoke test passes |
| 1 | 2 | MPU probe pilot | G1 |
| 2 | 3 | MPU SAE features | G2 |
| 2–3 | 4 | MPU causal triangle + rubric | G3 + G4 |
| **3** | **MPU GATE** | **PROCEED or PIVOT** | **all 4 green** |
| 4–6 | 5 | Full 600-pair dataset | IAA >= 0.67 |
| 7 | 6 | Full probing on 2B + 9B | AUROC tables |
| 8 | 7 | Full SAE feature study | top-3 candidates |
| 9–11 | 8 | Baselines + full causal triangle | rubric scores |
| 12–13 | 9 | Japanese comparison | layer comparison |
| 14 | 10 | Statistical analysis | p-values + CIs |
| 15–16 | 11 | Paper draft + artifacts | submission-ready |

---

## Spec Coverage Check

| Spec requirement | Plan phase |
|---|---|
| RQ1 — localizability probe | Phase 2, 6 |
| RQ2 — SAE feature ID | Phase 3, 7 |
| RQ3 — full causal triangle | Phase 4, 8 |
| RQ4 — Japanese probe | Phase 9 |
| RQ5 — structural locus comparison | Phase 9 |
| H1 — AUROC > 0.70 | Phase 2, G1 gate |
| H2 — feature d > 0.5, sel > 0.6 | Phase 3, G2 gate |
| H3 — causal triangle | Phase 4, G3+G4 gates |
| §5.6 confounds | Phase 1.3, 3.3, 7.1 |
| §5.4 baselines | Phase 8 |
| §5.5 statistical plan | Phase 10 |
| §5.9 rubric | Phase 4 (manual), 10 |
| §9.1 failure modes | Ablation arm + failure characterization |
| Releasable artifacts | Phase 11.2 |
| Pre-registration | Before Phase 6 |
| AACL SRW mentorship (deadline 2026-06-08) | Phase 11.2 |
| MPU go/no-go §5.7 | Explicit gate after Phase 4 |
