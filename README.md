# Where Does Politeness Live in Hindi?

AACL-IJCNLP 2026 Student Research Workshop submission.

A mechanistic interpretability case study of Hindi honorific encoding (`aap` / `tum` / `tu` + verb agreement) in publicly-released Gemma Scope Sparse Autoencoders, with Japanese keigo as a comparative reference point.

**Primary document:** [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) — full proposal v4 with abstract, gap analysis, methodology, confound controls, MPU plan, and rubric.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

Pre-pilot. Next step: §10 first-week MPU pilot from `PROBLEM_STATEMENT.md`.

## Hardware target

NVIDIA RTX 4500 Ada (24GB VRAM). Code is designed to fit in 24GB with Gemma-2-9B bf16 + Gemma Scope SAE forward.

## Repo structure (planned)

```
.
├── PROBLEM_STATEMENT.md   # canonical proposal v4
├── README.md              # this file
├── CLAUDE.md              # context for future Claude Code sessions
├── requirements.txt
├── .gitignore
├── src/                   # (to be created) probe, SAE, steering, ablation
├── data/                  # (to be created) contrastive pairs, annotations
├── experiments/           # (to be created) pilot scripts
└── notebooks/             # (to be created) exploratory analysis
```

## License

Code: MIT (to be added).
Data + rubric artifacts: CC-BY-SA 4.0 (to be added on release).
