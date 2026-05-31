# Where Does Politeness Live in Hindi?
## A Mechanistic Case Study of Honorific Encoding in Multilingual LLMs, with Japanese as a Comparative Control

**Target venue:** AACL-IJCNLP 2026 Student Research Workshop
**Track:** Research Paper (Long, 8 pages)
**Status:** Pre-submission draft v4 — for novelty validation
**Author context:** Native Hindi speaker based in India; NVIDIA RTX 4500 Ada (24GB VRAM) lab GPU
**Last updated:** 2026-05-30

---

## Abstract

Sparse autoencoders (SAEs) have recently shown that multilingual language models encode language identity as low-dimensional, causally manipulable directions in their residual stream (Deng et al., 2025; Liu et al., 2025). These studies treat each language as a monolithic label. We ask a narrower question with sharper sociolinguistic and venue-relevant stakes: **within Hindi — a major Indic language whose honorific marking is encoded pronominally (`aap` / `tum` / `tu`) plus verb agreement rather than via the verb-suffix morphology of East Asian honorific systems — do publicly-released SAEs (Gemma Scope) contain features that selectively encode honorific level, and can they be used to causally steer Hindi generation while controlling for the obvious confounds (pronoun token detection, length, code-switching, script choice)?**

The paper's primary contribution is a careful, controlled case study on Hindi `aap` / `tum` / `tu` register using Gemma-2-2B and Gemma-2-9B: (1) layer-wise linear probing to localize register, (2) SAE feature attribution with explicit confound controls, and (3) the full *causal triangle* — correlation (probe), sufficiency (additive steering), AND necessity (ablation) — evaluated against strong baselines (prompting, FUDGE, LoRA) on a structured multi-axis rubric scored by native speakers.

As a *comparative control* (not a co-equal claim), we replicate the probing and feature-identification pipeline on **Japanese keigo** — the canonical verb-suffix honorific system — to ask whether the representational structure we find in Hindi is locus-dependent (pronoun-tied) or part of a more abstract politeness axis the model has discovered. Japanese is included because it gives interpretability-skeptical reviewers a known-cleaner phenomenon as a reference point; we make no strong cross-lingual identity claims.

We are explicit about what we cannot establish (true monosemanticity of SAE features, robustness in long generations, generalization beyond contrastive pairs). The work produces four releasable artifacts: a confound-controlled Hindi honorific dataset annotated by the author and one additional Hindi-native rater, a multi-axis honorific evaluation rubric specialized for Hindi, identified candidate SAE feature indices for Gemma Scope, and an inference-time steering+ablation pipeline reproducible on a single 24GB consumer GPU.

---

## 1. Problem Statement

### 1.1 What's the problem?

Modern multilingual LLMs deployed in Indian-language contexts routinely misjudge honorific level. A model that:

- addresses an elderly Hindi speaker with `tu` instead of `aap`,
- uses the singular verb form (`karta hai`) when the plural-respectful (`karte hain`) is required,
- mixes `aap` (formal pronoun) with non-plural verb agreement in the same clause (a grammatical violation, not just a stylistic slip),

is producing fluent Hindi that nonetheless commits a social or grammatical error a competent native speaker would never make. These errors are visible in production deployments across India (chatbots, customer service, voice assistants serving Hindi-speaking users) and they matter materially: addressing a judge, an elderly relative, or a stranger with `tu` is socially loaded in a way that "casual vs formal English" is not.

The current toolkit for addressing this is narrow: prompting works inconsistently and degrades on long generations; fine-tuning is expensive per-language and risks regressing other capabilities; MT-style honorific control is partially solved for sentence-level translation but does not transfer to interactive generation.

What we *do not know* is whether the underlying multilingual LLMs already represent honorific level in Hindi as a structured, controllable feature internally. Without that knowledge, every intervention is surface-level.

### 1.2 Why now?

Three converging developments make this the right moment:

1. **Public SAE infrastructure has matured.** Gemma Scope (Lieberum et al., 2024) provides pretrained, interpretable SAE feature decoders at every layer of Gemma-2-2B and Gemma-2-9B. Eighteen months ago this study would have required training your own SAEs at significant compute cost.

2. **Multilingual mechanistic interpretability has broken out of English-only mech-interp.** A wave of 2025 papers (arXiv:2504.04264, arXiv:2505.05111, arXiv:2507.13410) has established that language identity is causally encoded in middle-layer residual streams. What *else* lives in this region — register, formality, honorifics, dialect — is an open question.

3. **Indic-language NLP has matured as a research priority but has been almost entirely external to mech-interp.** There are dozens of Indic benchmarks (IndicXTREME, IndicGenBench, MILU), a wave of Indic-focused open models (Sarvam-1, Airavata, OpenHathi, Krutrim), and an active community — but essentially zero published work opening these or any other multilingual models to ask *where* Indic sociolinguistic phenomena are represented internally.

### 1.3 Why this matters for AACL — and specifically for an Indian author

The Association for Computational Linguistics' Asia-Pacific chapter (AACL) has a stated and reviewer-cultural priority for Asian-language NLP and culturally-grounded phenomena. Indic-NLP is one of the most strongly-represented sub-communities at AACL. Three structural advantages compound for this submission:

1. **Indian native-speaker authorship on an Indian-language interpretability paper** — the reviewer pool will recognize that the linguistic claims are grounded in lived intuition, not third-party annotation. This is a credibility signal AACL reviewers actively notice.

2. **Indic + interpretability is a near-empty intersection.** The closest prior work is monolingual evaluation papers; we know of no SAE-based or mechanistic-interpretability study on any Indic language at the time of writing.

3. **The Japanese comparison gives a credibility anchor for interp-skeptical reviewers** without diluting the primary Indic-NLP framing. Reviewers from the East Asian NLP community can read this as "Hindi as the focal case, Japanese as the familiar reference," which broadens the supportive reviewer base.

---

## 2. Background

### 2.1 Hindi honorifics — the primary focal phenomenon

Hindi marks honorific status primarily through:

- **Second-person pronouns** — `aap` (formal / respectful / addressing elders, strangers, superiors), `tum` (familiar / addressing equals, peers, close acquaintances), `tu` (intimate / addressing close family, deities, or dismissively)
- **Verb agreement** — plural verb forms (`karte hain`, `gaye the`) obligatorily co-occur with `aap`; singular forms (`karta hai`) co-occur with `tu` and (usually) `tum`
- **Honorific suffixation** — `-ji` on names (`Rohan-ji`), occasional honorific verb substitutions
- **Cross-clausal consistency** — switching between pronoun levels mid-conversation is a marked social act

The pronoun-plus-agreement structure means Hindi honorifics are *grammatically distributed*: a single sentence's honorific status is jointly determined by pronoun choice AND verb agreement. This is structurally different from Japanese, where the verb suffix alone (`-masu` vs. plain form) carries the bulk of the marking. This typological difference is the heart of the comparative claim (§2.2).

A practical note for SAE work: pronoun tokens in Hindi are highly informative surface markers. A "feature that detects `aap`" is trivially predictable and is NOT what we want to find. The §5.6 confound controls are specifically designed to factor out pronoun-token detection from honorific-level encoding.

**Honorifics ≠ general formality (important distinction).** "Formality" in NLP often conflates politeness markers, register, lexical choice, and discourse appropriateness. Hindi grammatical honorific marking is more specific: it is a *morphosyntactically encoded* relation between speaker and addressee, with specific grammatical reflexes (pronoun + agreement). Two utterances can be matched on general formality but differ in honorific status (a formal news headline has no honorific status because it lacks second-person reference; a formal address to a superior does). Throughout this paper, *honorific* means the grammatically-marked phenomenon; *formality* means the broader discourse-pragmatic dimension. Our contrastive pairs (§5.2), confound controls (§5.6), and rubric (§5.9) are constructed to isolate them.

### 2.2 Japanese keigo — the comparative control

We use Japanese keigo as a *comparative reference case*, not a co-equal contribution. Japanese is the canonical honorific-rich language in linguistics; its multi-axis system has three categories:

- *Teineigo* — polite addressee-honoring (`desu/masu` endings)
- *Sonkeigo* — referent-honoring, elevating others
- *Kenjōgo* — referent-honoring, lowering self

Critically, Japanese encodes honorific status almost entirely in **verb suffix morphology**. This contrasts with Hindi's **pronoun + agreement** locus. The comparison answers a question that is interesting independently of either language: does Gemma's SAE structure encode honorifics in a *locus-aware* way (different features for different structural positions) or in an *abstract* way (a shared "politeness axis" that fires regardless of where the honorific information sits)?

We do NOT claim symmetric depth of treatment. Japanese in this paper is one set of probe + feature-identification experiments plus a high-level comparison plot. Hindi is the full causal-triangle study.

### 2.3 Sparse autoencoders — short primer

A sparse autoencoder (SAE) applied to a neural network's residual stream learns an over-complete dictionary of feature directions such that each forward-pass activation can be decomposed as a sparse positive combination of those features. The resulting features tend to be more *monosemantic* than the raw neurons (each feature often corresponds to a single coherent concept), which makes them an interpretability tool.

SAEs trained on Gemma-2 (Gemma Scope) are publicly released at multiple layers, sparsity levels, and feature counts. The key operation for this work is *feature steering*: identify a feature `f_i` whose activation correlates with a target property, then at inference time inject `+α · decoder_row_i` (additive steering) or `−α · decoder_row_i` (ablative steering) into the residual stream at the layer where the SAE was trained. This causally biases generation toward or away from the property without retraining.

---

## 3. Gap Analysis (the novelty claim)

The novelty claim rests on a specific gap: **no published work uses SAE-based or steering-based mechanistic interpretability to study honorific or register phenomena in any Indic language, and no work compares honorific encoding across languages with structurally different honorific loci (pronoun-driven vs. verbal-morphological).**

The closest prior work and its precise distance:

| Prior work | What it does | What it does NOT do (= our gap) |
|---|---|---|
| Deng et al., "Unveiling Language-Specific Features via SAEs" (arXiv:2505.05111, May 2025) | SAE features for language identity (EN/JA/ZH/...) | Treats each language as one label; no sub-language axes; no honorifics; no Indic |
| Liu et al., "Causal Language Control via Sparse Feature Steering" (arXiv:2507.13410, Jul 2025) | Steers EN→{ZH,JA,ES,FR} via SAE features on Gemma-2-2B/9B | Cross-language switching only; no within-language axes; no honorifics; no Indic |
| Park et al., "SAEs Capture Language-Specific Concepts" (arXiv:2507.11230, Jul 2025) | SAE-LAPE for language ID | Language-ID only; no register; no honorifics |
| Schut et al., "Lost in Multilinguality" (arXiv:2504.04264, Apr 2025) | Cross-lingual factual *inconsistency* | Factual, not stylistic; no honorifics; no Indic |
| Indic-NLP benchmarks (IndicXTREME, IndicGenBench, MILU) | Evaluation across many tasks and Indic languages | Evaluation only; no internals; no SAEs |
| Sarvam-1, Airavata, OpenHathi releases | Indic-focused LLMs | Model releases; no mechanistic analysis |
| Hindi `aap`/`tum`/`tu` work in NLP | Mostly translation evaluation | No internal representation studies; no SAEs |
| Yoo & Cho, "How LMs Understand Honorific Mismatches in Korean" (Lang. Res. 2024) | Behavioral probing of Korean honorific awareness | Behavioral only; no internals; no SAEs; not Indic |
| Korean SemEval 2025 Task 2 (arXiv:2504.20451) | Honorific MT error taxonomy | Behavioral only; no internals |
| Japanese keigo NLP studies | Mostly classification or rule-based generation | No representation-level analysis in modern LLMs |

**The precise gap (one sentence):** No work has used mechanistic interpretability to localize, characterize, or causally manipulate honorific representations *within* any Indic language, and no work has compared honorific encoding across typologically different honorific systems at the representation level.

**Why this gap exists:** Mechanistic interpretability has been overwhelmingly English-centric until ~12 months before the time of writing. The multilingual extension is fresh, and the early multilingual SAE work has focused on East Asian languages (Japanese, Chinese, Korean) because of their visibility in the field and the relative ease of finding native speakers in interp-research labs. Indic-NLP has matured as a community but has not yet intersected with mech-interp. This paper is at exactly that intersection.

### 3.1 What this paper is NOT claiming (deliberate humility)

To pre-empt reviewer skepticism about novelty inflation and over-reach common in mechanistic-interpretability work, we explicitly state what this paper does NOT claim:

- **NOT** the first multilingual SAE paper — Deng et al. (2025) and contemporaries hold that distinction
- **NOT** the first steering-vector paper — activation engineering and FUDGE-style control predate us
- **NOT** a claim that SAE features are perfectly monosemantic — we treat features as *candidates* and validate empirically
- **NOT** a claim that we have "solved" Hindi honorific generation — controlled-prompt steering is not the same as robust deployment behavior
- **NOT** a claim that the Japanese comparison establishes any universal cross-lingual honorific subspace — the comparison is *descriptive*, not a hypothesis test
- **NOT** a claim that Hindi is the only Indic language where this approach would work — it is the language the author is native in; Bengali, Tamil, Marathi, etc. are future work

What we *do* claim: a focused, controlled empirical demonstration that an existing public SAE contains identifiable features *associated with* measurable Hindi honorific shifts under both **additive intervention (steering)** and **ablation**, with confounds controlled, evaluation operationalized via a structured multi-axis rubric, and failure modes characterized. The Japanese comparison provides typological context, not a separate claim.

---

## 4. Research Questions and Hypotheses

### 4.1 Research Questions (primary = Hindi, comparison = Japanese)

**Primary (Hindi-focused):**

**RQ1 (Localizability, Hindi):** In Gemma-2-2B and Gemma-2-9B, does a linear probe trained on residual-stream activations reliably distinguish Hindi sentences by honorific level (`aap`-class vs `tum`/`tu`-class) AFTER controlling for pronoun-token presence and verb-agreement-token presence? At which layers does this succeed?

**RQ2 (Feature identification, Hindi):** Do Gemma Scope SAE features at the identified layer contain individual features whose activation correlates with Hindi honorific level *better* than they correlate with surface confounds (pronoun token, verb-suffix token, length, code-switched English material, script choice)?

**RQ3 (Full causal triangle, Hindi):** Does the discovered feature satisfy all three causal-triangle legs — correlation (probe), sufficiency (additive steering shifts honorific marking), and necessity (ablation reduces honorific marking on naturally-honorific prompts) — evaluated on the §5.9 structured rubric by native speakers, while staying within publishable fluency and content-preservation bounds?

**Comparative (Japanese as control, lighter-weight):**

**RQ4 (Japanese probe analog):** Does the same probe+SAE-feature-identification pipeline, applied to Japanese keigo (`desu/masu` vs. plain form), find honorific features at comparable layers and with comparable selectivity? Does the *layer* of best probe performance differ between languages?

**RQ5 (Structural locus comparison, descriptive):** Are the candidate features found for Hindi and Japanese qualitatively different — e.g., do Hindi features show stronger activation on pronoun positions while Japanese features show stronger activation on sentence-final positions? This is a *descriptive comparison*, not a hypothesis test.

### 4.2 Hypotheses (sharp for Hindi; descriptive for Japanese)

**H1 (Hindi, primary):** A linear probe at some middle layer (predicted band: 8-16 in Gemma-2-2B) will achieve AUROC > 0.70 on the *confound-stripped* Hindi honorific classification task. We deliberately set this threshold lower than the Japanese-primary version (0.75) because Gemma's Hindi pretraining exposure is lower than its Japanese exposure; a 0.70 floor still represents strong signal above chance.

**H2 (Hindi, primary):** At least one Gemma Scope SAE feature at the H1-identified layer will show (a) effect size d > 0.5 between `aap`-class and `tum`/`tu`-class, and (b) selectivity above 0.7 when controlled for the confounds enumerated in §5.6 (pronoun token, agreement suffix, length, script).

**H3 (Hindi, primary, full causal triangle):** Across additive steering, ablation, and encode-side feature masking, the candidate feature(s) will produce a rubric-rated honorific shift in the predicted direction with PPL inflation < 1.5× and content-preservation BLEURT drop < 0.05. The necessity arm (ablation) is the strictest test — failure here would convert the contribution to "feature is sufficient but not necessary," which itself is a publishable refinement.

**H4 (Japanese, comparison, exploratory):** We expect to replicate prior intuitions that Japanese probes succeed at higher AUROC and lower layer counts than Hindi (because Japanese honorific morphology is more frequent in Gemma's pretraining). We make NO directional prediction about cross-lingual feature similarity.

**H5 (Structural locus, descriptive):** We expect — but do not require for the paper's contribution — to observe that Hindi candidate features activate more strongly near pronoun and agreement positions, while Japanese features activate more strongly near sentence-final verb-suffix positions. If observed, this is one interpretive finding among others. If not observed, it suggests the model has discovered a more abstract honorific axis.

The original v1/v2/v3 cross-lingual cosine-similarity hypothesis is deliberately removed — steering-vector cosine across languages is too confounded by language-ID, tokenization, and layer-realignment effects to support a directional claim.

---

## 5. Proposed Approach (Methodology Overview)

### 5.1 Models

| Model | Role | Memory (bf16) | SAE availability |
|---|---|---|---|
| Gemma-2-2B | Primary probing + steering + ablation | ~5 GB | Gemma Scope (all layers) |
| Gemma-2-9B | Scale-up validation | ~18 GB | Gemma Scope (all layers) |
| Qwen2.5-1.5B-Instruct | Control (multilingual coverage check) | ~3 GB | Train one small SAE if needed |
| Sarvam-1 (2B, Indic-focused, Apache-2.0) | Control (Indic-saturated training) | ~5 GB | None (probe-only) |

**Why this set:**

- Gemma Scope SAEs are publicly released — eliminates multi-week SAE training (load-bearing for the project's feasibility on a 24GB GPU)
- Sarvam-1 is included as a probe-only control: if Hindi honorific signal is much stronger in an Indic-saturated model, that itself is a useful comparative finding
- Gemma-2-9B is the strongest model that fits comfortably in 24GB with bf16 + activations + SAE forward

**Memory feasibility — Gemma-2-9B + Gemma Scope SAE on 24GB Ada:**

| Component | Memory |
|---|---|
| Model weights bf16 | ~18 GB |
| Activations (batch 1, seq 512) | ~3 GB |
| SAE forward (16k features, layer 20) | ~0.5 GB |
| CUDA overhead | ~1.5 GB |
| **Total** | **~23 GB — fits with margin** |

### 5.2 Datasets

**New construction (the primary releasable artifact):**

- ~600 minimally-contrastive Hindi sentence pairs
- Each pair: same propositional content, two distinct honorific levels (`aap` vs `tum`, `aap` vs `tu`, `tum` vs `tu`)
- **Author + one additional Hindi-native rater** do annotation directly — no Prolific budget required; estimated 1-2 weeks
- Krippendorff's α reported for inter-annotator agreement
- Per-item confound labels (length-matched, pronoun-stripped, code-switched control, script variant) — see §5.6
- Released CC-BY-SA

**Japanese comparison set (smaller, ~200 pairs):**

- Sourced from existing Japanese honorific datasets (JESC speech-level subsets, KEIGO corpora) plus minimal new construction
- Annotated by 1-2 Japanese-native raters via Prolific (estimated ~$150)

**Existing sources for downstream evaluation:**

- Hindi: HiDial dialogue corpus; IIT-Bombay parallel; AI4Bharat Aksharantar (for transliteration confound material)
- Japanese: JESC (CC-BY); CSJ register-annotated subsets

### 5.3 Pipeline (4 stages, all on Hindi first)

**Stage 1 — Probing (weeks 1-3):**
Layer-wise linear probes (logistic regression with L2) for binary honorific classification (`aap`-class vs `tum`/`tu`-class) on Hindi contrastive pairs. Probe twice: (a) on raw input, (b) on the confound-stripped condition (§5.6). Identify the layer with best confound-stripped AUROC.

**Stage 2 — SAE feature discovery (weeks 4-6):**
Run Gemma Scope SAEs at the identified layer. Rank features by (a) effect size, (b) selectivity against confound dimensions, (c) auto-interpretability score. Manually validate top-20 features against held-out examples.

**Stage 3 — Causal triangle (weeks 7-9):**
For the top-3 candidate features, run all three causal-triangle conditions (correlation/sufficiency/necessity — see §5.8). Compare against the baselines in §5.4.

**Stage 4 — Japanese comparison (weeks 10-11):**
Repeat Stage 1 + Stage 2 (only) on Japanese keigo contrastive pairs. Report layer and selectivity differences. No causal-triangle replication on Japanese — it's a *comparison*, not a co-equal claim. This bounds the scope (one paper, not two).

### 5.4 Baselines

| Baseline | Purpose |
|---|---|
| Zero-shot prompting in Hindi (`aap`-form instruction) | Standard control |
| 4-shot prompting | Closes gap with steering on simple cases |
| LoRA fine-tuning on the contrastive set | Strongest learnable baseline |
| FUDGE-style classifier-guided decoding (with a Hindi honorific classifier) | Best inference-time prior method |
| Random-feature steering | Null distribution |

### 5.5 Statistical analysis plan

- Mixed-effects models with random intercepts for prompt and rater
- Bonferroni correction across the rubric axes
- Bootstrap 95% CIs on all effect sizes
- N ≥ 200 generations per condition; 2-3 raters per generation; 5 seeds for LoRA baselines
- RQs and hypotheses pre-registered (see §8) to limit p-hacking

### 5.6 Confounds and Controls (the part reviewers will press on)

The single biggest threat is the reviewer comment "you're not measuring honorifics, you're measuring [confound]". We pre-empt each:

| Confound | What it would falsely look like | Control |
|---|---|---|
| **Pronoun token detection** (`aap` / `tum` / `tu` are highly informative single tokens) | A feature that fires only when one of these pronouns appears | Build a *pronoun-stripped probe set* where the pronoun is masked or removed, leaving only verb agreement and surrounding context. Report AUROC on stripped set separately. This is the single most important Hindi-specific control. |
| **Verb agreement suffix detection** (plural endings co-occur with `aap`) | Feature picks up the suffix, not the honorific status | Stratify probe set by verb tense/aspect; report per-stratum AUROC; include items where pronoun is present but agreement is incorrect (testing whether the feature tracks the *expected* honorific status vs. the *surface* suffix) |
| **Length confounds** (formal Hindi tends longer; `aap` is a longer token sequence) | Probe learns length, not honorifics | Length-matched contrastive pairs; report length-only baseline |
| **Code-switching / English insertion** (Hindi-English code-mixing is very common in real Hindi) | Feature confounds "English-mixed" with "informal" or vice versa | Include a code-switched control condition; report stratified results |
| **Script choice** (Devanagari vs. Roman/Hinglish) | Feature picks up script identity, not honorifics | Include both Devanagari and Roman-script versions of every pair (transliterated via `indic-trans`); report stratified results |
| **Topic / register entanglement** (formal topics tend honorific) | Feature confounds topic with honorifics | Balanced topic distribution; per-topic effect reporting |
| **Language-ID leakage** (cross-comparison with Japanese) | A "Hindi honorific" feature is really a "Hindi language" feature | Compare against existing language-ID features from Liu et al. (2025); report Hindi-honorific effects ABOVE the language-ID baseline |
| **Annotator disagreement on edge cases** | Honorific judgments vary on context | 2-3 native annotators per item; Krippendorff's α reported; items below threshold analyzed separately |
| **SAE feature mixing** (one feature ≠ one concept) | A "candidate honorific feature" actually encodes multiple things | Auto-interpretability validation: top-activating examples described by a separate LM; manual native-speaker validation on top-20 features |
| **Cherry-picked α** in steering | Effects only appear at carefully-tuned strengths | Pre-register α-selection procedure (held-out validation); report α-sweep curves |

### 5.7 Minimum Publishable Unit (the killer first experiment)

**The MPU experiment:**
- **One language:** Hindi
- **One model:** Gemma-2-2B
- **One layer:** the layer with highest pronoun-stripped probe AUROC
- **One feature:** the top-ranked SAE feature by effect size + selectivity
- **Three steering conditions:** (a) +α additive steering, (b) −α ablative steering, (c) encode-side zero-ablation — the full causal triangle (§5.8)
- **Rubric-based evaluation:** 150-200 native-speaker ratings (author + 1-2 collaborators) on the §5.9 four-axis Hindi honorific rubric

**Go criteria (need all four to commit to the full paper):**
- Pronoun-stripped probe AUROC > 0.70 ✓
- ≥ 1 SAE feature with d > 0.5 and selectivity > 0.6 ✓
- Additive steering produces rubric-rated addressee-honorificity shift (p < 0.01) with PPL inflation < 1.5× ✓
- Ablation produces measurable drop in addressee-honorificity on naturally `aap`-form prompts (necessity arm) ✓

**If MPU fails** (no clean probe, no usable feature, or no causal effect): pivot to "Hindi honorific representation in Gemma Scope SAEs is distributed and resists feature-localized intervention — implications for Indic-NLP mechanistic interpretability." Still publishable at SRW; the negative result is itself a first.

**Timeline:** MPU is ~4 weeks. Decision point on whether to add Japanese comparison and 9B scale-up at week 4.

### 5.8 Closing the causal loop: correlation + steering + ablation

The standard reviewer attack on SAE-based causal claims is "you showed a correlation and a steering effect — but did you show the feature is *necessary*?" We include all three legs:

| Leg | What it shows | Operationalization |
|---|---|---|
| **Correlation** (probing + SAE attribution) | Feature covaries with honorific level | Per-layer probe AUROC; per-feature effect size + selectivity (with §5.6 controls) |
| **Sufficiency** (additive steering) | Activating the feature *increases* honorific marking | `+α · decoder_f` injection; rubric-rated honorific shift vs. baseline |
| **Necessity** (ablation) | Suppressing the feature *decreases* honorific marking on naturally honorific input | `−α · decoder_f` AND encode-side zero-ablation; rubric-rated drop on held-out naturally-`aap`-form prompts |

The ablation arm is what most SAE-steering papers omit. Including it converts the claim from "we can push the model toward honorific output" to "this feature is causally implicated in honorific generation" — meaningfully stronger with the same compute budget.

If correlation + sufficiency hold but ablation fails, the paper honestly reports the asymmetry ("sufficient but not necessary"), itself a citable refinement of mech-interp claims.

### 5.9 Structured Hindi-specific honorific evaluation rubric

A single "formality" rating collapses several distinct phenomena. We use a multi-axis rubric, Hindi-specialized, applied per-generation by Hindi-native raters:

| Axis | What it measures | Scale |
|---|---|---|
| **Addressee honorificity** | Is the second-person pronoun appropriate (`aap` / `tum` / `tu`)? | 0-3 ordinal |
| **Verb agreement consistency** | Does the verb agreement match the pronoun's grammatical number (plural verb with `aap`)? | 0-3 ordinal |
| **Register consistency across the generation** | Does the honorific level stay constant through the whole output, or slip? | 0-3 ordinal |
| **Context appropriateness** | Given the prompt's implied social context, is the chosen honorific level pragmatically right (not over- or under-shooting)? | 0-3 ordinal |
| **Fluency (control)** | Independent fluency rating | 0-3 ordinal |
| **Content preservation (control)** | Does the generation answer the prompt? | 0-3 ordinal |

This rubric (a) decouples grammatical honorific marking from general formality, (b) makes failure modes visible (e.g., correct pronoun + wrong agreement is a known LLM error mode that needs to be measured separately), and (c) pre-empts the "underdefined metric" rejection. The rubric, training materials, and item-level annotations are released as a secondary artifact.

For the Japanese comparison work (Stage 4), a simpler 2-axis rubric (addressee honorificity, fluency) is applied — consistent with the lighter-weight comparison role.

---

## 6. Expected Contributions

1. **A controlled mechanistic case study of Hindi honorific encoding** in publicly-released SAEs, with the full causal triangle (correlation + sufficiency + necessity) — the first mech-interp study of any Indic-language sociolinguistic phenomenon, to the best of our knowledge.
2. **A confound-controlled Hindi honorific dataset (~600 pairs)** with explicit per-item confound labels (pronoun-stripped, length-matched, script variant, code-switching control), released CC-BY-SA and annotated by Hindi-native authors.
3. **A Hindi-specialized multi-axis honorific evaluation rubric** (addressee / agreement / register / context) with rater training materials — distinguishing grammatical honorific marking from general formality.
4. **A catalog of candidate honorific-associated SAE feature indices** for Gemma Scope on Gemma-2-2B and Gemma-2-9B, with selectivity scores and ablation effects, usable for follow-up work on Indic mech-interp.
5. **A reproducible inference-time steering + ablation pipeline** runnable on a single 24GB consumer GPU.
6. **A comparative descriptive analysis with Japanese keigo** — the first Hindi-vs-Japanese mech-interp comparison of honorific encoding, providing typological context without making strong cross-lingual identity claims.

Even if the primary causal-loop hypotheses fail, the paper has a publishable contribution: "Hindi honorific representations in current public multilingual SAEs are distributed, partially confounded with surface markers (pronoun tokens, agreement suffixes), and resist single-feature ablation" is itself a novel, citable finding for Indic-NLP mech-interp.

---

## 7. Why AACL SRW (Venue Fit)

**AACL 2026 main-conference priority alignment:**

- *Multilinguality and Language Diversity* — direct hit, with Indic focus
- *Low-resource Methods for NLP* — Indic-NLP framing
- *Interpretability, Model Editing, Transparency* — direct hit
- *Safety and Alignment in LLMs* — honorific control is alignment-adjacent (mis-leveling honorifics is a social-harm vector specifically relevant to Indian deployments)
- *Human-AI Interaction* — honorifics are central to HCI in Hindi-speaking contexts

**AACL SRW paper-shape match:**

- 8-page focused contribution
- Small-to-mid-scale model regime (1.5B-9B) — matches the historical AACL SRW model-scale distribution (no AACL 2020/2022 SRW paper used a model > 1B)
- Indic-language phenomenon (Hindi) with comparison to East Asian (Japanese) — broad Asian-language reach with focused primary claim
- Linguistically motivated (honorifics, register, agreement)
- Releasable artifacts (data + rubric + feature indices + code)

**Structural advantages from author profile:**

- Indian author working on Indian-language interpretability → reviewer pool recognizes lived linguistic intuition behind the claims
- Indic + interpretability is a near-empty intersection — high "fresh angle" signal
- AACL SRW historically has strong Indic-language participation (4 Hindi-related papers in AACL 2020 SRW alone)

**Closest analogues in prior AACL SRW (papers that *got in* with similar shape):**

- Tater et al., AACL 2022 — "Concreteness vs. Abstractness: A Selectional Preference Perspective"
- Khullar, AACL 2020 — "Exploring Statistical and Neural Models for Noun Ellipsis Detection"
- Pant & Dadu, AACL 2020 — "Towards Code-switched Classification Exploiting Constituent Language Resources" (Hindi-English code-switching, AACL-perfect framing)

Each took a linguistic phenomenon and applied modern modeling tooling — exactly the pattern this work follows, with 2025 SAE infrastructure and an Indic primary focus.

---

## 8. Novelty Validation Checklist

If you're a reviewer/advisor/friend evaluating novelty, run these checks:

- [ ] Google Scholar: `"sparse autoencoder" Hindi` — expected: 0 directly relevant hits
- [ ] Google Scholar: `"sparse autoencoder" honorific` — expected: 0 directly relevant hits
- [ ] arXiv: `steering vector Hindi` OR `steering vector Indic` — expected: 0 directly relevant hits
- [ ] arXiv: `SAE Indic` OR `mechanistic interpretability Hindi` — expected: 0 directly relevant hits
- [ ] OpenReview (ICLR/ACL/NAACL/EMNLP 2024-2026): `honorific interpretability Indic` — expected: 0 directly relevant hits
- [ ] Check whether Gemma Scope multilingual SAE papers cite any Indic-NLP work — expected: no
- [ ] Ask three NLP researchers (one Indic-NLP, one interp, one general): "do you know any work using SAEs on Hindi sociolinguistic phenomena?" — expected: no
- [ ] Check whether the proposed Hindi `aap`/`tum`/`tu` contrastive dataset format duplicates any existing resource — expected: no SAE-confound-controlled version exists

**If any check turns up a paper within ~6 months of submission window, re-evaluate before submitting.** Re-run monthly until camera-ready.

---

## 9. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| No clean Hindi honorific direction exists (H1 fails) | Medium | Pivot to "Hindi honorifics are distributed in current SAEs — implications for Indic mech-interp" — novel and citable as a constraint |
| Gemma's Hindi pretraining exposure is too low for usable signal | Medium | Sarvam-1 control: if signal is much stronger in an Indic-saturated model, the comparison itself is the finding |
| Annotation bottleneck (Hindi self-annotation creates author bias) | Low-Medium | One additional Hindi-native rater for inter-annotator agreement; release annotation guidelines publicly |
| SAE features too noisy | Medium | Use Gemma Scope JumpReLU (most interpretable SAE family); fall back to PCA directions from probe weights |
| Concurrent submission appears | Low-Medium | Re-run §8 checklist monthly; pivot to a more specific framing if needed |
| Steering effects too small to matter | Medium | Frame as benchmark + mechanistic characterization; SRW reviewers receptive to honest negative results |
| Japanese comparison gets ignored by reviewers | Low | Frame is descriptive; it's a credibility anchor, not load-bearing |
| Reviewer 2 lottery | Inherent | Use the SRW mentorship program (deadline 2026-06-08) |

### 9.1 Failure Modes & Boundary Conditions (mandatory paper section)

The paper will include an explicit section on where the method *fails*. Anticipated failure modes to investigate and report:

1. **Steering at very high α destroys fluency** — characterize the cliff; report the operating range
2. **Steering fails on long generations** — repeat steering loses effect over multiple tokens; report effect decay curves
3. **Steering fails when verb agreement and pronoun choice conflict** in the input — does the feature track the pronoun or the agreement?
4. **Some honorific features confound with pronoun-token detection** even after §5.6 controls — report which features are cleanest
5. **Cross-script robustness** — does the feature behave the same on Devanagari vs. Roman Hindi inputs? Report stratified results
6. **Code-switching breakdown** — when the input is heavily code-mixed with English, does the steering still work?
7. **Annotator disagreement on intermediate cases** (e.g., `tum` is genuinely ambiguous between familiar-respectful and casual)
8. **Native-speaker raters disagree with classifier scores** — when they disagree, report both; do not silently default

Including this explicit failure-mode section converts mixed results from a rejection risk into a reviewer-confidence signal.

---

## 10. First-Week Action Plan

The first week derisks the **Minimum Publishable Unit** (§5.7) — Hindi only, single feature, single steering condition.

| Day | Action |
|---|---|
| 1 | Read in order: Gemma Scope paper (Lieberum et al., 2024); arXiv:2507.13410 (Liu et al., causal language steering on Gemma-2-2B/9B). Output: 1-page lit-map situating this work against them specifically |
| 2 | Read arXiv:2505.05111 (Deng et al.) and arXiv:2504.04264 (Schut et al., Lost in Multilinguality). Identify exact layers and SAE configurations they use; match infrastructure to reduce one variable |
| 3 | Install `sae_lens`, `transformer_lens`, `transformers`, `datasets`. Load Gemma-2-2B + a Gemma Scope mid-layer residual SAE. **Smoke test:** reproduce one figure from arXiv:2507.13410. Verify 24GB headroom for Gemma-2-9B + SAE forward |
| 4 | Draft **30 Hindi minimally-contrastive pairs** (10 `aap` vs `tum`, 10 `aap` vs `tu`, 10 `tum` vs `tu`) with deliberate confound balance: 10 pronoun-stripped variants, 10 length-matched, 10 script-variant pairs. Validate naturalness on yourself + one other Hindi speaker if available |
| 5 | **MPU pilot — probe stage:** Run layer-wise linear probe on the 30-pair set across all Gemma-2-2B residual layers. Report raw AUROC and pronoun-stripped AUROC. **Go/no-go:** pronoun-stripped AUROC > 0.65 at any layer → green light |
| 6 | **MPU pilot — feature stage:** At the best layer, run Gemma Scope SAE; rank features by effect size + selectivity. Compute selectivity against confound dimensions for top-3. If at least one feature has d > 0.4 and selectivity > 0.6 → green light steering experiments |
| 7 | Sign up for the **AACL SRW pre-submission mentorship** (deadline 2026-06-08 — this is the single highest-leverage move). Write a 2-page pre-registration document from this PROBLEM_STATEMENT.md. Share with advisor; lock co-authors |

**End-of-week-1 decision:**
- All three pilot signals green → proceed to full MPU steering + ablation experiments (weeks 2-4), then Japanese comparison (weeks 10-11)
- Any signal red → diagnose (data quality? layer choice? Sarvam-1 instead of Gemma?)
- Two consecutive failure diagnoses → pivot to null-result framing (still publishable)

---

## 11. Honest Caveats

- This work does *not* claim to "solve" Hindi honorific generation. It **characterizes and localizes** an internal representation — a different and more tractable contribution.
- No peer-reviewed work is ever guaranteed accepted. This design maximizes *expected* acceptance probability through clear contribution + clean execution + perfect venue fit + multiple defensible fallback modes (positive result, null result, benchmark artifact).
- The compute budget (~120 GPU-hours on a single 24GB Ada) is tight but realistic. The Japanese comparison adds ~30 GPU-hours.
- Self-annotation by the author introduces a known bias; mitigated by the additional rater and public release of guidelines. The trade-off (free + fast + culturally authoritative vs. potential bias) favors moving forward given SRW's typical scale.
- Korean is deliberately dropped from this scope (vs. earlier versions) to keep the paper a *single primary claim* with one comparative reference. Future work could extend to Korean as a third typological point.

---

## 12. References (selected)

**Mechanistic interpretability / multilingual:**

- Lieberum, T. et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All at Once on Gemma 2*. arXiv:2408.05147
- Schut, L. et al. (2025). *Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency*. arXiv:2504.04264
- Deng, B. et al. (2025). *Unveiling Language-Specific Features in LLMs via Sparse Autoencoders*. arXiv:2505.05111
- Liu, X. et al. (2025). *Causal Language Control in Multilingual Transformers via Sparse Feature Steering*. arXiv:2507.13410
- Park, J. et al. (2025). *Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages*. arXiv:2507.11230
- Templeton, A. et al. (2024). *Scaling Monosemanticity*. Anthropic
- Anonymous (2025). *Mechanistic Understanding and Mitigation of Language Confusion*. arXiv:2505.16538
- Anonymous (2025). *Language-Specific Latent Process Hinders Cross-Lingual Performance*. arXiv:2505.13141

**Indic NLP / Hindi:**

- AI4Bharat — IndicTrans2, Aksharantar (transliteration); IndicXTREME, IndicGenBench (benchmarks)
- Sarvam-1 (2B, Apache-2.0, Indic-focused) — model release
- Airavata, OpenHathi, Krutrim — Indic LLMs
- "Tokenization Disparities as Infrastructure Bias" (arXiv:2510.12389, Oct 2025) — Indic tokenization
- "Script Gap: Evaluating LLM Triage on Indian Languages" (arXiv:2512.10780, Dec 2025)
- "Modeling Romanized Hindi and Bengali" (arXiv:2511.22769, Nov 2025)
- IndicSuperTokenizer (arXiv:2511.03237, Nov 2025)

**Honorifics / register (East Asian comparison material):**

- Korean SemEval 2025 Task 2 (arXiv:2504.20451) — honorific MT error taxonomy
- Yoo & Cho (2024). *How LMs Understand Honorific Mismatches in Korean*. Language Research
- MDPI (2021). *Neural MT Strategies for Generating Honorific-style Korean*
- Japanese keigo NLP literature (heterogeneous; cited inline in paper)

**Steering / activation engineering:**

- Anonymous (2025). *Improving Multilingual via Steering*. arXiv:2505.12584
- Anonymous (2025). *Steering into New Embedding Spaces*. arXiv:2502.15639
- FUDGE (Yang & Klein, 2021) — classifier-guided decoding baseline

**Datasets:**

- JESC — Japanese-English Subtitle Corpus
- HiDial — Hindi Dialogue Corpus
- IIT-Bombay Hindi-English Parallel Corpus

**Venue:**

- AACL-IJCNLP 2026 SRW Call for Papers — https://2026.aaclnet.org/calls/srw/
- AACL-IJCNLP 2022 SRW Proceedings — https://aclanthology.org/volumes/2022.aacl-srw/
- AACL-IJCNLP 2020 SRW Proceedings — https://aclanthology.org/volumes/2020.aacl-srw/

---

*End of problem statement v4. Share with advisor + 1-2 NLP researchers (one Indic-NLP, one interp) for novelty validation. Re-run §8 checklist before final submission. Begin §10 first-week MPU pilot to derisk before committing further effort.*
