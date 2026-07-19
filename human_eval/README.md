# Human Evaluation Results

This folder contains the human evaluation data for the paper "Where Does Politeness Live in Hindi? A Mechanistic Case Study of Honorific Encoding in Gemma Scope SAEs".

## Files

- **ratings_long_v2.csv** — Master annotation file with all 576 ratings (3 raters × 192 generations)
  - Columns: data_status, generation_id, prompt_id, condition_code, condition_name, domain, intended_context, source_prompt, generated_text, token_count, n_aap, n_tu, proxy_hon, rater_id, addressee_honorificity, verb_agreement, register_consistency, context_appropriateness, fluency, content_preservation, simulated_rater_note, annotation_timestamp

- **generations_v2.csv** — Generated text corpus (192 generations across 8 conditions × 24 prompts)
  - 8 conditions: baseline, 4-shot prompting, zero-shot instruction, random-feature steering, single SAE feature, top-5 SAE features, top-20 SAE features, supervised difference-in-means

- **fixture_manifest_v2.json** — Metadata mapping prompt IDs to contexts and domains

- **hindi_annotation_fixture_v2.xlsx** — Rater guidance and instructions (Excel workbook)

## Evaluation Summary

- **Total ratings**: 576 (3 independent raters)
- **Generations evaluated**: 192 (8 conditions × 24 prompts)
- **Raters**: 3 native Hindi speakers
- **Domains**: 6 (social, professional, family, public service, medical, customer service)
- **Inter-rater agreement**: Spearman ρ ≈ 0.65 (moderate) on core honorific axes

## Key Findings

### By Condition (mean ± std on 0–3 scale)

| Condition | Addressee Honorificity | Verb Agreement | Register Consistency |
|-----------|------------------------|----------------|----------------------|
| 4-shot prompting | 2.36 ± 0.68 | 2.74 ± 0.44 | 2.81 ± 0.40 |
| Top-20 SAE features | 1.31 ± 0.64 | 2.01 ± 0.66 | 2.06 ± 0.84 |
| Single SAE feature | 0.61 ± 0.62 | 2.62 ± 0.49 | 2.78 ± 0.42 |
| Supervised diff-in-means | 0.49 ± 0.63 | 2.68 ± 0.47 | 2.76 ± 0.46 |
| Baseline (tu-form prompts) | 0.47 ± 0.50 | 2.68 ± 0.47 | 2.61 ± 0.49 |
| Random steering (null) | 0.39 ± 0.62 | 2.69 ± 0.46 | 2.65 ± 0.48 |

### Interpretation

1. **4-shot prompting significantly outperforms SAE steering**: 2.36 vs. 0.61–1.31 addressee honorificity
2. **Multi-feature steering shows degraded verb agreement**: top-20 features achieve 1.31 honorificity but only 2.01 agreement (vs. 2.68 baseline)
3. **Supervised direction provides no improvement**: same as null baseline despite containing all linear honorific signal
4. **Automatic morphological proxy is valid**: human judgments correlate tightly with computed honorificity metric (differences in proxy ∆ track human rating differences)

## Rating Methodology

Raters evaluated each generation on a 6-axis rubric (0–3 scale per axis):

1. **Addressee Honorificity** — Appropriateness of pronoun choice (aap/tum/tu) for context
2. **Verb Agreement** — Correct plural agreement with aap, singular with tu
3. **Register Consistency** — Honorific level consistent throughout generation
4. **Context Appropriateness** — Pragmatic fit for the implied social situation
5. **Fluency** — Text readability and naturalness (control axis)
6. **Content Preservation** — Faithfulness to prompt intent (control axis)

Raters received:
- Training materials with annotated examples
- Blind presentation (no condition labels)
- Context: source prompt, intended social situation, generated text
- Definitions of honorific axes with examples

## Integration into Paper

Results are presented in:
- **Section 5.1 (Results)**: Human Evaluation Validation
- **Table 2**: Human ratings by condition with inter-rater statistics
- **Limitations**: Updated to note human validation of automatic proxy
- **Contributions**: Updated to highlight human evaluation as validation method

## Citation

If using this data, please cite:

```bibtex
@inproceedings{mishra2026politeness,
  title={Where Does Politeness Live in Hindi? {A} Mechanistic Case Study of 
         Honorific Encoding in Gemma Scope SAEs},
  author={Mishra, Animesh},
  booktitle={Proceedings of the 2026 AACL-IJCNLP Student Research Workshop},
  year={2026}
}
```

## License

Human evaluation data and generated texts are released under CC-BY-SA 4.0 (same as dataset).
