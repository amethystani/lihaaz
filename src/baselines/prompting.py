# src/baselines/prompting.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    import torch
    tokens = model.to_tokens(prompt).to(device)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    return model.to_string(out[0, tokens.shape[1]:])
