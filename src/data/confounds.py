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
