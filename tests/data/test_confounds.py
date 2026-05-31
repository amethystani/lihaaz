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
