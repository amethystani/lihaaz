import pytest

from src.baselines.prompting import zero_shot_prompt, four_shot_prompt


@pytest.mark.unit
def test_zero_shot_prompt_aap_level_and_sentence():
    # Arrange
    sentence = "tum kaise ho?"

    # Act
    prompt = zero_shot_prompt(sentence, "aap")

    # Assert
    assert "aap-rup (formal)" in prompt
    assert sentence in prompt


@pytest.mark.unit
def test_zero_shot_prompt_tum_level():
    # Arrange
    sentence = "aap kahan ja rahe hain?"

    # Act
    prompt = zero_shot_prompt(sentence, "tum")

    # Assert
    assert "tum-rup" in prompt
    assert sentence in prompt


@pytest.mark.unit
def test_zero_shot_prompt_tu_level_and_sentence():
    # Arrange
    sentence = "aap kaise hain?"

    # Act
    prompt = zero_shot_prompt(sentence, "tu")

    # Assert
    assert "tu-rup (informal)" in prompt
    assert sentence in prompt


@pytest.mark.unit
def test_four_shot_prompt_aap_includes_all_exemplars_and_final_cue():
    # Arrange
    sentence = "tu kya kar raha hai?"
    exemplars = [
        ("tum kaise ho?", "aap kaise hain?"),
        ("tera naam kya hai?", "aapka naam kya hai?"),
        ("tum kahan ja rahe ho?", "aap kahan ja rahe hain?"),
        ("tumhara kaam ho gaya.", "aapka kaam ho gaya."),
    ]

    # Act
    prompt = four_shot_prompt(sentence, "aap")

    # Assert
    for src, tgt in exemplars:
        assert f"Original: {src}\naap-rup: {tgt}" in prompt
    # ends with the new sentence followed by the trailing cue and no target
    assert prompt.endswith(f"Original: {sentence}\naap-rup:")


@pytest.mark.unit
def test_four_shot_prompt_tum_absent_key_yields_zero_exemplar_prompt():
    # Arrange
    sentence = "aap kaise hain?"

    # Act
    prompt = four_shot_prompt(sentence, "tum")

    # Assert: tum is not in FOUR_SHOT_EXAMPLES, so only the final block remains
    assert prompt == f"Original: {sentence}\ntum-rup:"


@pytest.mark.unit
def test_module_imports_without_torch_installed():
    # Arrange / Act: importing the module must not require torch in this venv
    import importlib

    module = importlib.import_module("src.baselines.prompting")

    # Assert: torch must NOT have been imported as a side effect of import
    import sys

    assert "torch" not in sys.modules
    assert hasattr(module, "generate_with_prompt")
