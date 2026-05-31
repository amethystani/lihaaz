# tests/data/test_annotate.py
import pytest


@pytest.mark.unit
def test_krippendorff_alpha_perfect_agreement():
    """Integration: real krippendorff call on two identical ordinal raters -> 1.0."""
    from src.data.annotate import krippendorff_alpha

    data = [[1, 2, 3, 1, 2], [1, 2, 3, 1, 2]]
    assert krippendorff_alpha(data) == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "stub_alpha, expected_status",
    [
        (0.85, "strong"),
        (0.80, "strong"),  # boundary
        (0.70, "moderate"),
        (0.67, "moderate"),  # boundary
        (0.50, "weak — review disagreements before proceeding"),
    ],
)
def test_report_iaa_status_banding(monkeypatch, stub_alpha, expected_status):
    """Status banding is deterministic when krippendorff_alpha is stubbed."""
    import src.data.annotate as annotate

    monkeypatch.setattr(annotate, "krippendorff_alpha", lambda *a, **k: stub_alpha)

    result = annotate.report_iaa([[1, 2], [1, 2]])

    assert result["status"] == expected_status
    assert result["alpha"] == stub_alpha


@pytest.mark.unit
def test_report_iaa_end_to_end_perfect_agreement():
    """End-to-end: perfect agreement -> strong status and alpha == 1.0."""
    from src.data.annotate import report_iaa

    result = report_iaa([[1, 2, 3, 1, 2], [1, 2, 3, 1, 2]])

    assert result["status"] == "strong"
    assert result["alpha"] == pytest.approx(1.0)
