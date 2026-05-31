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
