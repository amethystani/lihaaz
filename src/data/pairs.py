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
