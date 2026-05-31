"""Pytest configuration: ensure the repository root is importable.

Placing this at the repo root lets test modules use absolute imports such as
``from src.data.pairs import HonorificPair`` without installing the package.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
