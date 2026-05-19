"""
services/pose_suggester.py
----------------------------
Thin wrapper around Phase 3's PoseSuggester,
resolving library paths relative to the backend directory.
"""

import sys
from pathlib import Path

# Allow importing from phase3 directly
PHASE3_DIR = Path(__file__).parent.parent.parent / "phase3_suggester"
sys.path.insert(0, str(PHASE3_DIR))

from suggest import PoseSuggester  # noqa: E402

# Override library path to point at backend's local copy
from suggest import LIBRARY_DIR as _DEFAULT_LIBRARY  # noqa: E402
import suggest as _suggest_module

_suggest_module.LIBRARY_DIR = Path(__file__).parent.parent / "pose_library"

# Singleton
_suggester_instance: PoseSuggester | None = None


def get_pose_suggester() -> PoseSuggester:
    global _suggester_instance
    if _suggester_instance is None:
        _suggester_instance = PoseSuggester()
    return _suggester_instance