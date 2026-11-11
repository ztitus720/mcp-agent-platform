"""Make helpers in tests/ importable from other test files (e.g. ScriptedModel)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
