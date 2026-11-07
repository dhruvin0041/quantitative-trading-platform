"""
Institutional Automated Daily Execution Runner.
Re-exported from backend/execution/paper_runner.py for package integration.
"""

import importlib.util
from pathlib import Path

_impl_path = Path(__file__).resolve().parent.parent.parent / "execution" / "paper_runner.py"
_spec = importlib.util.spec_from_file_location("_paper_runner_impl", str(_impl_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DailyPaperRunner = _mod.DailyPaperRunner

__all__ = ["DailyPaperRunner"]
