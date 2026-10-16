"""Shared test setup: import from src/, and stub cmdstanpy so fit.py imports without CmdStan."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import cmdstanpy  # noqa: F401
except ImportError:
    stub = types.ModuleType("cmdstanpy")
    stub.CmdStanModel = object
    sys.modules["cmdstanpy"] = stub
