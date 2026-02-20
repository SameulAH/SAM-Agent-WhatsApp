"""
tests/integration/conftest.py

Ensures the project root is in sys.path before any integration test imports
trigger the webhook → transport → config import chain.

This is needed because pytest's rootdir/pythonpath settings can race with
fixture-time imports on some Python versions (3.12+).
"""
import sys
from pathlib import Path

# Project root = two levels up from this file (tests/integration/conftest.py)
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
