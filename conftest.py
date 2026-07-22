"""
conftest.py — pytest configuration.

Adds the project root to sys.path so that `from src.xxx import ...`
works correctly when running pytest from the project root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
