"""Global test configuration for pytest."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so local plugins can be imported
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Automatically load the local pytest_asyncio plugin
pytest_plugins = ("pytest_asyncio",)
