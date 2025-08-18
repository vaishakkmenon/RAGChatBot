# tests/conftest.py
import sys
import os
from pathlib import Path

# Insert the project root (one level up) at the front of sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))