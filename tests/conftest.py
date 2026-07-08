"""Shared test setup.

Importing main.py has side effects (load_dotenv + setup_tracing), so tests
force TRACING_PROVIDER=none BEFORE any project import: load_dotenv() never
overrides existing env vars, and setup_tracing() is a no-op for "none" — no
network calls from test runs.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TRACING_PROVIDER"] = "none"
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("MODEL_NAME", "test/model")
