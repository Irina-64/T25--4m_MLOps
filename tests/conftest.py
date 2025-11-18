import os
import sys
import pathlib
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path so `import src` works in tests
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Skip heavy model loading during tests (api reads SKIP_MODEL_LOAD on startup)
os.environ.setdefault("SKIP_MODEL_LOAD", "1")

# Import the FastAPI app from the project after setting env and sys.path
from src.api import app


@pytest.fixture(scope="session")
def client():
    """TestClient fixture for FastAPI app (session-scoped)."""
    with TestClient(app) as c:
        yield c
    