import sys
import os
import pytest
from fastapi.testclient import TestClient

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.Api import app


@pytest.fixture(scope="session")
def client():
    return TestClient(app)
