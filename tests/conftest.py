import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app from the project
from src.api import app


@pytest.fixture(scope="session")
def client():
    """TestClient fixture for FastAPI app (session-scoped)."""
    with TestClient(app) as c:
        yield c
    