import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    from src.api import app

    return TestClient(app)


