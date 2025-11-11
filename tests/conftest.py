import os
import sys

import pytest
from fastapi.testclient import TestClient

from src.api import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client():
    return TestClient(app)
