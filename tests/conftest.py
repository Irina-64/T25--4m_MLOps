import pytest
from fastapi.testclient import TestClient
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import src.api as api

class DummyModel:
    """Фейковая модель, которая заменяет настоящую при тестах."""
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.3, 0.7]])  # фиксированный output для стабильных тестов


@pytest.fixture(scope="session")
def client(monkeypatch):
    """
    Подменяем модель в api.model на фейковую DummyModel(),
    чтобы тесты работали без наличия models/*.joblib.
    """
    monkeypatch.setattr("api.model", DummyModel())
    return TestClient(api.app)
