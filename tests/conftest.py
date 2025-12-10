import pytest
from fastapi.testclient import TestClient
from api import app


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
    return TestClient(app)
