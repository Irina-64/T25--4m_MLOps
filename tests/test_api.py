import pytest
from fastapi.testclient import TestClient
import src.api as api


class DummyModel:
    """Фейковая модель для тестов API."""
    def predict_proba(self, X):
        # Возвращаем вероятность 0.7 для теста
        import numpy as np
        return np.array([[0.3, 0.7]])


@pytest.fixture
def client(monkeypatch):
    """
    Подменяем реальную модель в api.py на фейковую,
    чтобы не зависеть от models/*.joblib.
    """
    from api import model
    monkeypatch.setattr("api.model", DummyModel())
    return TestClient(api.app)


def test_predict_ok(client):
    payload = {"feature1": 1.0, "feature2": "A"}

    response = client.post("/predict", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert "delay_prob" in data
    assert isinstance(data["delay_prob"], float)
    assert 0 <= data["delay_prob"] <= 1


def test_invalid_payload(client):
    # Плохой payload — передаём не-словарь
    response = client.post("/predict", json=["bad", "data"])

    # В этом случае pandas выбросит ошибку -> 400
    assert response.status_code in (400, 422)
