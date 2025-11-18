import pytest
from src.api import MODEL_FEATURES


def test_predict_returns_probability_in_range(client):
    payload = {name: 0 for name in MODEL_FEATURES}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "churn_probability" in body

    prob = body["churn_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


