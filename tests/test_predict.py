import os
os.environ.setdefault("SKIP_MODEL_LOAD", "1")

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from src import api

client = TestClient(api.app)


def test_predict_unit(monkeypatch):
    """Unit test for /predict endpoint: mock prepare_sequences and model."""

    def fake_prepare_sequences(df, scaler):
        # return single user id and a dummy sequence (batch, seq_len, features)
        return np.array([0]), np.zeros((1, 5, 2))

    monkeypatch.setattr(api, "prepare_sequences", fake_prepare_sequences)

    class DummyModel:
        def __call__(self, x):
            # return tensor shaped (batch, 1)
            return torch.tensor([[0.75]])

    monkeypatch.setattr(api, "model", DummyModel())

    payload = {"transactions": [{"amount": 1.0, "count": 2, "date": "2024-01-01"}]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "churn_probability" in data
    assert abs(data["churn_probability"] - 0.75) < 1e-6


def test_predict_batch_unit(monkeypatch):
    """Unit test for /predict_batch endpoint: mock main() to return a DataFrame."""

    df = pd.DataFrame({"user_id": [1, 2], "churn_probability": [0.1, 0.9]})

    # mock main to return DataFrame regardless of input
    monkeypatch.setattr(api, "main", lambda input_path, output_path: df)

    csv_bytes = b"user_id,date,amount\n1,2024-01-01,10\n2,2024-01-02,20\n"
    files = {"file": ("data.csv", csv_bytes, "text/csv")}
    resp = client.post("/predict_batch", files=files)
    assert resp.status_code == 200
    result = resp.json()
    assert isinstance(result, list)
    assert result[0]["user_id"] == 1
    assert "churn_probability" in result[0]
