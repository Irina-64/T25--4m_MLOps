def test_predict_ok(client):
    payload = {"a": 1, "b": 2, "c": 3}

    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert "win_probability" in data
    assert isinstance(data["win_probability"], float)
