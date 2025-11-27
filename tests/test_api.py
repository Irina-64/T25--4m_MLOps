def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_basic(client):
    payload = {"text": "you are stupid"}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert "detox_text" in data
    assert isinstance(data["detox_text"], str)
