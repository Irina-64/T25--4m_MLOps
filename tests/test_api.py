def test_predict_output_type(client):
    payload = {"carrier": "AA", "dep_hour": 9}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "win_probability" in data
    assert isinstance(data["win_probability"], float)
