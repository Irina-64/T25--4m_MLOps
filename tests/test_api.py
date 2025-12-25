def test_root_endpoint(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Personality Classifier API"

def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_model_info_endpoint(test_client):
    response = test_client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features" in data
    assert len(data["features"]) == 7

def test_predict_endpoint(test_client, sample_extrovert_data):
    response = test_client.post("/predict", json=sample_extrovert_data)
    assert response.status_code == 200
    data = response.json()
    assert "probability_extrovert" in data
    assert "prediction" in data
    assert "personality" in data
    assert isinstance(data["probability_extrovert"], float)
    assert isinstance(data["prediction"], int)
    assert data["personality"] in ["Extrovert", "Introvert"]

def test_predict_batch_endpoint(test_client, sample_extrovert_data, sample_introvert_data):
    request_data = {"samples": [sample_extrovert_data, sample_introvert_data]}
    response = test_client.post("/predict_batch", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "predictions" in data
    assert data["count"] == 2
    assert len(data["predictions"]) == 2

def test_predict_invalid_data(test_client):
    invalid_data = {"invalid": "data"}
    response = test_client.post("/predict", json=invalid_data)
    assert response.status_code == 422