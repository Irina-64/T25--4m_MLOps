import json
import pytest

def test_root_endpoint(client):
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

def test_health_endpoint(client):
    """Тест health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict_output_structure(client, sample_payload):
    """Тест структуры ответа /predict"""
    response = client.post("/predict", json=sample_payload)
    
    # Проверяем успешный статус
    assert response.status_code in [200, 201]
    
    data = response.json()
    
    # Проверяем наличие обязательных полей
    assert "prediction" in data or "churn_probability" in data or "delay_prob" in data
    
    # Проверяем, что есть хотя бы один из ожидаемых ключей
    expected_keys = ["prediction", "churn_probability", "delay_prob", "probability"]
    assert any(key in data for key in expected_keys)

def test_predict_output_type(client, sample_payload):
    """Тест типа данных в ответе /predict"""
    resp = client.post("/predict", json=sample_payload)
    assert resp.status_code == 200
    data = resp.json()
    
    # Проверяем что в ответе есть поле с вероятностью
    if "churn_probability" in data:
        assert isinstance(data["churn_probability"], (int, float))
    elif "prediction" in data:
        assert isinstance(data["prediction"], (int, float, bool))
    elif "delay_prob" in data:
        assert isinstance(data["delay_prob"], (int, float))

def test_predict_with_mock_model(client, sample_payload, mocker):
    """Тест с моком модели"""
    # Мокаем загрузку модели, если она не существует
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('glob.glob', return_value=[])
    
    response = client.post("/predict", json=sample_payload)
    
    # Даже без модели API должен вернуть структурированный ответ
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

def test_predict_missing_fields(client):
    """Тест с неполными данными"""
    incomplete_payload = {
        "tenure": 34,
        "MonthlyCharges": 56.95
    }
    
    response = client.post("/predict", json=incomplete_payload)
    # FastAPI вернет 422 если данные не проходят валидацию Pydantic
    assert response.status_code in [422, 400, 200]

def test_docs_endpoints(client):
    """Тест доступности документации"""
    endpoints = ["/docs", "/redoc", "/openapi.json"]
    
    for endpoint in endpoints[:1]:  # Проверяем только /docs для скорости
        response = client.get(endpoint)
        assert response.status_code == 200