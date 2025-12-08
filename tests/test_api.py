import pytest
import json
import os
import numpy as np
from unittest.mock import patch

def test_health_check(client):
    """Тест проверки здоровья API"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_root_endpoint(client):
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

def test_predict_output_structure(client, sample_input_data):
    """Тест структуры выходных данных предсказания"""
    # Мокаем модель для тестов
    with patch('src.api.model') as mock_model, \
         patch('src.api.scaler') as mock_scaler, \
         patch('src.api.feature_names') as mock_features:

        # Настраиваем моки
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_scaler.transform.return_value = np.random.rand(1, 32)
        mock_features = list(sample_input_data.keys())

        response = client.post("/predict", json=sample_input_data)
        assert response.status_code == 200

        data = response.json()
        # Проверяем только существующие поля
        assert "delay_prob" in data
        assert "prediction" in data
        assert "confidence" in data

def test_predict_missing_fields(client):
    """Тест обработки отсутствующих полей"""
    incomplete_data = {
        "USD_RUB": 90.5,
        "EUR_RUB": 98.2
        # Остальные поля отсутствуют
    }
    
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422  # Validation error

def test_api_without_models(client):
    """Тест API когда модели не загружены"""
    with patch('src.api.model', None), \
         patch('src.api.scaler', None), \
         patch('src.api.feature_names', None):

        response = client.post("/predict", json={
            "USD_RUB": 90.5, "EUR_RUB": 98.2, "GBP_RUB": 115.1, "day_of_week": 2,
            "is_weekend": 0, "USD_RUB_lag_1": 90.3, "USD_RUB_lag_2": 90.1,
            "USD_RUB_lag_3": 89.8, "USD_RUB_lag_5": 89.5, "USD_RUB_lag_7": 89.2,
            "EUR_RUB_lag_1": 98.0, "EUR_RUB_lag_2": 97.8, "EUR_RUB_lag_3": 97.5,
            "EUR_RUB_lag_5": 97.2, "EUR_RUB_lag_7": 96.9, "GBP_RUB_lag_1": 114.8,
            "GBP_RUB_lag_2": 114.5, "GBP_RUB_lag_3": 114.2, "GBP_RUB_lag_5": 113.9,
            "GBP_RUB_lag_7": 113.6, "USD_RUB_MA_3": 90.1, "USD_RUB_MA_5": 89.9,
            "USD_RUB_MA_7": 89.7, "EUR_RUB_MA_3": 97.8, "EUR_RUB_MA_5": 97.6,
            "EUR_RUB_MA_7": 97.4, "GBP_RUB_MA_3": 114.5, "GBP_RUB_MA_5": 114.3,
            "GBP_RUB_MA_7": 114.1, "USD_RUB_change_1": 0.2, "USD_RUB_change_3": 0.7
        })
        # Ожидаем 503 Service Unavailable когда модели не загружены
        assert response.status_code == 503
