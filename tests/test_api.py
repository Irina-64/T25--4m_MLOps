# tests/test_api.py


def test_predict_api_returns_delay_minutes(client):
    """
    Проверяет контракт API /predict:
    - HTTP 200
    - наличие поля delay_minutes
    - тип значения int
    """

    payload = {
        "carrier": "PKP Intercity",
        "connection": "Wrocław - Bydgoszcz",
        "name": "Bydgoszcz Główna",
        "id_main": "ID001",
        "arrival": "21:48",
        "year": 2024,
        "month": 5,
        "day": 8,
        "dayofweek": 2,
        "season": 1,
        "is_weekend": False,
        "is_holiday": False,
        "hour": 12,
    }

    response = client.post("/predict", json=payload)

    # HTTP уровень
    assert response.status_code == 200

    body = response.json()

    # КОНТРАКТ API (строго из src/api.py)
    assert "delay_minutes" in body
    assert isinstance(body["delay_minutes"], int)
