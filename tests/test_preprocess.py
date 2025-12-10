import pytest
import pandas as pd
import numpy as np
from preprocess import preprocess_telco_data


@pytest.fixture
def sample_raw_df():
    """Минимальный DataFrame, имитирующий реальный Telco набор."""
    return pd.DataFrame({
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [0, 5],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["29.85", "113.95"],
        "Churn": ["No", "Yes"]
    })


def test_preprocess_telco_data(monkeypatch, sample_raw_df):
    """Проверяет корректность работы preprocess_telco_data."""

    # 1. Мокаем pd.read_csv → вернёт наш тестовый DataFrame
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: sample_raw_df.copy())

    # 2. Мокаем to_csv, чтобы тест НЕ создавал файлов
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)

    # 3. Запускаем препроцессинг
    df_processed = preprocess_telco_data()

    # --- Проверки ---
    assert isinstance(df_processed, pd.DataFrame)

    # Целевая колонка должна быть числовой 0/1
    assert df_processed["Churn"].isin([0, 1]).all()

    # Не должно оставаться пропусков
    assert df_processed.isnull().sum().sum() == 0

    # Должны появиться новые фичи
    assert "avg_charge_per_month" in df_processed.columns
    assert "charge_ratio" in df_processed.columns

    # OHE: столбцы должны быть созданы
    assert any(col.startswith("InternetService_") for col in df_processed.columns)
    assert any(col.startswith("Contract_") for col in df_processed.columns)
    assert any(col.startswith("PaymentMethod_") for col in df_processed.columns)

    # Значения бинарных фич преобразованы в 0/1
    assert df_processed["Partner"].isin([0, 1]).all()
    assert df_processed["Dependents"].isin([0, 1]).all()

    # Размер > 0
    assert len(df_processed) > 0
