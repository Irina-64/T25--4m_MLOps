import pytest
from fastapi.testclient import TestClient
import sys
import os

# Добавляем src в путь Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import app

@pytest.fixture
def client():
    """Фикстура для тестового клиента FastAPI"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def sample_payload():
    """Фикстура с примером данных для предсказания"""
    return {
        "SeniorCitizen": 0,
        "tenure": 34,
        "MonthlyCharges": 56.95,
        "TotalCharges": 1889.5,
        "avg_charge_per_month": 53.98571428571429,
        "charge_ratio": 1.0549087060068802,
        "gender": 1,
        "InternetService_DSL": True,
        "InternetService_Fiber optic": False,
        "InternetService_No": False,
        "Contract_Month-to-month": False,
        "Contract_One year": True,
        "Contract_Two year": False,
        "PaymentMethod_Bank transfer (automatic)": False,
        "PaymentMethod_Credit card (automatic)": False,
        "PaymentMethod_Electronic check": False,
        "PaymentMethod_Mailed check": True,
        "tenure_group_0-1y": False,
        "tenure_group_1-2y": False,
        "tenure_group_2-3y": True,
        "tenure_group_3-4y": False,
        "tenure_group_4-5y": False,
        "tenure_group_5y+": False,
        "MultipleLines": 0.0,
        "OnlineSecurity": 1,
        "OnlineBackup": 0,
        "DeviceProtection": 1,
        "TechSupport": 0,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "Partner": 0,
        "Dependents": 0,
        "PhoneService": 1,
        "PaperlessBilling": 0,
        "Churn": 0
    }