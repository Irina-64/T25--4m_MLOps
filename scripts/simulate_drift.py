# scripts/simulate_drift.py
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

def simulate_drift_api_calls():
    """Имитация запросов к API с постепенным дрейфом"""
    base_url = "http://localhost:8080"
    
    # Базовые значения признаков
    base_features = {
        "USD_RUB": 90.0,
        "EUR_RUB": 98.0,
        "GBP_RUB": 115.0,
        "day_of_week": 2,
        "is_weekend": 0,
        "USD_RUB_lag_1": 89.8,
        "USD_RUB_lag_2": 89.6,
        "USD_RUB_lag_3": 89.4,
        "USD_RUB_lag_5": 89.0,
        "USD_RUB_lag_7": 88.6,
        "EUR_RUB_lag_1": 97.8,
        "EUR_RUB_lag_2": 97.6,
        "EUR_RUB_lag_3": 97.4,
        "EUR_RUB_lag_5": 97.0,
        "EUR_RUB_lag_7": 96.6,
        "GBP_RUB_lag_1": 114.8,
        "GBP_RUB_lag_2": 114.6,
        "GBP_RUB_lag_3": 114.4,
        "GBP_RUB_lag_5": 114.0,
        "GBP_RUB_lag_7": 113.6,
        "USD_RUB_MA_3": 89.7,
        "USD_RUB_MA_5": 89.5,
        "USD_RUB_MA_7": 89.3,
        "EUR_RUB_MA_3": 97.6,
        "EUR_RUB_MA_5": 97.4,
        "EUR_RUB_MA_7": 97.2,
        "GBP_RUB_MA_3": 114.6,
        "GBP_RUB_MA_5": 114.4,
        "GBP_RUB_MA_7": 114.2,
        "USD_RUB_change_1": 0.2,
        "USD_RUB_change_3": 0.6
    }
    
    print("Simulating API calls with drift...")
    
    # Имитируем 3 фазы: нормальная работа, умеренный дрейф, сильный дрейф
    phases = [
        ("normal", 0.0, 10),      # 10 запросов, без дрейфа
        ("moderate_drift", 0.15, 15), # 15 запросов, 15% дрейф
        ("strong_drift", 0.3, 20)     # 20 запросов, 30% дрейф
    ]
    
    for phase_name, drift_factor, n_requests in phases:
        print(f"\nPhase: {phase_name} (drift: {drift_factor*100}%)")
        
        for i in range(n_requests):
            # Добавляем дрейф к основным признакам
            features = base_features.copy()
            features["USD_RUB"] = features["USD_RUB"] * (1 + drift_factor)
            features["EUR_RUB"] = features["EUR_RUB"] * (1 + drift_factor * 0.5)
            
            try:
                response = requests.post(
                    f"{base_url}/predict",
                    json=features,
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Request {i+1}: prob={result['delay_prob']:.3f}, "
                          f"prediction={result['prediction']}")
                else:
                    print(f"Request {i+1}: Error {response.status_code}")
                    
            except Exception as e:
                print(f"Request {i+1}: Failed - {e}")
            
            time.sleep(0.5)  # Задержка между запросами
    
    print("\nDrift simulation completed!")

if __name__ == "__main__":
    # Убедитесь, что ваш API запущен (python src/api.py)
    simulate_drift_api_calls()