import pytest
import pandas as pd
import sys
import os

# Добавляем src в путь Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Пытаемся импортировать функции препроцессинга
try:
    from preprocess import preprocess_data
    HAS_PREPROCESS = True
except ImportError:
    HAS_PREPROCESS = False

@pytest.mark.skipif(not HAS_PREPROCESS, reason="preprocess module not available")
def test_preprocess_functions():
    """Тест функций препроцессинга"""
    # Создаем тестовые данные
    test_data = pd.DataFrame({
        'tenure': [1, 34, 72],
        'MonthlyCharges': [29.85, 56.95, 104.80],
        'TotalCharges': [29.85, 1889.5, 7500.0]
    })
    
    # Проверяем что функция существует и работает
    result = preprocess_data(test_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(test_data)
    assert 'avg_charge_per_month' in result.columns
    assert 'charge_ratio' in result.columns

@pytest.mark.skipif(not HAS_PREPROCESS, reason="preprocess module not available")
def test_preprocess_feature_engineering():
    """Тест feature engineering"""
    from preprocess import calculate_charge_ratio
    
    tenure = 34
    monthly = 56.95
    total = 1889.5
    
    ratio = calculate_charge_ratio(total, tenure, monthly)
    
    assert isinstance(ratio, float)
    assert ratio > 0