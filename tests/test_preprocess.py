import pandas as pd
import numpy as np
import sys
import os
import pytest

# Добавляем корневую директорию в Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.preprocess import prepare_features, get_feature_names

def create_sample_data():
    """Создание тестовых данных для предобработки"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'USD_RUB': np.random.uniform(70, 100, 100),
        'EUR_RUB': np.random.uniform(80, 110, 100),
        'GBP_RUB': np.random.uniform(90, 120, 100),
        'day_of_week': dates.dayofweek,  # Добавляем day_ofweek
        'is_weekend': (dates.dayofweek >= 5).astype(int)  # Добавляем is_weekend
    })
    return data

def test_prepare_features_structure():
    """Тест структуры данных после подготовки признаков"""
    test_data = create_sample_data()
    processed_data = prepare_features(test_data)
    
    assert isinstance(processed_data, pd.DataFrame)
    assert 'USD_RUB_target' in processed_data.columns
    assert len(processed_data) > 0
    assert not processed_data.isnull().any().any()

def test_get_feature_names():
    """Тест получения имен признаков"""
    feature_names = get_feature_names()
    
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert all(isinstance(name, str) for name in feature_names)
    
    expected_features = ['USD_RUB', 'EUR_RUB', 'GBP_RUB', 'day_of_week', 'is_weekend']
    for feature in expected_features:
        assert feature in feature_names
    
    assert 'USD_RUB_target' not in feature_names
    assert 'date' not in feature_names

def test_feature_engineering():
    """Тест создания фич"""
    test_data = create_sample_data()
    processed_data = prepare_features(test_data)
    
    # Проверяем создание лагов
    assert 'USD_RUB_lag_1' in processed_data.columns
    assert 'EUR_RUB_lag_1' in processed_data.columns
    assert 'GBP_RUB_lag_1' in processed_data.columns
    
    # Проверяем создание скользящих средних
    assert 'USD_RUB_MA_3' in processed_data.columns
    assert 'EUR_RUB_MA_5' in processed_data.columns
    
    # Проверяем создание изменений
    assert 'USD_RUB_change_1' in processed_data.columns

def test_feature_names_consistency():
    """Тест согласованности между prepare_features и get_feature_names"""
    test_data = create_sample_data()
    processed_data = prepare_features(test_data)
    feature_names = get_feature_names()
    
    # Проверяем что все признаки из feature_names присутствуют в processed_data
    missing_features = [f for f in feature_names if f not in processed_data.columns]
    assert len(missing_features) == 0, f"Отсутствующие признаки: {missing_features}"
