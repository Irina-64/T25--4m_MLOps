"""
Скрипт для подготовки данных для системы обнаружения дрейфа
"""

import pandas as pd
import os
import sys

# Добавляем корневую директорию в путь для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Теперь импортируем после добавления пути
from src.preprocess import prepare_features
import joblib

def prepare_all_data():
    """Подготовка всех необходимых данных для дрейфа"""
    print("Подготовка данных для системы обнаружения дрейфа...")
    
    # 1. Загружаем processed.csv
    processed_path = os.path.join(project_root, 'data/processed/processed.csv')
    if not os.path.exists(processed_path):
        print(f"❌ Файл не найден: {processed_path}")
        return False
    
    data = pd.read_csv(processed_path)
    print(f"Исходные данные: {data.shape}")
    
    # Проверяем наличие колонки date
    if 'date' not in data.columns:
        print("⚠️  Колонка 'date' не найдена, создаем искусственную...")
        from datetime import datetime, timedelta
        start_date = datetime(2020, 1, 1)
        data['date'] = [start_date + timedelta(days=i) for i in range(len(data))]
    
    # 2. Применяем prepare_features
    try:
        data_with_features = prepare_features(data)
        print(f"Данные с фичами: {data_with_features.shape}")
        
        # Проверяем наличие целевой переменной
        if 'USD_RUB_target' in data_with_features.columns:
            print(f"✅ Целевая переменная создана: {data_with_features['USD_RUB_target'].value_counts().to_dict()}")
        else:
            print("❌ Целевая переменная не создана!")
            return False
    except Exception as e:
        print(f"❌ Ошибка при создании признаков: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Сохраняем обновленные данные
    data_with_features.to_csv(processed_path, index=False)
    print(f"✅ Данные обновлены: {processed_path}")
    
    # 4. Создаем референсные данные
    train_size = int(0.8 * len(data_with_features))
    reference_data = data_with_features.iloc[:train_size]
    reference_path = os.path.join(project_root, 'data/processed/train_reference.csv')
    reference_data.to_csv(reference_path, index=False)
    print(f"✅ Референсные данные созданы: {reference_path} ({reference_data.shape})")
    
    # 5. Проверяем модель и фичи
    model_path = os.path.join(project_root, 'models/best_model.joblib')
    if os.path.exists(model_path):
        print(f"✅ Модель существует: {model_path}")
        
        # Проверяем feature_names
        feature_path = os.path.join(project_root, 'models/feature_names.joblib')
        if os.path.exists(feature_path):
            feature_names = joblib.load(feature_path)
            print(f"✅ Загружено {len(feature_names)} признаков модели")
            
            # Проверяем совпадение признаков
            missing_in_data = [f for f in feature_names if f not in data_with_features.columns]
            if missing_in_data:
                print(f"⚠️  Отсутствуют в данных: {len(missing_in_data)} признаков")
                print(f"   Пример: {missing_in_data[:5]}")
                
                # Показываем какие признаки есть
                print(f"\n   Доступные признаки в данных ({len(data_with_features.columns)}):")
                print(f"   {list(data_with_features.columns)[:10]}...")
            else:
                print(f"✅ Все признаки модели есть в данных")
        else:
            print("⚠️  Файл feature_names.joblib не найден")
    else:
        print("⚠️  Модель не найдена. Сначала обучите модель.")
    
    return True

if __name__ == "__main__":
    prepare_all_data()