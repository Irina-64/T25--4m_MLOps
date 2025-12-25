"""
Скрипт для подготовки данных для Feast Feature Store
Добавляет timestamp колонки к обработанным данным
"""
import pandas as pd
import os
from datetime import datetime

def prepare_feast_data():
    """Подготавливает данные для Feast, добавляя timestamp колонки"""
    import os
    base_path = os.getenv("WORKSPACE_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_path = os.path.join(base_path, "data/processed/processed.csv")
    feast_data_path = os.path.join(base_path, "data/processed/feast_features.csv")
    
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Файл не найден: {processed_path}")
    
    # Загружаем обработанные данные
    df = pd.read_csv(processed_path)
    
    # Добавляем timestamp колонки для Feast
    # Используем фиксированную дату для всех записей (можно использовать текущую дату)
    event_timestamp = datetime(2024, 12, 1)
    created_timestamp = datetime(2024, 12, 1)
    
    df["event_timestamp"] = event_timestamp
    df["created_timestamp"] = created_timestamp
    
    # Сохраняем данные для Feast (без churn, так как это целевая переменная)
    feast_df = df.drop(columns=["churn"])
    
    # Создаем директорию если нужно
    os.makedirs(os.path.dirname(feast_data_path), exist_ok=True)
    
    # Сохраняем
    feast_df.to_csv(feast_data_path, index=False)
    print(f"Данные для Feast сохранены в: {feast_data_path}")
    print(f"   Количество записей: {len(feast_df)}")
    print(f"   Количество признаков: {len(feast_df.columns) - 3}")  # -3 для user_id, event_timestamp, created_timestamp
    
    return feast_data_path

if __name__ == "__main__":
    prepare_feast_data()

