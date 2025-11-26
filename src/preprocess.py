import pandas as pd
import numpy as np
import os
import chardet
from datetime import datetime, timedelta

def detect_encoding(file_path):
    """
    Автоматическое определение кодировки файла
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"Определена кодировка: {encoding} (уверенность: {confidence:.2f})")
        return encoding

def preprocess_data():
    """
    Скрипт предобработки данных для прогнозирования курсов валют
    """
    # Создаем директорию для processed данных, если ее нет
    os.makedirs('data/processed', exist_ok=True)
    
    # Чтение исходных данных с определением кодировки
    file_path = 'data/raw/flights_sample.csv'
    print(f"Чтение данных из {file_path}...")
    
    try:
        # Пытаемся определить кодировку
        encoding = detect_encoding(file_path)
        # Используем точку с запятой как разделитель
        data = pd.read_csv(file_path, encoding=encoding, sep=';')
    except Exception as e:
        print(f"Ошибка при чтении с определенной кодировкой: {e}")
        print("Попытка чтения с альтернативными кодировками...")
        
        # Попробуем распространенные кодировки
        encodings_to_try = ['windows-1251', 'cp1251', 'latin1', 'iso-8859-1', 'utf-8']
        for enc in encodings_to_try:
            try:
                data = pd.read_csv(file_path, encoding=enc, sep=';')
                print(f"Успешно прочитано с кодировкой: {enc}")
                break
            except:
                continue
        else:
            # Если все кодировки не подошли, пробуем с ошибками='ignore'
            print("Использую кодировку с игнорированием ошибок...")
            data = pd.read_csv(file_path, encoding='utf-8', errors='ignore', sep=';')
    
    # Выводим информацию о данных
    print(f"Исходная форма данных: {data.shape}")
    print("Столбцы в данных:", data.columns.tolist())
    
    # Если данные все еще в одной колонке, попробуем разделить их
    if len(data.columns) == 1 and ';' in data.columns[0]:
        print("Данные в одной колонке, пытаюсь разделить...")
        # Разделяем первую колонку по точке с запятой
        split_data = data.iloc[:, 0].str.split(';', expand=True)
        
        # Предполагаем, что первая строка - заголовки
        if split_data.shape[1] >= 3:
            # Используем первую строку как заголовки
            data = split_data.iloc[1:].copy()
            data.columns = split_data.iloc[0].values[:data.shape[1]]
            # Оставляем только нужные колонки
            data = data.iloc[:, :3]
            data.columns = ['USD_RUB', 'EUR_RUB', 'GBP_RUB']
            print("Данные успешно разделены")
    
    # Проверяем наличие необходимых колонок
    required_columns = ['EUR_RUB', 'GBP_RUB', 'USD_RUB']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"ВНИМАНИЕ: Отсутствуют колонки: {missing_columns}")
        print("Доступные колонки:", data.columns.tolist())
        
        # Если колонок нет, но есть три колонки с числовыми данными, переименуем их
        if len(data.columns) >= 3:
            print("Переименовываю первые три колонки в USD_RUB, EUR_RUB, GBP_RUB")
            column_names = ['USD_RUB', 'EUR_RUB', 'GBP_RUB']
            data.columns = column_names[:len(data.columns)]
            # Обновляем список missing_columns
            missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось найти колонки: {missing_columns}")
        return None
    
    # Очистка NaN
    print("Обработка пропущенных значений...")
    initial_rows = len(data)
    data.dropna(inplace=True)
    print(f"Удалено строк с NaN: {initial_rows - len(data)}")
    
    # Приведение типов - убеждаемся, что числовые колонки имеют правильный тип
    numeric_columns = ['EUR_RUB', 'GBP_RUB', 'USD_RUB']
    for col in numeric_columns:
        if col in data.columns:
            # Заменяем запятые на точки и конвертируем в float
            data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
    
    # Удаляем строки, где преобразование не удалось
    data.dropna(subset=numeric_columns, inplace=True)
    
    # Создаем искусственную временную шкалу
    print("Создание искусственной временной шкалы...")
    
    # Начинаем с произвольной даты (например, 2020-01-01)
    start_date = datetime(2020, 1, 1)
    
    # Создаем столбец даты на основе индекса строки
    data['date'] = [start_date + timedelta(days=i) for i in range(len(data))]
    
    # Создаем признаки на основе искусственной даты
    data['day_of_week'] = data['date'].dt.dayofweek
    data['is_weekend'] = (data['date'].dt.dayofweek >= 5).astype(int)
    
    # Для departure_hour_bucket используем равномерное распределение
    data['departure_hour_bucket'] = pd.cut(
        data.index % 24,
        bins=[0, 6, 12, 18, 24], 
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        include_lowest=True
    )
    
    print("Созданные признаки:")
    print(f"- day_of_week: {sorted(data['day_of_week'].unique())}")
    print(f"- is_weekend: {data['is_weekend'].value_counts().to_dict()}")
    print(f"- departure_hour_bucket: {data['departure_hour_bucket'].value_counts().to_dict()}")
    
    # Сохраняем обработанные данные
    output_path = 'data/processed/processed.csv'
    data.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Предобработка завершена!")
    print(f"Итоговая форма данных: {data.shape}")
    print(f"Обработанные данные сохранены в: {output_path}")
    
    # Выводим основную информацию о данных
    print("\nОсновная информация о данных:")
    available_numeric = [col for col in numeric_columns if col in data.columns]
    if available_numeric:
        print(data[available_numeric].describe())
    else:
        print("Нет числовых колонок для описания")
    
    return data

def prepare_features(data):
    """
    Создание признаков для временных рядов валют
    (перенесено из train.py для устранения дублирования)
    """
    data = data.copy()
    
    # Сортируем по дате
    data = data.sort_values('date').reset_index(drop=True)
    
    # Создаем целевую переменную - направление изменения USD_RUB на следующий день
    data['USD_RUB_target'] = (data['USD_RUB'].shift(-1) > data['USD_RUB']).astype(int)
    
    # Создаем признаки БЕЗ утечки данных
    for lag in [1, 2, 3, 5, 7]:
        data[f'USD_RUB_lag_{lag}'] = data['USD_RUB'].shift(lag)
        data[f'EUR_RUB_lag_{lag}'] = data['EUR_RUB'].shift(lag)
        data[f'GBP_RUB_lag_{lag}'] = data['GBP_RUB'].shift(lag)
    
    for window in [3, 5, 7]:
        data[f'USD_RUB_MA_{window}'] = data['USD_RUB'].shift(1).rolling(window=window, min_periods=1).mean()
        data[f'EUR_RUB_MA_{window}'] = data['EUR_RUB'].shift(1).rolling(window=window, min_periods=1).mean()
        data[f'GBP_RUB_MA_{window}'] = data['GBP_RUB'].shift(1).rolling(window=window, min_periods=1).mean()
    
    data['USD_RUB_change_1'] = data['USD_RUB'] - data['USD_RUB'].shift(1)
    data['USD_RUB_change_3'] = data['USD_RUB'] - data['USD_RUB'].shift(3)
    
    # Удаляем строки с пропусками
    data = data.dropna()
    
    return data

def get_feature_names():
    """
    Возвращает список всех признаков, используемых в модели
    (перенесено из train.py для устранения дублирования)
    """
    base_features = ['USD_RUB', 'EUR_RUB', 'GBP_RUB', 'day_of_week', 'is_weekend']
    
    lag_features = []
    for lag in [1, 2, 3, 5, 7]:
        for currency in ['USD_RUB', 'EUR_RUB', 'GBP_RUB']:
            lag_features.append(f'{currency}_lag_{lag}')
    
    ma_features = []
    for window in [3, 5, 7]:
        for currency in ['USD_RUB', 'EUR_RUB', 'GBP_RUB']:
            ma_features.append(f'{currency}_MA_{window}')
    
    change_features = ['USD_RUB_change_1', 'USD_RUB_change_3']
    
    all_features = base_features + lag_features + ma_features + change_features
    return [f for f in all_features if f not in ['date', 'USD_RUB_target']]

if __name__ == "__main__":
    preprocess_data()

