import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

print("=== STARTING PREPROCESSING ===")

try:
    # Чтение данных из CSV файла с правильным разделителем
    try:
        data = pd.read_csv('data/raw/flights_sample.csv', encoding='utf-8', sep=';')
        print("File read with UTF-8 encoding")
    except UnicodeDecodeError:
        try:
            data = pd.read_csv('data/raw/flights_sample.csv', encoding='cp1251', sep=';')
            print("File read with CP1251 encoding")
        except UnicodeDecodeError:
            data = pd.read_csv('data/raw/flights_sample.csv', encoding='latin1', sep=';')
            print("File read with Latin-1 encoding")
    
    print(f"Original data shape: {data.shape}")
    print(f"Columns in data: {list(data.columns)}")
    print(f"First few rows:")
    print(data.head())
    
    # Очистка NaN и приведение типов
    data.dropna(inplace=True)
    print(f"After dropping NaN: {data.shape}")
    
    # Приведение типов для валютных колонок
    numeric_columns = ['EUR_RUB', 'GBP_RUB', 'USD_RUB']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.dropna(inplace=True)
    print(f"After cleaning data shape: {data.shape}")

    # Сохранение результата
    os.makedirs('data/processed', exist_ok=True)
    data = data[['EUR_RUB', 'GBP_RUB', 'USD_RUB']]
    data.to_csv('data/processed/processed.csv', index=False)
    
    print("Preprocessing completed successfully!")
    print(f"Final data shape: {data.shape}")
    print(f"Processed data saved to: data/processed/processed.csv")
    
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise
