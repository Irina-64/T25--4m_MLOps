import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def preprocess_telco_data():
    # Чтение данных
    df = pd.read_csv('data/raw/raw_dataset.csv')
    
    print("Исходные данные:")
    print(f"Размер: {df.shape}")
    print(f"Пропуски: {df.isnull().sum().sum()}")
    
    # Анализ и очистка TotalCharges 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Заполнение пропусков в TotalCharges
    # Если tenure = 0, то TotalCharges должен быть 0 или MonthlyCharges
    mask = (df['tenure'] == 0) & (df['TotalCharges'].isna())
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']
    
    # Остальные пропуски заполняем медианой
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Создание новых фич
    df['avg_charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 чтобы избежать деления на 0
    df['charge_ratio'] = df['MonthlyCharges'] / df['avg_charge_per_month']
    df['tenure_group'] = pd.cut(df['tenure'], 
                               bins=[0, 12, 24, 36, 48, 60, np.inf],
                               labels=['0-1y', '1-2y', '2-3y', '3-4y', '4-5y', '5y+'])
    
    # Обработка категориальных переменных
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']
    
    # Бинарные переменные
    binary_map = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map(binary_map)
    
    # Сложные категориальные
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in service_cols:
        df[col] = df[col].map({'No': 0, 'Yes': 1, 'No internet service': 0})
    
    # One-Hot Encoding для остальных категориальных
    ohe_cols = ['InternetService', 'Contract', 'PaymentMethod', 'tenure_group']
    df_encoded = pd.get_dummies(df[ohe_cols], prefix=ohe_cols)
    
    # Label Encoding для пола
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    
    # Целевая переменная
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Объединяем все фичи
    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                   'avg_charge_per_month', 'charge_ratio', 'gender']
    
    # Собираем финальный датасет
    final_features = pd.concat([df[numeric_cols], df_encoded, df[service_cols], 
                               df[['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']]], axis=1)
    
    # Добавляем целевую переменную
    final_features['Churn'] = df['Churn']
    
    # Удаляем возможные NaN после преобразований
    final_features = final_features.dropna()
    
    print("\nПосле препроцессинга:")
    print(f"Финальный размер: {final_features.shape}")
    print(f"Количество фичей: {final_features.shape[1] - 1}")  # -1 потому что Churn
    print(f"Баланс классов: {final_features['Churn'].value_counts()}")
    
    # Сохраняем
    final_features.to_csv('data/processed/processed.csv', index=False)
    print("✅ Данные сохранены в data/processed/processed.csv")
    
    return final_features

if __name__ == "__main__":
    preprocess_telco_data()