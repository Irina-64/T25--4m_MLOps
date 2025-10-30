import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import mlflow
import mlflow.sklearn

# Фиксатор случайности для воспроизводимости результатов
np.random.seed(42)

print("=== STARTING MODEL TRAINING ===")

try:
    # 1. Загрузка data/processed/processed.csv
    print("Loading processed data...")
    data = pd.read_csv('data/processed/processed.csv')
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # 2. Подготовка данных для обучения
    print("Preparing data for training...")
    
    # Создаем признаки и цели для прогнозирования КАЖДОЙ валюты
    # Используем данные дня t для предсказания дня t+1
    
    # Признаки: текущие значения всех валют (все кроме последней строки)
    X = data[['USD_RUB', 'GBP_RUB', 'EUR_RUB']].values[:-1]
    
    # Цели: значения на следующий день для КАЖДОЙ валюты
    targets = {
        'USD_RUB': data['USD_RUB'].values[1:],
        'EUR_RUB': data['EUR_RUB'].values[1:],
        'GBP_RUB': data['GBP_RUB'].values[1:]
    }
    
    print(f"Features shape: {X.shape}")
    for currency, y in targets.items():
        print(f"Target {currency} shape: {y.shape}")
    
    # 3. Обучаем отдельную модель для каждой валюты
    models = {}
    metrics = {}
    
    # Настройка MLflow
    mlflow.set_experiment("flight_delay")
    
    for currency, y in targets.items():
        print(f"\n{'='*50}")
        print(f"TRAINING MODEL FOR {currency}")
        print(f"{'='*50}")
        
        with mlflow.start_run(run_name=f"RandomForest_{currency}"):
            # Разделение на train/test для текущей валюты
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Обучение RandomForestRegressor для текущей валюты
            print(f"Training RandomForestRegressor for {currency}...")
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            model.fit(X_train, y_train)
            
            # Предсказания и вычисление метрик
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model performance for {currency}:")
            print(f"MAE: {mae:.4f}")
            print(f"MSE: {mse:.4f}") 
            print(f"R2 Score: {r2:.4f}")
            
            # Сохраняем модель и метрики
            models[currency] = model
            metrics[currency] = {'mae': mae, 'mse': mse, 'r2': r2}
            
            # MLflow логирование для каждой валюты
            mlflow.log_param("model", "RandomForestRegressor")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("target_currency", currency)
            
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            
            # Логирование модели в MLflow
            mlflow.sklearn.log_model(model, f"model_{currency}")
    
    # 4. Сохранение всех моделей
    print(f"\n{'='*50}")
    print("SAVING ALL MODELS")
    print(f"{'='*50}")
    
    os.makedirs('models', exist_ok=True)
    
    for currency, model in models.items():
        model_path = f'models/random_forest_{currency.lower()}.joblib'
        joblib.dump(model, model_path)
        print(f"Model for {currency} saved to: {model_path}")
    
    # 5. Вывод итоговых результатов
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    
    for currency, metric in metrics.items():
        print(f"{currency}: MAE={metric['mae']:.4f}, R2={metric['r2']:.4f}")
    
    print("=" * 50)
    print("ALL MODELS TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
        
except Exception as e:
    print(f"Error during model training: {e}")
    raise
