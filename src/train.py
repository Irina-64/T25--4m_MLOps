import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def prepare_classification_data(data):
    """
    Подготовка данных для задачи классификации БЕЗ утечки данных
    """
    data = data.copy()
    
    # Сортируем по дате
    data = data.sort_values('date').reset_index(drop=True)
    
    # Создаем целевую переменную - направление изменения USD_RUB на следующий день
    data['USD_RUB_target'] = (data['USD_RUB'].shift(-1) > data['USD_RUB']).astype(int)
    
    # Создаем признаки БЕЗ утечки данных - используем только исторические данные
    # Лаги (значения из предыдущих дней)
    for lag in [1, 2, 3, 5, 7]:
        data[f'USD_RUB_lag_{lag}'] = data['USD_RUB'].shift(lag)
        data[f'EUR_RUB_lag_{lag}'] = data['EUR_RUB'].shift(lag)
        data[f'GBP_RUB_lag_{lag}'] = data['GBP_RUB'].shift(lag)
    
    # Исторические скользящие средние (только по прошлым данным)
    for window in [3, 5, 7]:
        data[f'USD_RUB_MA_{window}'] = data['USD_RUB'].shift(1).rolling(window=window, min_periods=1).mean()
        data[f'EUR_RUB_MA_{window}'] = data['EUR_RUB'].shift(1).rolling(window=window, min_periods=1).mean()
        data[f'GBP_RUB_MA_{window}'] = data['GBP_RUB'].shift(1).rolling(window=window, min_periods=1).mean()
    
    # Разности (изменения за предыдущие периоды)
    data['USD_RUB_change_1'] = data['USD_RUB'] - data['USD_RUB'].shift(1)
    data['USD_RUB_change_3'] = data['USD_RUB'] - data['USD_RUB'].shift(3)
    
    # Удаляем строки с пропусками (из-за лагов)
    data = data.dropna()
    
    return data

def main():
    # Создаем директорию для моделей если нет
    os.makedirs('models', exist_ok=True)
    
    # Загружаем обработанные данные
    print("Загрузка обработанных данных...")
    data = pd.read_csv('data/processed/processed.csv')
    data['date'] = pd.to_datetime(data['date'])  # Конвертируем дату
    print(f"Загружено {len(data)} строк")
    
    # Подготавливаем данные для классификации
    print("Подготовка данных для классификации...")
    data = prepare_classification_data(data)
    print(f"После подготовки: {len(data)} строк")
    
    # Определяем числовые признаки
    numeric_features = [
        'USD_RUB', 'EUR_RUB', 'GBP_RUB', 
        'USD_RUB_lag_1', 'USD_RUB_lag_2', 'USD_RUB_lag_3', 'USD_RUB_lag_5', 'USD_RUB_lag_7',
        'EUR_RUB_lag_1', 'EUR_RUB_lag_2', 'EUR_RUB_lag_3', 'EUR_RUB_lag_5', 'EUR_RUB_lag_7',
        'GBP_RUB_lag_1', 'GBP_RUB_lag_2', 'GBP_RUB_lag_3', 'GBP_RUB_lag_5', 'GBP_RUB_lag_7',
        'USD_RUB_MA_3', 'USD_RUB_MA_5', 'USD_RUB_MA_7',
        'EUR_RUB_MA_3', 'EUR_RUB_MA_5', 'EUR_RUB_MA_7',
        'GBP_RUB_MA_3', 'GBP_RUB_MA_5', 'GBP_RUB_MA_7',
        'USD_RUB_change_1', 'USD_RUB_change_3',
        'day_of_week', 'is_weekend'
    ]
    
    # Оставляем только существующие колонки
    numeric_features = [col for col in numeric_features if col in data.columns]
    
    # Проверяем наличие целевой колонки
    target_column = 'USD_RUB_target'
    if target_column not in data.columns:
        print(f"Ошибка: целевая колонка {target_column} не найдена")
        return
    
    # Подготавливаем данные
    X = data[numeric_features]
    y = data[target_column]
    
    print(f"Используется {len(numeric_features)} числовых признаков")
    print(f"Распределение классов: {y.value_counts().to_dict()}")
    
    # Для временных рядов используем специальное разделение
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Разделяем на train/test с учетом временного порядка
    train_size = int(0.8 * len(data))
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    print(f"\nРазмер тренировочной выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Масштабируем признаки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Устанавливаем эксперимент MLflow
    mlflow.set_experiment("flight_delay")
    
    # Обучаем модели с более реалистичными параметрами
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42
        )
    }
    
    best_score = 0
    best_model = None
    best_model_name = ""
    
    for model_name, model in models.items():
        print(f"\n--- Обучение {model_name} ---")
        
        with mlflow.start_run(run_name=model_name):
            # Логируем параметры модели
            if model_name == "RandomForest":
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 5)
                mlflow.log_param("min_samples_split", 20)
            else:  # LogisticRegression
                mlflow.log_param("C", 0.1)
                mlflow.log_param("max_iter", 1000)
            
            # Обучаем модель
            model.fit(X_train_scaled, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Вычисление метрик
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Кросс-валидация с временными рядами
            X_scaled = scaler.transform(X)
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Логируем параметры и метрики в MLflow
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_features", len(numeric_features))
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("test_size", len(X_test))
            
            # ВАЖНО: Логируем модель в MLflow правильно
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"{model_name}_flight_delay"
            )
            
            # Сохраняем модель с помощью joblib
            model_filename = f"models/{model_name.lower()}_model.joblib"
            joblib.dump(model, model_filename)
            
            # Также логируем файл модели как артефакт
            mlflow.log_artifact(model_filename, artifact_path="models")
            
            # Сохраняем RandomForest с правильным именем для DVC
            if model_name == "RandomForest":
                joblib.dump(model, "models/random_forest_model.joblib")
                mlflow.log_artifact("models/random_forest_model.joblib", artifact_path="dvc_models")
                print("Сохранена модель для DVC: models/random_forest_model.joblib")
            
            print(f"Модель сохранена как: {model_filename}")
            
            # Обновляем лучшую модель
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_model_name = model_name
    
    print(f"\n=== ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с ROC AUC: {best_score:.4f} ===")
    
    # Сохраняем лучшую модель и scaler
    if best_model is not None:
        joblib.dump(best_model, "models/best_model.joblib")
        joblib.dump(scaler, "models/scaler.joblib")
        joblib.dump(numeric_features, "models/feature_names.joblib")
        
        # Логируем лучшую модель как отдельный артефакт
        with mlflow.start_run(run_name="best_model"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_roc_auc", best_score)
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                registered_model_name="best_flight_delay_model"
            )
            mlflow.log_artifact("models/best_model.joblib")
            mlflow.log_artifact("models/scaler.joblib")
            mlflow.log_artifact("models/feature_names.joblib")
        
        print("Лучшая модель, scaler и feature_names сохранены и залогированы в MLflow!")
        
        # Детальный анализ лучшей модели
        y_pred_best = best_model.predict(X_test_scaled)
        y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nДетальный анализ лучшей модели:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_best):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_best))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_best))

if __name__ == "__main__":
    main()
