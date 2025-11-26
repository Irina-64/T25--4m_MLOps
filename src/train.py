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
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import prepare_features, get_feature_names

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
    data = prepare_features(data)
    print(f"После подготовки: {len(data)} строк")
    
    # Получаем имена признаков
    feature_columns = get_feature_names()
    
    # Проверяем наличие целевой колонки
    target_column = 'USD_RUB_target'
    if target_column not in data.columns:
        print(f"Ошибка: целевая колонка {target_column} не найдена")
        return
    
    # Подготавливаем данные
    X = data[feature_columns]
    y = data[target_column]
    
    print(f"Используется {len(feature_columns)} числовых признаков")
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
            mlflow.log_param("n_features", len(feature_columns))
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("test_size", len(X_test))
            
            # Логируем модель в MLflow
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
        joblib.dump(feature_columns, "models/feature_names.joblib")
        
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
