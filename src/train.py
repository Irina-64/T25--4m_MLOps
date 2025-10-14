import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

def train_model():
    # Загрузка обработанных данных
    df = pd.read_csv('data/processed/processed.csv')
    
    print("Данные загружены. Размер:", df.shape)
    print("Колонки:", df.columns.tolist())
    
    # Разделение на features и target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Class balance - Train: {y_train.value_counts()}")
    
    # Настройка MLflow
    mlflow.set_experiment("telco_churn")
    
    # Тестируем разные модели
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_score = 0
    best_model = None
    best_model_name = ""
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M')}"):
            print(f"\n=== Training {model_name} ===")
            
            # Обучение модели
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(classification_report(y_test, y_pred))
            
            # Логирование в MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("features_count", X_train.shape[1])
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("class_0_count", (y_test == 0).sum())
            mlflow.log_metric("class_1_count", (y_test == 1).sum())
            
            # Логирование модели
            mlflow.sklearn.log_model(model, "model")
            
            # Сохранение лучшей модели
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_model_name = model_name
    
    # Сохранение лучшей модели в файл
    if best_model is not None:
        model_filename = f"models/{best_model_name.lower()}_model.joblib"
        joblib.dump(best_model, model_filename)
        print(f"\n✅ Best model saved: {model_filename}")
        print(f"📊 Best ROC-AUC: {best_score:.4f}")
        
        # Логируем информацию о лучшей модели
        with mlflow.start_run(run_name="best_model_final"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_roc_auc", best_score)
            mlflow.log_artifact(model_filename)
    
    return best_model, best_score

if __name__ == "__main__":
    # Создаем папку для моделей если её нет
    import os
    os.makedirs("models", exist_ok=True)
    
    model, score = train_model()