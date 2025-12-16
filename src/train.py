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
import os

def train_model():
    # Настройка MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("telco_churn")
    
    # Включаем autolog для автоматического логирования моделей
    mlflow.sklearn.autolog()
    
    # Загрузка обработанных данных
    df = pd.read_csv('data/processed/processed.csv')
    print(f"📊 Данные загружены: {df.shape}")
    print(f"📋 Колонки: {df.columns.tolist()[:10]}...")  # Только первые 10
    
    # Разделение на признаки и целевую переменную
    if 'customerID' in df.columns:
        X = df.drop(columns=['Churn', 'customerID'])
    else:
        X = df.drop(columns=['Churn'])
    
    y = df['Churn']
    
    print(f"🎯 Признаков: {X.shape[1]}, Целевая переменная: {y.shape[0]}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"⚖️ Баланс классов в train: {pd.Series(y_train).value_counts().to_dict()}")
    
    # Модели
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    }
    
    best_score = 0
    best_model = None
    best_model_name = ""
    best_run_id = ""
    
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%H%M%S')}") as run:
            print(f"\n🤖 Обучение {name}...")
            
            # Обучение
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"✅ Accuracy: {accuracy:.4f}")
            print(f"✅ ROC-AUC: {roc_auc:.4f}")
            
            # Вручную логируем метрики для надежности
            mlflow.log_param("model", name)
            mlflow.log_param("features_count", X_train.shape[1])
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("test_size", X_test.shape[0])
            
            # Дополнительные метрики
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Вручную логируем модель с явным указанием artifact_path
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",  # Важно: именно "model" а не что-то другое
                registered_model_name="telco_churn_model"
            )
            
            print(f"✅ Модель залогирована в MLflow (run_id: {run.info.run_id})")
            
            # Проверяем лучшую модель
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_model_name = name
                best_run_id = run.info.run_id
    
    # Отключаем autolog чтобы не мешал другим скриптам
    mlflow.sklearn.autolog(disable=True)
    
    # Сохранение лучшей модели в файл
    if best_model is not None:
        os.makedirs("models", exist_ok=True)
        model_filename = f"models/{best_model_name.lower()}_model.joblib"
        joblib.dump(best_model, model_filename)
        
        # Также сохраняем как model.joblib для API
        joblib.dump(best_model, "models/model.joblib")
        
        print(f"\n🎉 Лучшая модель: {best_model_name}")
        print(f"📊 ROC-AUC: {best_score:.4f}")
        print(f"💾 Сохранена как: {model_filename}")
        print(f"💾 И как: models/model.joblib (для API)")
        print(f"🔗 Run ID: {best_run_id}")
        
        # Сохраняем run_id в файл для регистрации модели
        with open("models/best_run_id.txt", "w") as f:
            f.write(best_run_id)
        
        # Также сохраняем метаданные
        with open("models/best_model_info.json", "w") as f:
            import json
            json.dump({
                "model_name": best_model_name,
                "roc_auc": best_score,
                "run_id": best_run_id,
                "timestamp": datetime.now().isoformat()
            }, f)
    
    print(f"\n📊 MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print("📁 Для просмотра результатов: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    
    return best_model, best_score, best_run_id

if __name__ == "__main__":
    model, score, run_id = train_model()