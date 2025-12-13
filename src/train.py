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
from feast import FeatureStore

def train_model():
    # Безопасная настройка MLflow - только логирование в SQLite
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Загрузка обработанных данных
    df = pd.read_csv('data/processed/processed.csv')
    
    print("Данные загружены. Размер:", df.shape)
    print("Колонки:", df.columns.tolist())
    
    # Конвертация CSV в Parquet
    csv_source_path = 'feature_repo/data/telco_features.csv'
    parquet_source_path = 'feature_repo/data/telco_features.parquet'
    
    if not os.path.exists(parquet_source_path):
        print(f"\n🔧 Конвертация источника данных в Parquet...")
        print(f"   Исходный файл: {csv_source_path}")
        source_df = pd.read_csv(csv_source_path)
        
        source_df['event_timestamp'] = pd.to_datetime(source_df['event_timestamp'])
        source_df = source_df.replace(r'^\s*$', np.nan, regex=True)
        
        numeric_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_columns:
            if col in source_df.columns:
                source_df[col] = pd.to_numeric(source_df[col], errors='coerce')
        
        for col in numeric_columns:
            if col in source_df.columns and source_df[col].isna().any():
                median_val = source_df[col].median()
                na_count = source_df[col].isna().sum()
                source_df[col] = source_df[col].fillna(median_val)
                print(f"   Заполнено {na_count} NaN в колонке {col} (медиана: {median_val})")
        
        source_df.to_parquet(parquet_source_path, index=False)
        print(f"   Parquet файл создан: {parquet_source_path}")
    
    # Разделение на target и создание entity DataFrame для Feast
    y = df["Churn"]
    
    entity_df = pd.DataFrame({
        "customer_id": df["customerID"],
        "event_timestamp": pd.to_datetime("2020-01-01")
    })
    
    store = FeatureStore(repo_path="feature_repo")
    
    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "telco_features:SeniorCitizen",
            "telco_features:tenure",
            "telco_features:MonthlyCharges",
            "telco_features:TotalCharges",
        ],
    ).to_df()
    
    X = features_df.drop(columns=["customer_id", "event_timestamp"])
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Class balance - Train: {y_train.value_counts()}")
    
    # Настройка MLflow эксперимента
    mlflow.set_experiment("telco_churn")
    
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
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            # Логируем только параметры и метрики (без артефактов)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("features_count", X_train.shape[1])
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("class_0_count", (y_test == 0).sum())
            mlflow.log_metric("class_1_count", (y_test == 1).sum())
            
            # НЕ логируем модель чтобы избежать ошибки с путями
            # mlflow.sklearn.log_model(model, "model")
            
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
        
        # Логируем информацию о лучшей модели (без артефактов)
        with mlflow.start_run(run_name="best_model_final"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_roc_auc", best_score)
            mlflow.log_param("model_path", model_filename)
    
    print(f"\n📊 MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"📁 MLflow data сохраняется в: mlflow.db (SQLite)")
    print("   Для просмотра результатов: mlflow ui")
    
    return best_model, best_score

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    
    model, score = train_model()