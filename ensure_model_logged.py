# ensure_model_logged.py - гарантирует что модель залогирована в MLflow

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

def ensure_model_logged_in_mlflow(run_id=None):
    """
    Убедиться что модель залогирована в MLflow.
    Если нет - залогировать заново.
    """
    print("🔍 Проверка логов модели в MLflow...")
    
    # Настройка MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    if run_id is None:
        # Пытаемся прочитать из файла
        if os.path.exists("models/best_run_id.txt"):
            with open("models/best_run_id.txt", "r") as f:
                run_id = f.read().strip()
            print(f"📄 Run ID из файла: {run_id}")
        else:
            print("❌ Не найден run_id в файле")
            return False
    
    try:
        # Проверяем существование run
        run = client.get_run(run_id)
        print(f"✅ Run найден: {run_id}")
        
        # Проверяем наличие артефактов модели
        artifacts = client.list_artifacts(run_id, "model")
        if artifacts:
            print("✅ Артефакты модели найдены:")
            for art in artifacts:
                print(f"   - {art.path}")
            
            # Проверяем можно ли загрузить модель
            try:
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.sklearn.load_model(model_uri)
                print("✅ Модель успешно загружена из MLflow")
                return True
            except Exception as e:
                print(f"⚠ Не удалось загрузить модель: {e}")
                return False
        else:
            print("❌ Артефакты модели не найдены в run")
            return False
            
    except Exception as e:
        print(f"❌ Run не найден: {e}")
        return False

def relog_model_from_file():
    """
    Перезалогировать модель из файла в MLflow.
    """
    print("🔄 Перезалогирование модели из файла...")
    
    # Загрузка данных для метрик
    df = pd.read_csv('data/processed/processed.csv')
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Загрузка модели из файла
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ Нет моделей в папке models")
        return None
    
    # Берем первую модель
    model_path = f"models/{model_files[0]}"
    model = joblib.load(model_path)
    model_name = model_files[0].replace('.joblib', '').replace('_model', '')
    
    print(f"📦 Загружена модель: {model_name} из {model_path}")
    
    # Создаем новый run и логируем модель
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("telco_churn")
    
    with mlflow.start_run(run_name=f"relog_{model_name}") as run:
        # Вычисляем метрики
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Логируем метрики
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Логируем модель
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="telco_churn_model"
        )
        
        new_run_id = run.info.run_id
        print(f"✅ Модель перезалогирована с run_id: {new_run_id}")
        print(f"📊 Метрики: ROC-AUC={roc_auc:.4f}, Accuracy={accuracy:.4f}")
        
        # Обновляем файл с run_id
        with open("models/best_run_id.txt", "w") as f:
            f.write(new_run_id)
        
        return new_run_id
    
    return None

def fix_mlflow_model_registration():
    """
    Полное исправление проблемы регистрации модели.
    """
    print("🔧 Запуск исправления регистрации модели...")
    
    # 1. Проверяем текущий run_id
    if os.path.exists("models/best_run_id.txt"):
        with open("models/best_run_id.txt", "r") as f:
            run_id = f.read().strip()
        print(f"📄 Текущий run_id: {run_id}")
        
        # 2. Проверяем залогирована ли модель
        if ensure_model_logged_in_mlflow(run_id):
            print("✅ Модель уже залогирована в MLflow")
            return True
        else:
            print("⚠ Модель не залогирована, пытаемся исправить...")
    
    # 3. Перезалогировываем модель из файла
    new_run_id = relog_model_from_file()
    if new_run_id:
        print(f"✅ Модель успешно перезалогирована с run_id: {new_run_id}")
        return True
    else:
        print("❌ Не удалось перезалогировать модель")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Исправление логов модели в MLflow")
    parser.add_argument("--check", action="store_true", help="Проверить логи модели")
    parser.add_argument("--fix", action="store_true", help="Исправить логи модели")
    parser.add_argument("--run-id", type=str, help="Проверить конкретный run_id")
    
    args = parser.parse_args()
    
    if args.run_id:
        ensure_model_logged_in_mlflow(args.run_id)
    elif args.check:
        ensure_model_logged_in_mlflow(None)
    elif args.fix:
        fix_mlflow_model_registration()
    else:
        print("Использование:")
        print("  python ensure_model_logged.py --check  # Проверить логи")
        print("  python ensure_model_logged.py --fix    # Исправить логи")
        print("  python ensure_model_logged.py --run-id RUN_ID  # Проверить конкретный run")