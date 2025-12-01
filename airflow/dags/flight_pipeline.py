from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import json
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'flight_pipeline',
    default_args=default_args,
    description='ETL → Training → Evaluation → Registration Pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'pipeline'],
)

# Получение путей из переменных Airflow или значения по умолчанию
workspace_path = Variable.get("workspace_path", default_var="/opt/airflow/workspace")
threshold_roc_auc = float(Variable.get("threshold_roc_auc", default_var="0.7"))

# Задача 1: Загрузка данных (если нужно)
download_data = BashOperator(
    task_id='download_data',
    bash_command=f'cd {workspace_path} && dvc pull data/raw/train_data.csv.dvc || echo "Data already exists or DVC not configured"',
    dag=dag,
)

# Задача 2: Предобработка данных
preprocess = BashOperator(
    task_id='preprocess',
    bash_command=f'cd {workspace_path} && python src/preprocess.py',
    dag=dag,
)

# Задача 3: Обучение модели
train = BashOperator(
    task_id='train',
    bash_command=f'cd {workspace_path} && python src/train.py',
    dag=dag,
)

# Задача 4: Оценка модели
evaluate = BashOperator(
    task_id='evaluate',
    bash_command=f'cd {workspace_path} && python src/evaluate.py',
    dag=dag,
)

# Задача 5: Регистрация модели (при успешной оценке)
def register_model(**context):
    """Регистрирует модель в MLflow при условии успешной оценки"""
    report_path = os.path.join(workspace_path, "reports/eval.json")
    
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report not found: {report_path}")
    
    with open(report_path, 'r') as f:
        metrics = json.load(f)
    
    roc_auc = metrics.get("roc_auc", 0.0)
    
    if roc_auc < threshold_roc_auc:
        raise ValueError(f"Model ROC-AUC {roc_auc} is below threshold {threshold_roc_auc}")
    
    import mlflow
    mlflow.set_tracking_uri(f"file:{os.path.join(workspace_path, 'mlruns')}")
    mlflow.set_experiment("_experiment")
    
    with mlflow.start_run(run_name=f"registered_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        model_path = os.path.join(workspace_path, "models/lgb_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        import joblib
        import pandas as pd
        model = joblib.load(model_path)
        
        processed_path = os.path.join(workspace_path, "data/processed/processed.csv")
        df = pd.read_csv(processed_path)
        X = df.drop(columns=["user_id", "churn"], errors="ignore")
        input_example = X.sample(min(5, len(X))) if len(X) > 0 else X.head(1)
        
        mlflow.log_param("model", "LightGBM")
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", metrics.get("precision", 0))
        mlflow.log_metric("recall", metrics.get("recall", 0))
        mlflow.log_param("status", "registered")
        
        mlflow.sklearn.log_model(model, "model", input_example=input_example)
        
        model_uri = mlflow.get_artifact_uri("model")
        print(f"Model registered successfully: {model_uri}")

register = PythonOperator(
    task_id='register',
    python_callable=register_model,
    dag=dag,
)

# Определение зависимостей между задачами
download_data >> preprocess >> train >> evaluate >> register
