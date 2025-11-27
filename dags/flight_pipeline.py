from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import os
import json

# Default arguments
default_args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='flight_pipeline',
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training']
) as dag:

    # Path Variables from Airflow Variables, with defaults
    DATA_RAW = Variable.get('DATA_RAW', default_var='/opt/airflow/data/raw/churn_predict.csv')
    DATA_PROCESSED = Variable.get('DATA_PROCESSED', default_var='/opt/airflow/data/processed/processed.csv')
    MODEL_PATH = Variable.get('MODEL_PATH', default_var='/opt/airflow/model.pt')
    MODELS_FOLDER = Variable.get('MODELS_FOLDER', default_var='/opt/airflow/models')
    METRIC_THRESHOLD = float(Variable.get('MODEL_REGISTRATION_THRESHOLD', default_var='0.6'))

    # Ensure directories exist
    os.makedirs(os.path.dirname(DATA_RAW), exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PROCESSED), exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    download_data = BashOperator(
        task_id='download_data',
        bash_command=(
            # run download or create dummy data if file not present
            'python /opt/airflow/src/download_data.py ' \
            f'--raw-path {DATA_RAW}'
        )
    )

    preprocess = BashOperator(
        task_id='preprocess',
        bash_command=(
            'python /opt/airflow/src/preprocess.py '
            f'--raw-path {DATA_RAW} --processed-path {DATA_PROCESSED}'
        )
    )

    train = BashOperator(
        task_id='train',
        bash_command=(
            'python /opt/airflow/src/train.py '
            f'--processed-path {DATA_PROCESSED} --model-path {MODEL_PATH}'
        )
    )

    evaluate = BashOperator(
        task_id='evaluate',
        bash_command=(
            'python /opt/airflow/src/evaluate.py '
            f'--processed-path {DATA_PROCESSED} --model-path {MODEL_PATH} --out-json /opt/airflow/data/result.json'
        )
    )

    def register_conditional(**context):
        # read metric from file or xcom, and register model if above threshold
        out_path = '/opt/airflow/data/result.json'
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                r = json.load(f)
            metric = r.get('roc_auc', 0.0)
        else:
            metric = 0.0
        if metric >= METRIC_THRESHOLD:
            # Save model into models folder
            src_model = '/opt/airflow/model.pt'
            dst = f"{MODELS_FOLDER}/model_{metric:.4f}.pt"
            if os.path.exists(src_model):
                import shutil
                shutil.copy(src_model, dst)
                return f"Model registered to {dst}"
            return 'Model not found to register'
        else:
            return f'Metric {metric:.4f} lower than threshold {METRIC_THRESHOLD}, not registering'

    register = PythonOperator(
        task_id='register',
        python_callable=register_conditional,
        provide_context=True
    )

    # Set dependencies
    download_data >> preprocess >> train >> evaluate >> register
