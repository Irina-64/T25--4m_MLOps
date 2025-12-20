from datetime import datetime

from airflow.operators.bash import BashOperator

from airflow import DAG

with DAG(
    dag_id="retrain_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    retrain = BashOperator(
        task_id="retrain_model",
        bash_command="cd /opt/airflow && python src/train.py",
    )

    retrain
