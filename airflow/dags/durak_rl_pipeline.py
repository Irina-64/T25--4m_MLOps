from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "student",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="durak_rl_pipeline",
    default_args=default_args,
    description="RL pipeline for Durak game",
    start_date=datetime(2025, 12, 17),
    schedule_interval=None,   # вручную
    catchup=False,
    tags=["rl", "mlops", "durak"],
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="python /opt/airflow/src/preprocess.py",
    )

    train = BashOperator(
        task_id="train",
        bash_command="python /opt/airflow/src/train.py",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="python /opt/airflow/src/evaluate.py",
    )

    preprocess >> train >> evaluate
