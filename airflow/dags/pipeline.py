from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

default_args = {
    "owner": "data-scientist",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="train_delay_pipeline_bash",
    default_args=default_args,
    description="ML пайплайн через BashOperator: preprocess → train → evaluate",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "train_delay", "bash"],
    max_active_runs=1,
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="python /opt/airflow/src/preprocess.py",
        cwd="/opt/airflow/src",
    )

    split = BashOperator(
        task_id="split",
        bash_command="python /opt/airflow/src/split.py",
        cwd="/opt/airflow/src",
    )

    train = BashOperator(
        task_id="train",
        bash_command="python /opt/airflow/src/train.py",
        cwd="/opt/airflow/src",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="python /opt/airflow/src/evaluate.py",
        cwd="/opt/airflow/src",
    )

    preprocess >> split >> train >> evaluate
