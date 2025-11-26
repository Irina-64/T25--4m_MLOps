from datetime import datetime

from airflow.operators.bash import BashOperator

from airflow import DAG

default_args = {"owner": "airflow", "start_date": datetime(2024, 1, 1)}

with DAG(
    "ufc_ml_pipeline", default_args=default_args, schedule_interval=None, catchup=False
) as dag:
    preprocess = BashOperator(
        task_id="preprocess", bash_command="cd /opt/airflow && python src/preprocess.py"
    )

    train = BashOperator(
        task_id="train", bash_command="cd /opt/airflow && python src/train.py"
    )

    evaluate = BashOperator(
        task_id="evaluate", bash_command="cd /opt/airflow && python src/evaluate.py"
    )

    register_alias = BashOperator(
        task_id="register_alias", bash_command="cd /opt/airflow && python src/alias.py"
    )

    preprocess >> train >> evaluate >> register_alias
