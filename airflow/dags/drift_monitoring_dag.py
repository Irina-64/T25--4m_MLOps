from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="drift_monitoring",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/10 * * * *",  # ⏱ каждые 10 минут
    catchup=False,
) as dag:
    run_drift_check = BashOperator(
        task_id="run_drift_check",
        bash_command="cd /opt/airflow && python src/drift_check.py",
    )

    run_drift_check
