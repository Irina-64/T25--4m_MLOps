from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime

with DAG(
    dag_id='drift_monitor',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@hourly',
    catchup=False
) as dag:

    check_drift = BashOperator(
        task_id='check_drift',
        bash_command='python src/drift_check.py'
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id='trigger_retrain',
        trigger_dag_id='model_retrain'
    )

    check_drift >> trigger_retrain