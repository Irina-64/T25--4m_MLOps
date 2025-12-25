from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


dag = DAG(
    "drift_check",
    default_args=default_args,
    description="Drift detection and auto retrain",
    schedule_interval="0 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "drift"],
)


workspace_path = Variable.get("workspace_path", default_var="/opt/airflow/workspace")


drift_check = BashOperator(
    task_id="drift_check_task",
    bash_command=f"cd {workspace_path} && python src/drift_check.py",
    dag=dag,
)


