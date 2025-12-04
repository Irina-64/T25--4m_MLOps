from __future__ import annotations

import json
import shlex
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT_ROOT = Variable.get("project_root", "/opt/airflow")
PYTHON_BIN = Variable.get("python_bin", "python")
DRIFT_REPORT_PATH = Variable.get("drift_report_path", f"{PROJECT_ROOT}/reports/drift_report.json")
PROD_DATA_PATH = Variable.get("prod_data_path", f"{PROJECT_ROOT}/data/production/requests.csv")
CONTROL_DATA_PATH = Variable.get("control_data_path", f"{PROJECT_ROOT}/data/processed/test.csv")
MODEL_DIR = Variable.get("model_dir", f"{PROJECT_ROOT}/models/detox_model")

PSI_THRESHOLD = float(Variable.get("psi_threshold", 0.2))
KS_P_THRESHOLD = float(Variable.get("ks_pvalue_threshold", 0.05))
METRIC_DROP_THRESHOLD = float(Variable.get("metric_drop_threshold", 0.1))


def bash_python(command: str) -> str:
    project_dir = shlex.quote(PROJECT_ROOT)
    python_bin = shlex.quote(PYTHON_BIN)
    return f"cd {project_dir} && {python_bin} {command}"


default_args = {
    "owner": "team_detox",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="drift_monitoring",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 */6 * * *",  # every 6 hours
    catchup=False,
    tags=["mlops", "drift"],
) as dag:
    drift_cmd = (
        "src/drift_check.py "
        f"--train-path {shlex.quote(str(Path(PROJECT_ROOT) / 'data/raw/data.csv'))} "
        f"--prod-path {shlex.quote(PROD_DATA_PATH)} "
        f"--control-path {shlex.quote(CONTROL_DATA_PATH)} "
        f"--model-dir {shlex.quote(MODEL_DIR)} "
        f"--report-path {shlex.quote(DRIFT_REPORT_PATH)} "
        f"--psi-threshold {PSI_THRESHOLD} "
        f"--ks-pvalue-threshold {KS_P_THRESHOLD} "
        f"--metric-drop-threshold {METRIC_DROP_THRESHOLD}"
    )

    drift_check = BashOperator(
        task_id="drift_check",
        bash_command=bash_python(drift_cmd),
        append_env=True,
    )

    def choose_next_step(**_: dict) -> str:
        report_path = Path(DRIFT_REPORT_PATH)
        if not report_path.exists():
            print(f"Drift report not found at {report_path}, skipping retrain.")
            return "no_drift"
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        drift = bool(report.get("drift_detected"))
        reasons = report.get("drift_reasons", [])
        print(f"Drift detected: {drift}. Reasons: {reasons}")
        return "trigger_retrain" if drift else "no_drift"

    choose = BranchPythonOperator(
        task_id="decide",
        python_callable=choose_next_step,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="flight_pipeline",
        conf={"source": "drift_monitoring"},
    )

    no_drift = EmptyOperator(task_id="no_drift")

    drift_check >> choose >> [trigger_retrain, no_drift]
