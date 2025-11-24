from __future__ import annotations

import shlex
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Variable.get("project_root", "/opt/airflow")
PYTHON_BIN = Variable.get("python_bin", "python")
EVAL_REPORT_PATH = Variable.get("eval_report_path", f"{PROJECT_ROOT}/reports/eval.json")
TRAINED_MODEL_PATH = Variable.get("trained_model_path", f"{PROJECT_ROOT}/models/detox_model")
REGISTERED_MODEL_PATH = Variable.get(
    "registered_model_path", f"{PROJECT_ROOT}/models/production/detox_model"
)
BLEU_THRESHOLD = float(Variable.get("bleu_threshold", 0.20))


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
    dag_id="flight_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "lab8", "detox"],
) as dag:

    download_data = BashOperator(
        task_id="download_data",
        bash_command=bash_python("src/download_data.py"),
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=bash_python("src/preprocess.py"),
    )

    train = BashOperator(
        task_id="train",
        bash_command=bash_python("src/train.py"),
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=bash_python("src/evaluate_t5.py"),
    )

    register_cmd = (
        "src/register.py "
        f"--model-path {shlex.quote(TRAINED_MODEL_PATH)} "
        f"--registry-path {shlex.quote(REGISTERED_MODEL_PATH)} "
        f"--report-path {shlex.quote(EVAL_REPORT_PATH)} "
        f"--metric bleu --threshold {BLEU_THRESHOLD}"
    )

    register = BashOperator(
        task_id="register",
        bash_command=bash_python(register_cmd),
    )

    download_data >> preprocess >> train >> evaluate >> register