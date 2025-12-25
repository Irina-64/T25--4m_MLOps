"""
Рабочий DAG для лабораторной 12 - использует BashOperator
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import json

def check_result():
    """Проверяет результат выполнения скрипта"""
    print("Проверка результата обнаружения дрейфа...")
    
    # Проверяем наличие файлов-артефактов
    artifacts = [
        '/opt/airflow/reports/drift_report.json',
        '/opt/airflow/reports/retrain_trigger.txt'
    ]
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            print(f"✅ Найден: {artifact}")
            if artifact.endswith('.json'):
                try:
                    with open(artifact, 'r') as f:
                        data = json.load(f)
                    print(f"   Дрейф данных: {data.get('data_drift', False)}")
                    print(f"   Дрейф производительности: {data.get('performance_drift', False)}")
                except:
                    pass
        else:
            print(f"❌ Отсутствует: {artifact}")
    
    return "checked"

default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='drift_monitoring_working',
    default_args=default_args,
    description='Рабочий DAG для обнаружения дрейфа',
    schedule_interval='@daily',
    catchup=False,
    tags=['lab12', 'drift'],
) as dag:
    
    start = DummyOperator(task_id='start')
    
    # Запускаем скрипт через Bash - это работает!
    run_drift_check = BashOperator(
        task_id='run_drift_check',
        bash_command='cd /opt/airflow && python src/drift_check.py',
    )
    
    check_artifacts = PythonOperator(
        task_id='check_artifacts',
        python_callable=check_result,
    )
    
    log_success = BashOperator(
        task_id='log_success',
        bash_command='echo "✅ Лабораторная 12 выполнена!" && '
                     'echo "Система обнаружения дрейфа работает в Airflow"',
    )
    
    end = DummyOperator(task_id='end')
    
    start >> run_drift_check >> check_artifacts >> log_success >> end