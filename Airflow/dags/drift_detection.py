from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator 
import pandas as pd
import numpy as np
import subprocess
import pickle
import sys
import os


sys.path.insert(0, '/opt/airflow/src')

def create_reference_data(**context):
    np.random.seed(42)
    n_samples = 1000
    
    reference_data = pd.DataFrame({
        'time_broken_spent_alone': np.random.normal(5.5, 2.0, n_samples),
        'stage_fear': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'social_event_attendance': np.random.normal(4.0, 1.5, n_samples),
        'going_outside': np.random.normal(6.0, 1.5, n_samples),
        'drained_after_socializing': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'friends_circle_size': np.random.normal(13.0, 4.0, n_samples),
        'post_frequency': np.random.normal(5.0, 2.0, n_samples),
        'personality_encoded': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    })

    for col in ['time_broken_spent_alone', 'social_event_attendance', 
                'going_outside', 'friends_circle_size', 'post_frequency']:
        reference_data[col] = reference_data[col].clip(0, 20)

    os.makedirs('/opt/airflow/data/processed', exist_ok=True)
    with open('/opt/airflow/data/processed/train_reference.pkl', 'wb') as f:
        pickle.dump(reference_data, f)
    
    print(f"Эталонные данные сохранены: {reference_data.shape}")
    return True

def check_drift_decision(**context):
    result = subprocess.run(
        ['python', '/opt/airflow/src/drift_check.py'],
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode == 1:
        print("Дрейф обнаружен!")
        return 'trigger_retraining'
    else:
        print("Дрейф не обнаружен")
        return 'no_drift_detected'

def simulate_drift_for_demo(**context):
    np.random.seed(999)
    n_samples = 300

    demo_data = pd.DataFrame({
        'time_broken_spent_alone': np.random.normal(8.0, 1.5, n_samples),
        'stage_fear': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'social_event_attendance': np.random.normal(2.0, 1.0, n_samples),
        'going_outside': np.random.normal(8.5, 1.0, n_samples),
        'drained_after_socializing': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'friends_circle_size': np.random.normal(5.0, 2.0, n_samples),
        'post_frequency': np.random.normal(8.0, 1.5, n_samples)
    })

    os.makedirs('/opt/airflow/data/processed', exist_ok=True)
    demo_data.to_csv('/opt/airflow/data/processed/demo_drift_data.csv', index=False)
    
    print(f"Демо-данные сохранены: {demo_data.shape}")
    return True

with DAG(
    dag_id='drift_detection_daily',
    default_args={
        'owner': 'mlops',
        'start_date': datetime(2024, 1, 1),
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Ежедневная проверка дрейфа данных',
    schedule='@daily',
    catchup=False,
    tags=['drift', 'monitoring', 'mlops']
) as dag:
    
    create_reference = PythonOperator(
        task_id='create_reference_data',
        python_callable=create_reference_data,
    )

    simulate_drift = PythonOperator(
        task_id='simulate_drift_for_demo',
        python_callable=simulate_drift_for_demo,
    )

    check_drift = BranchPythonOperator(
        task_id='check_drift',
        python_callable=check_drift_decision,
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id='trigger_retraining',
        trigger_dag_id='feast_personality_pipeline',
        conf={"trigger_reason": "data_drift_detected"},
    )

    no_drift = BashOperator(
        task_id='no_drift_detected',
        bash_command='echo "No significant drift detected"',
    )

    complete = EmptyOperator(
    task_id='complete',
    trigger_rule='none_failed',
    dag=dag,
    )

    create_reference >> simulate_drift >> check_drift
    check_drift >> [trigger_retraining, no_drift]
    [trigger_retraining, no_drift] >> complete

