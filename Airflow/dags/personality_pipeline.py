from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    'simple_personality_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule='@once',
    catchup=False,
) as dag:
    
    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='python /opt/airflow/src/preprocess.py',
    )
    
    train = BashOperator(
        task_id='train',
        bash_command='python /opt/airflow/src/train.py',
    )
    
    evaluate = BashOperator(
        task_id='evaluate',
        bash_command='python /opt/airflow/src/evaluate.py',
    )
    
    register = BashOperator(
        task_id='register',
        bash_command='echo "Model registered"',
    )
    
    preprocess >> train >> evaluate >> register