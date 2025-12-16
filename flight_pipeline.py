from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'flight_ml_pipeline',
    default_args=default_args,
    description='ML Pipeline for flight predictions',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:
    
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')
    
    # Task 1: Препроцессинг
    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='cd /opt/airflow/src && python preprocess.py',
    )
    
    # Task 2: Обучение
    train = BashOperator(
        task_id='train',
        bash_command='cd /opt/airflow/src && python train.py',
    )
    
    # Task 3: Оценка
    evaluate = BashOperator(
        task_id='evaluate',
        bash_command='cd /opt/airflow/src && python evaluate.py',
    )
    
    # Task 4: Регистрация модели
    register = BashOperator(
        task_id='register',
        bash_command='echo "Registering model..."',
    )
    
    # Порядок выполнения
    start >> preprocess >> train >> evaluate >> register >> end