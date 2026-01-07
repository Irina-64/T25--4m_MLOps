"""
flight_pipeline.py - DAG для ML пайплайна
Соответствует лабораторной работе по Airflow
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
import json
import os

# Базовые аргументы DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

def register_model_if_good(**context):
    """
    Регистрирует модель в MLflow если оценка успешна
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # 1. Читаем результаты оценки
        eval_path = '/opt/airflow/reports/eval.json'
        if not os.path.exists(eval_path):
            print(f"⚠️ Файл оценки не найден: {eval_path}")
            return "skip_registration"

        with open(eval_path, 'r') as f:
            eval_data = json.load(f)

        # 2. Проверяем метрики (можно настроить через Airflow Variables)
        metrics = eval_data.get('metrics', {})
        accuracy = metrics.get('accuracy', 0)
        roc_auc = metrics.get('roc_auc', 0)

        print(f"📊 Метрики модели:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   ROC AUC: {roc_auc:.3f}")

        # 3. Получаем пороги из Airflow Variables (если заданы)
        try:
            min_accuracy = float(Variable.get("MIN_ACCURACY", default_var=0.6))
            min_roc_auc = float(Variable.get("MIN_ROC_AUC", default_var=0.7))
        except:
            min_accuracy = 0.6
            min_roc_auc = 0.7

        print(f"📏 Пороговые значения:")
        print(f"   MIN_ACCURACY: {min_accuracy}")
        print(f"   MIN_ROC_AUC: {min_roc_auc}")

        # 4. Проверяем соответствие порогам
        if accuracy >= min_accuracy and roc_auc >= min_roc_auc:
            print("✅ Модель проходит критерии качества. Регистрируем...")

            # 5. Регистрируем в MLflow
            mlflow.set_tracking_uri("http://host.docker.internal:5000")
            client = MlflowClient()

            # Ищем эксперимент
            experiment = client.get_experiment_by_name("flight_delay")
            if not experiment:
                print("⚠️ Эксперимент 'flight_delay' не найден")
                return "skip_registration"

            # Ищем последний запуск
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=1
            )

            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"

                print(f"🔍 Найден run_id: {run_id}")
                print(f"📦 Model URI: {model_uri}")

                try:
                    # Регистрируем модель
                    registered_model = mlflow.register_model(
                        model_uri=model_uri,
                        name="flight_delay_model"
                    )
                    print(f"🎉 Модель зарегистрирована!")
                    print(f"   Имя: {registered_model.name}")
                    print(f"   Версия: {registered_model.version}")

                    return "registration_success"

                except Exception as e:
                    print(f"❌ Ошибка регистрации: {e}")
                    return "registration_failed"
            else:
                print("⚠️ Не найдено запусков в эксперименте")
                return "skip_registration"
        else:
            print("❌ Модель не проходит критерии качества")
            return "skip_registration"

    except Exception as e:
        print(f"💥 Ошибка в register_model_if_good: {e}")
        import traceback
        traceback.print_exc()
        return "error"

# Создаем DAG
with DAG(
    dag_id='flight_pipeline',
    default_args=default_args,
    description='Пайплайн для прогнозирования курсов валют (аналог задержек рейсов)',
    schedule_interval='@daily',  # Запускается ежедневно
    catchup=False,
    tags=['mlops', 'lab', 'currency'],
) as dag:

    # Стартовая задача
    start = EmptyOperator(task_id='start')

    # Задача 1: Предобработка данных
    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='cd /opt/airflow && python src/preprocess.py',
    )

    # Задача 2: Обучение модели
    train = BashOperator(
        task_id='train',
        bash_command='cd /opt/airflow && python src/train.py',
    )
    # В airflow/dags/flight_pipeline.py измените задачу evaluate:
    evaluate = BashOperator(
        task_id='evaluate',
        bash_command='cd /opt/airflow && PYTHONPATH=/root/.local/lib/python3.8/site-packages:/usr/local/lib/python3.8/site-packages:$PYTHONPATH python src/evaluate.py',
        env={
            'PYTHONPATH': '/root/.local/lib/python3.8/site-packages:/usr/local/lib/python3.8/site-packages:/opt/airflow',
        },
    )

    # Задача 4: Регистрация модели (только если оценка хорошая)
    register = PythonOperator(
        task_id='register',
        python_callable=register_model_if_good,
    )

    # Финишная задача
    end = EmptyOperator(task_id='end')
    
    # Определяем зависимости (последовательность выполнения)
    start >> preprocess >> train >> evaluate >> register >> end