from datetime import datetime, timedelta
from airflow import DAG
from feast import FeatureStore, Entity, FeatureView
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
import subprocess
import traceback
import sqlite3
import sys
import os

def feast_materialize(**context):
    feast_repo_path = "/opt/airflow/feature_repo"

    if feast_repo_path not in sys.path:
        sys.path.insert(0, feast_repo_path)

    if not os.path.exists(feast_repo_path):
        raise FileNotFoundError(f"Feature repo not found: {feast_repo_path}")
    
    try:
        definitions_path = os.path.join(feast_repo_path, "definitions.py")
        if not os.path.exists(definitions_path):
            raise FileNotFoundError(f"definitions.py not found in {feast_repo_path}")
        
        sys.path.insert(0, feast_repo_path)
        try:
            import definitions        
            if hasattr(definitions, 'personality_features'):
                print(f" Найден FeatureView: personality_features")
            else:
                feature_views = []
                for attr_name in dir(definitions):
                    attr = getattr(definitions, attr_name)
                    if hasattr(attr, '__class__') and 'FeatureView' in str(attr.__class__):
                        feature_views.append(attr)
                        print(f"  - Found FeatureView: {attr_name}")
                
                if not feature_views:
                    raise ValueError("No FeatureView found in definitions.py")
                
                if 'personality_features' not in [fv.name for fv in feature_views]:
                    personality_features = feature_views[0]
                    print(f"🔄 Используем FeatureView: {personality_features.name}")
        except ImportError as e:
            print(f"❌ Ошибка импорта definitions: {e}")
            raise
        
        result = subprocess.run(
            ["feast", "apply", feast_repo_path],
            capture_output=True,
            text=True,
            cwd=feast_repo_path
        )
        
        if result.returncode != 0:
            fs = FeatureStore(repo_path=feast_repo_path) 
            objects_to_apply = []
            for attr_name in dir(definitions):
                attr = getattr(definitions, attr_name)
                if isinstance(attr, FeatureView):
                    objects_to_apply.append(attr)
                if isinstance(attr, Entity):
                    objects_to_apply.append(attr)
            
            if objects_to_apply:
                print(f"🔄 Применяем {len(objects_to_apply)} объектов...")
                fs.apply(objects_to_apply)
                print("✅ Объекты применены")
            else:
                print("⚠️  Нет объектов для применения")
        else:
            print(f"✅ Feast apply успешен: {result.stdout}")
        
        fs = FeatureStore(repo_path=feast_repo_path)
        
        try:
            fv = fs.get_feature_view("personality_features")
        except Exception as e:
            print(f"⚠️  Feature View 'personality_features' не найден: {e}")
            try:
                registry_path = os.path.join(fs.config.registry, "registry.db")
                if os.path.exists(registry_path):
                    conn = sqlite3.connect(registry_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM feature_view")
                    feature_views = cursor.fetchall()
                    print(f"Доступные Feature Views в реестре: {[fv[0] for fv in feature_views]}")
                    conn.close()
            except:
                pass
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        fs.materialize(start_date=start_date, end_date=end_date)

        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        traceback.print_exc()
        raise

with DAG(
    'feast_personality_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule='@once',
    catchup=False,
    default_args={
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
) as dag:
    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='cd /opt/airflow && python src/preprocess.py',
    )

    feast_materialize_task = PythonOperator(
        task_id='feast_materialize',
        python_callable=feast_materialize,
    )

    train_with_feast = BashOperator(
        task_id='train_with_feast',
        bash_command='cd /opt/airflow && python src/train.py --feature-store /opt/airflow/feature_repo',
        env={
            'PYTHONPATH': '/opt/airflow/src:/opt/airflow/feature_repo',
            'PATH': os.environ.get('PATH', '')
        }
    )

    evaluate = BashOperator(
        task_id='evaluate',
        bash_command='cd /opt/airflow && python src/evaluate.py',
    )
    
    save_report = BashOperator(
        task_id='save_report',
        bash_command='echo "Модель обучена на данных из Feast" > /opt/airflow/reports/feast_integration_report.txt',
    )

    preprocess >> feast_materialize_task >> train_with_feast >> evaluate >> save_report