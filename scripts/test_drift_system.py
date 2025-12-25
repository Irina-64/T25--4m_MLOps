"""
Тестирование системы обнаружения дрейфа
"""

import subprocess
import time
import os
import json

def test_drift_detection():
    """Тест обнаружения дрейфа"""
    print("Тестирование системы обнаружения дрейфа...")
    
    # 1. Запускаем проверку на оригинальных данных (дрейфа не должно быть)
    print("\n1. Проверка на оригинальных данных...")
    result = subprocess.run(['python', 'src/drift_check.py'], 
                          capture_output=True, text=True)
    
    print("Выходной код:", result.returncode)
    print("Вывод:", result.stdout[:500])
    
    if result.returncode == 0:
        print("✅ Нет дрейфа (как и ожидалось)")
    else:
        print("⚠️  Обнаружен дрейф на чистых данных (неожиданно)")
    
    # 2. Симулируем дрейф
    print("\n2. Симуляция дрейфа...")
    subprocess.run(['python', 'scripts/simulate_drift.py', '--type', 'concept'])
    
    # 3. Проверяем снова (должен обнаружить дрейф)
    print("\n3. Проверка после симуляции дрейфа...")
    time.sleep(2)
    
    result = subprocess.run(['python', 'src/drift_check.py'], 
                          capture_output=True, text=True)
    
    print("Выходной код:", result.returncode)
    if result.returncode == 1:
        print("✅ Дрейф успешно обнаружен!")
        
        # Проверяем отчет
        if os.path.exists('reports/drift_report.json'):
            with open('reports/drift_report.json', 'r') as f:
                report = json.load(f)
            print(f"   Дрейф данных: {report.get('data_drift', False)}")
            print(f"   Дрейф производительности: {report.get('performance_drift', False)}")
    else:
        print("❌ Дрейф не обнаружен (ошибка)")
    
    # 4. Восстанавливаем данные
    print("\n4. Восстановление оригинальных данных...")
    subprocess.run(['python', 'scripts/simulate_drift.py', '--type', 'reset'])
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*60)

def test_airflow_integration():
    """Тест интеграции с Airflow"""
    print("\nТестирование интеграции с Airflow...")
    
    # Проверяем существование DAG файлов
    dag_files = [
        'airflow/dags/drift_monitoring_dag.py',
        'airflow/dags/model_retraining_dag.py'
    ]
    
    for dag_file in dag_files:
        if os.path.exists(dag_file):
            print(f"✅ {dag_file} существует")
        else:
            print(f"❌ {dag_file} не найден")
    
    print("\nДля тестирования Airflow выполните:")
    print("1. Запустите Airflow: docker-compose up")
    print("2. Откройте http://localhost:8080")
    print("3. Включите DAG 'drift_monitoring'")
    print("4. Запустите DAG вручную")

if __name__ == "__main__":
    print("="*60)
    print("ТЕСТИРОВАНИЕ СИСТЕМЫ ОБНАРУЖЕНИЯ ДРЕЙФА")
    print("="*60)
    
    test_drift_detection()
    test_airflow_integration()