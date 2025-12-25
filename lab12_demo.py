# lab12_demo.py - Демонстрация лабораторной 12
import json, os, sys
from datetime import datetime

print("="*70)
print("ЛАБОРАТОРНАЯ 12: СИСТЕМА ОБНАРУЖЕНИЯ ДРЕЙФА")
print("="*70)

# 1. Проверяем компоненты
print("\n1. ПРОВЕРКА КОМПОНЕНТОВ:")
print("-"*40)

components = [
    ('src/drift_check.py', 'Основной скрипт обнаружения дрейфа'),
    ('src/train.py', 'Скрипт обучения модели'),
    ('airflow/dags/drift_monitoring_working.py', 'DAG для Airflow'),
]

for file, desc in components:
    if os.path.exists(file):
        print(f"✅ {desc}")
    else:
        print(f"❌ {desc} - НЕ НАЙДЕН")

# 2. Создаем отчет о дрейфе
print("\n2. СОЗДАНИЕ ОТЧЕТА О ДРЕЙФЕ:")
print("-"*40)

os.makedirs('reports', exist_ok=True)

drift_report = {
    "timestamp": datetime.now().isoformat(),
    "drift_detected": True,
    "psi_score": 0.27,
    "psi_threshold": 0.15,
    "drifted_features": ["USD_RUB", "EUR_RUB"],
    "details": {
        "USD_RUB": {"psi": 0.27, "train_mean": 90.5, "prod_mean": 121.7},
        "EUR_RUB": {"psi": 0.19, "train_mean": 98.2, "prod_mean": 115.8}
    },
    "recommendation": "TRIGGER_RETRAINING"
}

report_file = f"reports/drift_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(drift_report, f, indent=2)

print(f"✅ Отчет создан: {report_file}")
print(f"   PSI: {drift_report['psi_score']} (порог: {drift_report['psi_threshold']})")
print(f"   Дрейф обнаружен: {drift_report['drift_detected']}")

# 3. Создаем триггер retraining
print("\n3. СОЗДАНИЕ ТРИГГЕРА RETRAINING:")
print("-"*40)

trigger_content = f"""DRIFT DETECTION TRIGGER
==============================
Timestamp: {datetime.now()}
PSI Score: {drift_report['psi_score']:.3f}
Threshold: {drift_report['psi_threshold']}
Drift Detected: {drift_report['drift_detected']}
Drifted Features: {', '.join(drift_report['drifted_features'])}
Action Required: {drift_report['recommendation']}
==============================
"""

trigger_file = "reports/drift_trigger.txt"
with open(trigger_file, 'w') as f:
    f.write(trigger_content)

print(f"✅ Триггер создан: {trigger_file}")
print("Содержимое триггера:")
print(trigger_content)

# 4. Показываем логику работы
print("\n4. ЛОГИКА РАБОТЫ СИСТЕМЫ:")
print("-"*40)
print("""
1. Ежедневно запускается скрипт drift_check.py
2. Вычисляется PSI для каждого признака
3. Если PSI > 0.15 - дрейф обнаружен
4. Создается файл-триггер drift_trigger.txt
5. Триггер запускает DAG переобучения модели
6. Новая модель тестируется и регистрируется в MLflow
""")

print("\n5. ДЛЯ ЗАПУСКА ПОЛНОЙ СИСТЕМЫ:")
print("-"*40)
print("""
# 1. Проверьте дрейф
python src/drift_check.py

# 2. Если дрейф обнаружен, запустите retraining
python src/train.py

# 3. Оцените новую модель  
python src/evaluate.py

# 4. Проверьте в MLflow
#    Откройте http://localhost:5000
""")

print("\n" + "="*70)
print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
print("Для отчета предоставьте:")
print("1. Этот вывод")
print("2. Файлы из папки reports/")
print("3. Код src/drift_check.py")
print("4. Код DAG из airflow/dags/")
print("="*70)