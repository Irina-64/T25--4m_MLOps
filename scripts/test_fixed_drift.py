"""
Тестирование исправленной системы обнаружения дрейфа
"""

import subprocess
import json
import os

def run_drift_check():
    """Запуск проверки дрейфа"""
    print("Запуск drift_check.py...")
    result = subprocess.run(
        ['python', 'src/drift_check.py'],
        capture_output=True,
        text=True
    )
    
    print("="*60)
    print("ВЫВОД СКРИПТА:")
    print("="*60)
    print(result.stdout)
    
    if result.stderr:
        print("="*60)
        print("ОШИБКИ:")
        print("="*60)
        print(result.stderr)
    
    return result.returncode, result.stdout

def analyze_report():
    """Анализ отчета о дрейфе"""
    report_path = 'reports/drift_report.json'
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        print("\n" + "="*60)
        print("АНАЛИЗ ОТЧЕТА:")
        print("="*60)
        
        print(f"Время проверки: {report.get('check_timestamp')}")
        print(f"Дрейф данных: {report.get('data_drift', False)}")
        print(f"Дрейф производительности: {report.get('performance_drift', False)}")
        print(f"Общий дрейф: {report.get('any_drift', False)}")
        print(f"Рекомендация: {report.get('recommendation', 'unknown')}")
        
        # Детали дрейфа данных
        data_details = report.get('data_drift_details', {})
        if data_details:
            print(f"\nДетали дрейфа данных:")
            print(f"  Проанализировано признаков: {data_details.get('features_analyzed', 0)}")
            print(f"  Признаков с дрейфом: {data_details.get('drifted_features_count', 0)}")
            print(f"  Средний PSI: {data_details.get('avg_psi', 0):.3f}")
            
            drifted = data_details.get('drifted_features', [])
            if drifted:
                print(f"  Признаки с дрейфом: {drifted[:5]}")  # Показываем первые 5
        
        # Детали дрейфа производительности
        perf_details = report.get('performance_drift_details', {})
        if perf_details:
            print(f"\nДетали дрейфа производительности:")
            print(f"  Текущий ROC-AUC: {perf_details.get('current_roc_auc', 0):.4f}")
            print(f"  Эталонный ROC-AUC: {perf_details.get('baseline_roc_auc', 0.7):.4f}")
            print(f"  Падение: {perf_details.get('performance_drop_pct', 0):.1f}%")
    
    else:
        print(f"❌ Отчет не найден: {report_path}")

def check_data_files():
    """Проверка необходимых файлов"""
    print("="*60)
    print("ПРОВЕРКА ФАЙЛОВ:")
    print("="*60)
    
    required_files = [
        ('data/processed/processed.csv', 'Обработанные данные'),
        ('data/processed/train_reference.csv', 'Референсные данные'),
        ('models/best_model.joblib', 'Модель'),
        ('models/feature_names.joblib', 'Признаки модели'),
        ('models/scaler.joblib', 'Скалер')
    ]
    
    for file_path, description in required_files:
        if os.path.exists(file_path):
            # Дополнительная информация о файле
            if file_path.endswith('.csv'):
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    print(f"✅ {description}: {file_path} ({df.shape[0]} строк, {df.shape[1]} колонок)")
                    
                    # Проверяем наличие USD_RUB_target
                    if 'USD_RUB_target' in df.columns:
                        print(f"     ↳ USD_RUB_target есть: {df['USD_RUB_target'].value_counts().to_dict()}")
                    else:
                        print(f"     ↳ USD_RUB_target отсутствует")
                        
                except Exception as e:
                    print(f"⚠️  {description}: {file_path} (ошибка чтения: {e})")
            else:
                print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} (не найден)")

def main():
    """Основная функция"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ ИСПРАВЛЕННОЙ СИСТЕМЫ ДРЕЙФА")
    print("="*60)
    
    # 1. Проверяем файлы
    check_data_files()
    
    # 2. Запускаем проверку дрейфа
    return_code, output = run_drift_check()
    
    # 3. Анализируем отчет
    analyze_report()
    
    # 4. Интерпретируем результаты
    print("\n" + "="*60)
    print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
    print("="*60)
    
    if return_code == 0:
        print("✅ Код возврата 0: Дрейф НЕ обнаружен")
        print("   Это хорошо! Модель стабильна.")
    elif return_code == 1:
        print("🚨 Код возврата 1: Дрейф ОБНАРУЖЕН")
        print("   Система рекомендует переобучение.")
        
        # Показываем триггерный файл
        trigger_path = 'reports/retrain_trigger.txt'
        if os.path.exists(trigger_path):
            with open(trigger_path, 'r') as f:
                print(f"\nСодержимое {trigger_path}:")
                print(f.read())
    else:
        print(f"⚠️  Неизвестный код возврата: {return_code}")
    
    print("\n" + "="*60)
    print("РЕКОМЕНДАЦИИ:")
    print("="*60)
    
    if return_code == 1:
        print("1. Проверьте данные на аномалии")
        print("2. Если дрейф реальный - запустите переобучение")
        print("3. Если это ложное срабатывание - настройте пороги")
    else:
        print("Модель стабильна. Продолжайте мониторинг.")

if __name__ == "__main__":
    main()