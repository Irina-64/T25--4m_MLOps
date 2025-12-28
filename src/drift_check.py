# drift_check.py - Система обнаружения дрейфа данных
import pandas as pd
import numpy as np
import json
import os
import sys
from scipy import stats
from datetime import datetime, timedelta

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI)"""
    try:
        # Определяем границы корзин
        min_val = min(np.min(expected), np.min(actual))
        max_val = max(np.max(expected), np.max(actual))
        breakpoints = np.linspace(min_val, max_val, buckets + 1)
        breakpoints[-1] = breakpoints[-1] + 0.001
        
        # Считаем гистограммы
        expected_hist, _ = np.histogram(expected, bins=breakpoints)
        actual_hist, _ = np.histogram(actual, bins=breakpoints)
        
        # Добавляем небольшое значение чтобы избежать деления на 0
        expected_hist = expected_hist.astype(float) + 0.0001
        actual_hist = actual_hist.astype(float) + 0.0001
        
        # Нормализуем до вероятностей
        expected_perc = expected_hist / np.sum(expected_hist)
        actual_perc = actual_hist / np.sum(actual_hist)
        
        # Вычисляем PSI
        psi_values = (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
        total_psi = np.sum(psi_values)
        
        return float(total_psi)
    except Exception as e:
        print(f"Ошибка расчета PSI: {e}")
        return 1.0  # Возвращаем высокое значение при ошибке

def create_test_data():
    """Создание тестовых данных если их нет"""
    print("Создание тестовых данных...")
    
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'USD_RUB': np.random.normal(90, 2, 200),
        'EUR_RUB': np.random.normal(98, 2, 200),
        'GBP_RUB': np.random.normal(115, 2, 200),
        'day_of_week': dates.dayofweek,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })
    
    # Создаем папку если её нет
    os.makedirs('data/processed', exist_ok=True)
    data.to_csv('data/processed/processed.csv', index=False)
    
    return data

def load_reference_data():
    """Загрузка эталонных данных"""
    print("1. Загрузка тренировочных данных...")
    
    try:
        # Проверяем существует ли файл
        if not os.path.exists('data/processed/processed.csv'):
            print("   Файл данных не найден. Создаю тестовые данные...")
            return create_test_data()
        
        # Загружаем данные
        train_data = pd.read_csv('data/processed/processed.csv')
        
        # Если есть колонка date, конвертируем
        if 'date' in train_data.columns:
            train_data['date'] = pd.to_datetime(train_data['date'])
        
        print(f"   ✅ Загружено {len(train_data)} записей")
        print(f"   Колонки: {list(train_data.columns)}")
        
        return train_data
        
    except Exception as e:
        print(f"   ❌ Ошибка загрузки данных: {e}")
        print("   Создаю тестовые данные...")
        return create_test_data()

def collect_production_data(days_back=7):
    """Сбор прод данных (с имитацией дрейфа)"""
    print(f"2. Сбор продакшен данных (последние {days_back} дней)...")
    
    try:
        # Пытаемся загрузить существующие данные или создаем тестовые
        ref_data = load_reference_data()
        
        # Создаем "продакшен" данные с дрейфом
        # Увеличиваем USD_RUB на 40% для демонстрации дрейфа
        production_data = ref_data.copy()
        
        # Применяем дрейф
        drift_factor = 0.4  # 40% дрейф
        production_data['USD_RUB'] = production_data['USD_RUB'] * (1 + drift_factor)
        production_data['EUR_RUB'] = production_data['EUR_RUB'] * (1 + drift_factor * 0.5)
        
        # Берем последние N дней для имитации временного окна
        if 'date' in production_data.columns:
            # Сортируем по дате
            production_data = production_data.sort_values('date')
            # Берем последние записи
            production_data = production_data.tail(100)
        else:
            # Берем случайную выборку
            production_data = production_data.sample(min(100, len(production_data)), random_state=42)
        
        print(f"   ✅ Собрано {len(production_data)} записей продакшен данных")
        print(f"   Имитирован дрейф: USD_RUB +{drift_factor*100:.0f}%, EUR_RUB +{drift_factor*50:.0f}%")
        
        return production_data
        
    except Exception as e:
        print(f"   ❌ Ошибка сбора продакшен данных: {e}")
        # Создаем простые тестовые данные
        np.random.seed(42)
        return pd.DataFrame({
            'USD_RUB': np.random.normal(120, 3, 100),  # Высокие значения для дрейфа
            'EUR_RUB': np.random.normal(105, 3, 100),
            'GBP_RUB': np.random.normal(115, 3, 100),
        })

def check_drift_for_feature(ref_data, prod_data, feature_name, psi_threshold=0.15):
    """Проверка дрейфа для конкретного признака"""
    if feature_name not in ref_data.columns or feature_name not in prod_data.columns:
        return None
    
    try:
        ref_values = ref_data[feature_name].dropna().values
        prod_values = prod_data[feature_name].dropna().values
        
        if len(ref_values) < 10 or len(prod_values) < 10:
            return None
        
        psi = calculate_psi(ref_values, prod_values)
        ks_stat, ks_pvalue = stats.ks_2samp(ref_values, prod_values)
        
        return {
            'psi': psi,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'has_drift': psi > psi_threshold,
            'reference_mean': float(np.mean(ref_values)),
            'production_mean': float(np.mean(prod_values)),
            'reference_std': float(np.std(ref_values)),
            'production_std': float(np.std(prod_values)),
            'change_percent': ((np.mean(prod_values) - np.mean(ref_values)) / np.mean(ref_values) * 100) if np.mean(ref_values) != 0 else 0
        }
    except Exception as e:
        print(f"   Ошибка проверки признака {feature_name}: {e}")
        return None

def generate_drift_report(ref_data, prod_data, psi_threshold=0.15):
    """Генерация отчета о дрейфе"""
    print("3. Генерация отчета о дрейфе...")
    
    # Определяем какие признаки проверять
    features_to_check = ['USD_RUB', 'EUR_RUB', 'GBP_RUB']
    available_features = [f for f in features_to_check if f in ref_data.columns and f in prod_data.columns]
    
    if not available_features:
        print("   ❌ Нет общих признаков для сравнения")
        return None
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'psi_threshold': psi_threshold,
        'reference_samples': len(ref_data),
        'production_samples': len(prod_data),
        'features': {},
        'summary': {
            'drift_detected': False,
            'drifted_features': [],
            'max_psi': 0.0,
            'total_features_checked': len(available_features)
        }
    }
    
    # Проверяем каждый признак
    for feature in available_features:
        print(f"   Проверка {feature}...")
        result = check_drift_for_feature(ref_data, prod_data, feature, psi_threshold)
        
        if result:
            report['features'][feature] = result
            
            if result['has_drift']:
                report['summary']['drift_detected'] = True
                report['summary']['drifted_features'].append(feature)
                report['summary']['max_psi'] = max(report['summary']['max_psi'], result['psi'])
                
                print(f"     ✅ Дрейф обнаружен! PSI={result['psi']:.3f}, изменение: {result['change_percent']:.1f}%")
            else:
                print(f"     ✓ Нет дрейфа. PSI={result['psi']:.3f}")
    
    return report

def save_report_and_trigger(report, psi_threshold=0.15):
    """Сохранение отчета и создание триггера"""
    print("\n4. Сохранение результатов...")
    
    os.makedirs('reports', exist_ok=True)
    
    # Сохраняем JSON отчет
    report_file = f"reports/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Отчет сохранен: {report_file}")
    
    # Если дрейф обнаружен, создаем триггер
    if report['summary']['drift_detected']:
        trigger_file = "reports/drift_trigger.txt"
        trigger_content = f"""ДРЕЙФ ОБНАРУЖЕН - ТРЕБУЕТСЯ ПЕРЕОБУЧЕНИЕ
==========================================
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Причина: Обнаружен значительный дрейф данных
Порог PSI: {psi_threshold}
==========================================
Детали дрейфа:
"""
        
        for feature in report['summary']['drifted_features']:
            feat_data = report['features'][feature]
            trigger_content += f"- {feature}: PSI={feat_data['psi']:.3f}, "
            trigger_content += f"изменение: {feat_data['change_percent']:.1f}% "
            trigger_content += f"(было: {feat_data['reference_mean']:.2f}, стало: {feat_data['production_mean']:.2f})\n"
        
        trigger_content += f"""==========================================
Рекомендуемые действия:
1. Запустить переобучение модели: python src/train.py
2. Оценить новую модель: python src/evaluate.py
3. Зарегистрировать модель в MLflow
=========================================="""
        
        with open(trigger_file, 'w', encoding='utf-8') as f:
            f.write(trigger_content)
        
        print(f"   ⚠️  ТРИГГЕР СОЗДАН: {trigger_file}")
        print(f"   Признаки с дрейфом: {', '.join(report['summary']['drifted_features'])}")
        print(f"   Максимальный PSI: {report['summary']['max_psi']:.3f}")
        
        return True, report_file, trigger_file
    else:
        print("   ✅ Дрейф не обнаружен, система стабильна")
        return False, report_file, None

def main():
    """Основная функция проверки дрейфа"""
    print("=" * 70)
    print("СИСТЕМА ОБНАРУЖЕНИЯ ДРЕЙФА ДАННЫХ - ЛАБОРАТОРНАЯ 12")
    print("=" * 70)
    
    PSI_THRESHOLD = 0.15
    
    # 1. Загружаем эталонные данные
    ref_data = load_reference_data()
    if ref_data is None or len(ref_data) == 0:
        print("❌ Не удалось загрузить тренировочные данные")
        return False
    
    # 2. Собираем продакшен данные
    prod_data = collect_production_data(days_back=7)
    if prod_data is None or len(prod_data) == 0:
        print("❌ Не удалось собрать продакшен данные")
        return False
    
    # 3. Генерируем отчет
    report = generate_drift_report(ref_data, prod_data, PSI_THRESHOLD)
    if report is None:
        print("❌ Не удалось сгенерировать отчет")
        return False
    
    # 4. Сохраняем результаты
    drift_detected, report_file, trigger_file = save_report_and_trigger(report, PSI_THRESHOLD)
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
    print("=" * 70)
    
    print(f"Проверено признаков: {report['summary']['total_features_checked']}")
    print(f"Дрейф обнаружен: {'ДА' if drift_detected else 'НЕТ'}")
    
    if drift_detected:
        print(f"\n⚠️  ОБНАРУЖЕН ДРЕЙФ! Требуется переобучение модели.")
        print(f"   Файл отчета: {report_file}")
        print(f"   Файл триггера: {trigger_file}")
        print(f"\nДля переобучения выполните:")
        print("   1. python src/train.py")
        print("   2. python src/evaluate.py")
        print("   3. Проверьте результаты в MLflow (http://localhost:5000)")
    else:
        print(f"\n✅ Система стабильна, дрейф не обнаружен.")
        print(f"   Файл отчета: {report_file}")
    
    print("\n" + "=" * 70)
    return drift_detected

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 1 if success else 0  # Возвращаем 1 при обнаружении дрейфа
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)