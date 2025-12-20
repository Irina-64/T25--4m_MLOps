import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
from datetime import datetime
from pathlib import Path

def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.nanpercentile(expected, [100 / buckets * i for i in range(buckets + 1)])
    breakpoints = np.unique(breakpoints)
    
    if len(breakpoints) < 2:
        return 0.0
    
    expected_hist, _ = np.histogram(expected, bins=breakpoints)
    actual_hist, _ = np.histogram(actual, bins=breakpoints)
    
    expected_percent = expected_hist / len(expected) + 1e-10
    actual_percent = actual_hist / len(actual) + 1e-10
    
    psi_value = np.sum((actual_percent - expected_percent) * np.log(actual_percent / expected_percent))
    return float(psi_value)

def load_production_data():
    log_path = '/opt/airflow/data/production_logs/predictions.csv'
    
    if not os.path.exists(log_path):
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        n_samples = 50
        
        data = pd.DataFrame({
            'time_broken_spent_alone': np.random.normal(5.5, 2.0, n_samples),
            'stage_fear': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'social_event_attendance': np.random.normal(4.0, 1.5, n_samples),
            'going_outside': np.random.normal(6.0, 1.5, n_samples),
            'drained_after_socializing': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'friends_circle_size': np.random.normal(13.0, 4.0, n_samples),
            'post_frequency': np.random.normal(5.0, 2.0, n_samples)
        })
        
        for col in data.columns:
            data[col] = data[col].clip(0, 20)
        
        return data
    
    try:
        data = pd.read_csv(log_path)

        required_cols = ['time_broken_spent_alone', 'stage_fear', 'social_event_attendance',
                        'going_outside', 'drained_after_socializing', 'friends_circle_size',
                        'post_frequency']
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"⚠️ В логе отсутствуют колонки: {missing_cols}")
            return pd.DataFrame()

        recent_data = data.tail(100)
        return recent_data[required_cols]
    
    except Exception as e:
        print(f"❌ Ошибка чтения лога: {e}")
        return pd.DataFrame()

def main():
    ref_path = '/opt/airflow/data/processed/train_reference.pkl'
    if not os.path.exists(ref_path):
        print(f"Файл с эталонными данными не найден: {ref_path}")
        np.random.seed(42)
        n_samples = 500
        
        reference_data = pd.DataFrame({
            'time_broken_spent_alone': np.random.normal(5.5, 2.0, n_samples),
            'stage_fear': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'social_event_attendance': np.random.normal(4.0, 1.5, n_samples),
            'going_outside': np.random.normal(6.0, 1.5, n_samples),
            'drained_after_socializing': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'friends_circle_size': np.random.normal(13.0, 4.0, n_samples),
            'post_frequency': np.random.normal(5.0, 2.0, n_samples)
        })
        
        for col in reference_data.columns:
            reference_data[col] = reference_data[col].clip(0, 20)
        
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        with open(ref_path, 'wb') as f:
            pickle.dump(reference_data, f)
        
        print(f"Созданы эталонные данные: {reference_data.shape}")
    else:
        with open(ref_path, 'rb') as f:
            reference_data = pickle.load(f)
        print(f"Загружены эталонные данные: {reference_data.shape}")

    production_data = load_production_data()
    if production_data.empty:
        print("Нет прод-данных для анализа")
        sys.exit(0)
    
    print(f"Загружены прод-данные: {production_data.shape}")

    features = ['time_broken_spent_alone', 'stage_fear', 'social_event_attendance',
                'going_outside', 'drained_after_socializing', 'friends_circle_size', 
                'post_frequency']
    
    drift_results = {}
    drift_detected = False
    psi_threshold = 0.1
    
    for feature in features:
        if feature not in reference_data.columns or feature not in production_data.columns:
            continue
        
        ref_series = reference_data[feature].dropna()
        prod_series = production_data[feature].dropna()
        
        if len(ref_series) < 10 or len(prod_series) < 10:
            continue
        
        psi_val = calculate_psi(ref_series.values, prod_series.values)
        
        drift_results[feature] = {
            'psi': psi_val,
            'threshold': psi_threshold,
            'exceeds_threshold': psi_val > psi_threshold
        }
        
        if psi_val > psi_threshold:
            print(f"⚠️  Дрейф в '{feature}': PSI={psi_val:.3f} (порог: {psi_threshold})")
            drift_detected = True
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'reference_data_shape': reference_data.shape,
        'production_data_shape': production_data.shape,
        'drift_results': drift_results,
        'drift_detected': drift_detected,
        'threshold': psi_threshold
    }
    
    report_dir = Path('/opt/airflow/reports/drift')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    if drift_detected:
        print("\nДРЕЙФ ОБНАРУЖЕН!")
        print("Рекомендуется запустить переобучение модели.")
        sys.exit(1)
    else:
        print("\nДрейф не обнаружен. Модель стабильна.")
        sys.exit(0)

if __name__ == "__main__":
    main()