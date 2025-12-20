import pandas as pd
import numpy as np
import pickle
import json
from scipy import stats
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

def detect_feature_drift(reference_data, production_data, features=None):
    if features is None:
        features = [col for col in reference_data.columns if col in production_data.columns]
    
    drift_results = {}
    psi_threshold = 0.1
    ks_threshold = 0.05
    
    for feature in features:
        if feature not in reference_data.columns or feature not in production_data.columns:
            continue
        
        ref_series = reference_data[feature].dropna()
        prod_series = production_data[feature].dropna()
        
        if len(ref_series) < 10 or len(prod_series) < 10:
            continue

        psi_val = calculate_psi(ref_series.values, prod_series.values)
        ks_stat, ks_pvalue = stats.ks_2samp(ref_series.values, prod_series.values)
        
        drift_results[feature] = {
            'psi': psi_val,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'psi_exceeds_threshold': psi_val > psi_threshold,
            'ks_exceeds_threshold': ks_pvalue < ks_threshold,
            'drift_detected': (psi_val > psi_threshold) or (ks_pvalue < ks_threshold)
        }
    
    return drift_results

def save_drift_report(drift_results, output_path):
    report = {
        'timestamp': datetime.now().isoformat(),
        'drift_results': drift_results,
        'summary': {
            'total_features': len(drift_results),
            'features_with_drift': sum(1 for r in drift_results.values() if r['drift_detected']),
            'drift_detected': any(r['drift_detected'] for r in drift_results.values())
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report