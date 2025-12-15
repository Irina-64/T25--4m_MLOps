import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

PSI_THRESHOLD = 0.2
AUC_DROP_THRESHOLD = 0.05


def calculate_psi(expected, actual, bins=10):
    quantiles = np.percentile(expected, np.linspace(0, 100, bins + 1))
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6)))
    return psi


def main():
    train = pd.read_csv('data/train.csv')
    prod = pd.read_csv('data/prod.csv')
    control = pd.read_csv('data/control.csv')

    feature_drifts = {}
    drift_detected = False

    for col in train.columns:
        if col == 'target':
            continue

        psi = calculate_psi(train[col], prod[col])
        ks_stat, ks_p = ks_2samp(train[col], prod[col])

        feature_drifts[col] = {
            'psi': float(psi),
            'ks_stat': float(ks_stat),
            'ks_pvalue': float(ks_p)
        }

        if psi > PSI_THRESHOLD:
            drift_detected = True

    auc_train = roc_auc_score(control['target'], control['pred'])
    auc_recent = roc_auc_score(prod['target'], prod['pred'])

    auc_drop = auc_train - auc_recent
    if auc_drop > AUC_DROP_THRESHOLD:
        drift_detected = True

    report = {
        'feature_drift': feature_drifts,
        'auc_train': auc_train,
        'auc_recent': auc_recent,
        'auc_drop': auc_drop,
        'drift_detected': drift_detected
    }

    with open('reports/drift_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    if drift_detected:
        print('DRIFT_DETECTED')
        sys.exit(1)
    else:
        print('NO_DRIFT')


if __name__ == '__main__':
    main()