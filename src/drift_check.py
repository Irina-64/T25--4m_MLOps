import json
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ======================
# CONFIG
# ======================
PSI_THRESHOLD = 0.25
ROC_AUC_THRESHOLD = 0.75

TRAIN_PATH = "data/processed/train.csv"
PROD_PATH = "data/processed/prod_recent.csv"
MODEL_PATH = "models/model.joblib"
REPORT_PATH = "reports/drift_report.json"


# ======================
# PSI FUNCTION (SAFE)
# ======================
def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)

    if expected.nunique() < 2 or actual.nunique() < 2:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_percents = expected_counts / max(len(expected), 1)
    actual_percents = actual_counts / max(len(actual), 1)

    return float(
        np.sum(
            (expected_percents - actual_percents)
            * np.log((expected_percents + 1e-6) / (actual_percents + 1e-6))
        )
    )


# ======================
# MAIN
# ======================
def main():
    train = pd.read_csv(TRAIN_PATH)
    prod = pd.read_csv(PROD_PATH)

    model = joblib.load(MODEL_PATH)

    # =====================================================
    # MODEL FEATURE SPACE (STRICT)
    # =====================================================
    model_features = model.booster_.feature_name()

    for col in model_features:
        if col not in prod.columns:
            prod[col] = 0

    X_prod = prod[model_features].copy()

    # force numeric
    for col in X_prod.columns:
        if not pd.api.types.is_numeric_dtype(X_prod[col]):
            X_prod[col] = 0.0

    # =====================================================
    # PERFORMANCE DRIFT
    # =====================================================
    y_true = prod["target"]
    y_pred = model.predict_proba(X_prod)[:, 1]
    roc_auc = roc_auc_score(y_true, y_pred)

    drift_detected = roc_auc < ROC_AUC_THRESHOLD

    # =====================================================
    # FEATURE DRIFT (PSI — ONLY CONTINUOUS)
    # =====================================================
    psi_results = {}

    psi_features = [
        c
        for c in train.columns
        if (
            c in prod.columns
            and c != "target"
            and pd.api.types.is_float_dtype(train[c])
        )
    ]

    for col in psi_features:
        value = psi(train[col], prod[col])
        psi_results[col] = value
        if value > PSI_THRESHOLD:
            drift_detected = True

    # =====================================================
    # REPORT
    # =====================================================
    report = {
        "roc_auc": roc_auc,
        "roc_auc_threshold": ROC_AUC_THRESHOLD,
        "psi_threshold": PSI_THRESHOLD,
        "drift_detected": drift_detected,
        "psi": psi_results,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    # =====================================================
    # AUTO RETRAIN
    # =====================================================
    if drift_detected:
        print("\n🚨 Drift detected — triggering retrain pipeline")
        subprocess.run(
            ["airflow", "dags", "trigger", "retrain_pipeline", "ufc_ml_pipeline"],
            check=False,
        )
    else:
        print("\n✅ No significant drift detected")


if __name__ == "__main__":
    main()
