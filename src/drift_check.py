import json
import os
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REFERENCE_PATHS = [
    "/workspace/data/processed/processed.csv",
    "data/processed/processed.csv",
]

CURRENT_PATHS = [
    "/workspace/data/processed/test_data.csv",
    "data/processed/test_data.csv",
]

MODEL_PATHS = [
    "/workspace/models/lgb_model.joblib",
    "models/lgb_model.joblib",
]

EVAL_REPORT_PATHS = [
    "/workspace/reports/eval.json",
    "reports/eval.json",
]

DRIFT_REPORT_PATHS = [
    "/workspace/reports/drift_report.json",
    "reports/drift_report.json",
]

FEATURE_DRIFT_THRESHOLD = 0.2
ROC_AUC_THRESHOLD = 0.7


def first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def psi(reference, current, bins=10):
    reference = np.asarray(reference)
    current = np.asarray(current)
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(reference, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_dist = ref_counts / max(len(reference), 1)
    cur_dist = cur_counts / max(len(current), 1)
    eps = 1e-6
    ref_dist = np.where(ref_dist == 0, eps, ref_dist)
    cur_dist = np.where(cur_dist == 0, eps, cur_dist)
    values = (ref_dist - cur_dist) * np.log(ref_dist / cur_dist)
    return float(np.sum(values))


def main():
    reference_path = first_existing_path(REFERENCE_PATHS)
    current_path = first_existing_path(CURRENT_PATHS)
    model_path = first_existing_path(MODEL_PATHS)
    eval_report_path = first_existing_path(EVAL_REPORT_PATHS)
    drift_report_path = DRIFT_REPORT_PATHS[-1]

    if reference_path is None or current_path is None or model_path is None:
        raise FileNotFoundError("Не найдены входные файлы для проверки дрейфа")

    df_ref = pd.read_csv(reference_path)
    df_cur = pd.read_csv(current_path)

    model = joblib.load(model_path)

    feature_columns = [
        c
        for c in df_ref.columns
        if c in df_cur.columns and c not in ["user_id", "churn"]
    ]

    psi_values = {}
    feature_drift = False
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(df_ref[col]):
            continue
        value = psi(df_ref[col].values, df_cur[col].values)
        psi_values[col] = value
        if value > FEATURE_DRIFT_THRESHOLD:
            feature_drift = True

    X_ref = df_ref.drop(columns=["user_id", "churn"], errors="ignore")
    y_ref = df_ref["churn"]
    proba_ref = model.predict_proba(X_ref)[:, 1]
    roc_auc = float(roc_auc_score(y_ref, proba_ref))
    metric_drift = roc_auc < ROC_AUC_THRESHOLD

    previous_roc_auc = None
    if eval_report_path is not None and os.path.exists(eval_report_path):
        with open(eval_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        previous_roc_auc = report.get("roc_auc")

    trigger_retrain = feature_drift or metric_drift

    report_data = {
        "reference_path": reference_path,
        "current_path": current_path,
        "model_path": model_path,
        "feature_drift_threshold": FEATURE_DRIFT_THRESHOLD,
        "roc_auc_threshold": ROC_AUC_THRESHOLD,
        "psi": psi_values,
        "feature_drift": feature_drift,
        "current_roc_auc": roc_auc,
        "previous_roc_auc": previous_roc_auc,
        "metric_drift": metric_drift,
        "trigger_retrain": trigger_retrain,
    }

    os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
    with open(drift_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)

    if trigger_retrain:
        subprocess.run(["python", "src/train.py"], check=True)


if __name__ == "__main__":
    main()


