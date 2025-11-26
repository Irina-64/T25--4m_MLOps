import json
import os

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

data_path = "data/processed/processed.csv"
model_path = "models/model.joblib"
report_dir = "reports"
os.makedirs(report_dir, exist_ok=True)

df = pd.read_csv(data_path)
if "target" not in df.columns:
    raise ValueError("Колонка 'target' не найдена в processed.csv")

X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load(model_path)

y_pred = model.predict(X_test)
y_proba = (
    model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
)

metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

report_path = os.path.join(report_dir, "eval.json")
with open(report_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n✅ Отчёт сохранён в: {report_path}")
print(json.dumps(metrics, indent=4))

mlflow.set_experiment("ufc_winner_prediction")

with mlflow.start_run(run_name="model_evaluation") as run:
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            mlflow.log_metric(key, val)

    mlflow.log_artifact(report_path)

    mlflow.sklearn.log_model(
        model, artifact_path="model", registered_model_name="ufc_winner_model"
    )

client = MlflowClient()
latest = client.get_latest_versions("ufc_winner_model", stages=["None"])[0]

print(f"🎯 Модель успешно зарегистрирована! Версия: {latest.version}")
