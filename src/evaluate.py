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

DATA_PATH = "data/processed/processed.csv"
MODEL_PATH = "models/model.joblib"
FEATURES_PATH = "models/feature_names.json"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


df = pd.read_csv(DATA_PATH)

if "target" not in df.columns:
    raise ValueError("❌ Колонка 'target' не найдена")

y = df["target"]

DROP_COLS = ["target", "fight_id", "_R_key", "_B_key"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])


X = X.replace([float("inf"), float("-inf")], 0)

for col in X.columns:
    if X[col].dtype == "bool":
        X[col] = X[col].astype(int)

for col in X.columns:
    if X[col].dtype == "object":
        try:
            X[col] = X[col].astype(float)
        except:
            X = X.drop(columns=[col])

X = X.fillna(0)


with open(FEATURES_PATH) as f:
    train_features = json.load(f)


for col in train_features:
    if col not in X.columns:
        X[col] = 0.0


X = X[train_features]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


model = joblib.load(MODEL_PATH)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}


report_path = os.path.join(REPORT_DIR, "eval.json")
with open(report_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Evaluation report:")
print(json.dumps(metrics, indent=4))


mlflow.set_experiment("ufc_winner_prediction")

with mlflow.start_run(run_name="model_evaluation"):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

    mlflow.log_artifact(report_path)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="ufc_winner_model",
    )

client = MlflowClient()
latest = client.get_latest_versions("ufc_winner_model", stages=["None"])[0]

print(f"\n🎯 Model registered. Version: {latest.version}")
