import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# -------------------------
# Load features snapshot
# -------------------------
df = pd.read_parquet("data/processed/processed_feast.parquet")

y = df["target"]
X = df.drop(columns=["target", "event_timestamp", "fight_id"], errors="ignore")

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Train model
# -------------------------
model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"🎯 Accuracy: {acc:.4f}")
print(f"🎯 ROC AUC: {auc:.4f}")

# -------------------------
# Save artifacts
# -------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")

with open("models/feature_names.json", "w") as f:
    json.dump(list(X.columns), f)

with mlflow.start_run(run_name="ufc_training"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "model")

print("🚀 Training complete")
