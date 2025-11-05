import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

data_path = "data/processed/processed.csv"
df = pd.read_csv(data_path)

if "target" not in df.columns:
    raise ValueError("Колонка 'target' не найдена в processed.csv")

X = df.drop(columns=["target"])
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("ufc_winner_prediction")

# === Обучение модели ===
print("🚀 Training RandomForest pipeline")
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"✅ Accuracy: {accuracy:.4f} | ROC AUC: {roc_auc:.4f}")


os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
with open("models/feature_names.json", "w") as f:
    json.dump(list(X_train.columns), f)

print("✅ Финальная pipeline-модель сохранена: models/model.joblib")
print("✅ Список признаков сохранён: models/feature_names.json")

with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.sklearn.log_model(model, "model")


with open("models/feature_names.json", "w") as f:
    json.dump(list(X_train.columns), f)

print("✅ Список признаков сохранён: models/feature_names.json")
