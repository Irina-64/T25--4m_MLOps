import os
import warnings
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)
import mlflow
import mlflow.sklearn
import numpy as np

warnings.filterwarnings("ignore")


PROCESSED_DATA_PATH = "/workspace/data/processed/processed.csv"

if not os.path.exists(PROCESSED_DATA_PATH):
    raise FileNotFoundError(f"Файл не найден: {PROCESSED_DATA_PATH}")

features = pd.read_csv(PROCESSED_DATA_PATH)

X = features.drop(columns=["user_id", "churn"])
y = features["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

os.makedirs("/workspace/mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:/workspace/mlruns")
mlflow.set_experiment("_experiment")


'''const_cols = [col for col in X_train.columns if X_train[col].nunique() == 1]
X_train.drop(columns=const_cols, inplace=True)
X_test.drop(columns=const_cols, inplace=True)'''


with mlflow.start_run():
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc"
    )

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("n_estimators", 1000)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("max_depth", 7)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("accuracy", acc)

    input_example = X_train.sample(5)
    mlflow.sklearn.log_model(model, "model", input_example=input_example)

print("\n Model Evaluation:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", acc)


import joblib
joblib.dump(model, "/workspace/models/lgb_model.joblib")

