import argparse
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError("❌ 'target' missing in processed.csv")

    drop_cols = ["target", "fight_id", "_R_key", "_B_key"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df["target"]

    return X, y


def clean_for_model(X):
    print("\n🧹 Cleaning data...")

    X = X.replace([float("inf"), float("-inf")], 0)

    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)

    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = X[col].astype(float)
            except:
                print(f"⚠️ Dropping non-numeric column: {col}")
                X = X.drop(columns=[col])

    X = X.fillna(0)

    return X


def train_model(X_train, y_train):
    print("🚀 Training LightGBM...")

    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/processed.csv")
    args = parser.parse_args()

    print("📥 Loading processed.csv...")
    X, y = load_data(args.data)

    print("🧼 Cleaning...")
    X = clean_for_model(X)

    print("📏 Train/test split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    print(f"\n🎯 Accuracy: {acc:.4f}")
    print(f"🎯 ROC AUC: {auc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")

    with open("models/feature_names.json", "w") as f:
        json.dump(list(X.columns), f)

    print("\n💾 Model saved → models/model.joblib")
    print("💾 Feature names saved → models/feature_names.json")

    mlflow.set_experiment("ufc_prediction_lgbm")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
