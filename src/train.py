import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

data_path = "data/processed/processed.csv"
df = pd.read_csv(data_path)

if "target" not in df.columns:
    raise ValueError("Колонка 'target' не найдена в processed.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

mlflow.set_experiment("ufc_winner_prediction")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8, 12],
}

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(
    base_model,
    param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
)

for n_est in param_grid["n_estimators"]:
    for depth in param_grid["max_depth"]:
        with mlflow.start_run():
            print(
                f"\n🚀 Training RandomForest "
                f"(n_estimators={n_est}, max_depth={depth})"
            )
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"✅ Accuracy: {accuracy:.4f} | ROC AUC: {roc_auc:.4f}")

            mlflow.log_param("model", "RandomForestClassifier")
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.sklearn.log_model(
                model,
                "model",
            )

final_model = model
os.makedirs("models", exist_ok=True)
model_path = "models/model.joblib"
joblib.dump(final_model, model_path)
print(f"\n✅ Финальная модель сохранена: {model_path}")
