import argparse
import json
import os

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model_path, X_test_path, y_test_path, output_dir="reports"):
    # Загружаем данные
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    # Загружаем модель
    model = joblib.load(model_path)

    # Предсказания
    y_pred = model.predict(X_test)

    # Метрики регрессии
    metrics = {
        "rmse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    # Сохраняем отчёт
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "eval.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    with mlflow.start_run(run_name="baseline_evaluation") as run:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(json_path)
        mlflow.sklearn.log_model(model, "evaluated_model")
        # Сохраняем run_id
        with open("mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)

    print(f"✅ Evaluation complete. Report saved to {json_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", dest="X_test", required=True)
    parser.add_argument("--target_data", dest="y_test", required=True)
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.X_test, args.y_test)
