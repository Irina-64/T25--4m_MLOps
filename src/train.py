import argparse
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def load_config(config_path="params.yaml"):
    """Загрузка конфигурации"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config):
    """Загрузка подготовленных данных"""
    split_dir = config["data"]["split_dir"]

    X_train = pd.read_csv(f"{split_dir}/X_train.csv")
    X_test = pd.read_csv(f"{split_dir}/X_test.csv")
    y_train = pd.read_csv(f"{split_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{split_dir}/y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


def calculate_metrics_regression(y_true, y_pred):
    """Расчет метрик регрессии"""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def plot_feature_importance(model, feature_names, model_name):
    """Визуализация важности признаков"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Топ-15 признаков

        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importance - {model_name}")
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()

        plot_path = f"reports/feature_importance_{model_name.lower()}.png"
        os.makedirs("reports", exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return plot_path
    return None


def train_model(model_type, model_params, X_train, y_train, X_test, y_test, feature_names):
    """Обучение модели регрессии с логированием в MLflow"""

    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Логируем параметры
        mlflow.log_params(model_params)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset", "flight_delays")

        # Создаем модель
        if model_type == "random_forest":
            model = RandomForestRegressor(**model_params)
            mlflow.sklearn.autolog()  # type: ignore
        elif model_type == "xgboost":
            model = xgb.XGBRegressor(**model_params)
            mlflow.xgboost.autolog()  # type: ignore
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Обучение
        print(f"Training {model_type} model...")
        model.fit(X_train, y_train)

        # Предсказания
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Рассчитываем метрики
        train_metrics = calculate_metrics_regression(y_train, y_pred_train)
        test_metrics = calculate_metrics_regression(y_test, y_pred_test)

        # Логируем метрики
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)

        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        # Логируем overfitting индикатор (разница RMSE)
        mlflow.log_metric("overfitting_rmse", train_metrics["rmse"] - test_metrics["rmse"])

        # Логируем важность признаков
        fi_path = plot_feature_importance(model, feature_names, model_type)
        if fi_path:
            mlflow.log_artifact(fi_path)

        # Сохраняем модель
        model_path = f"models/{model_type}_model.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n{model_type.upper()} Results:")
        print("Train Metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("Test Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        return model, test_metrics["rmse"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Config file path")
    parser.add_argument("--model", help="Specific model to train (random_forest, xgboost)")
    args = parser.parse_args()

    # Загружаем конфигурацию
    config = load_config(args.config)

    # Настраиваем MLflow
    if config["experiments"]["tracking_uri"]:
        mlflow.set_tracking_uri(config["experiments"]["tracking_uri"])

    experiment_name = config["experiments"]["experiment_name"]
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    print(f"experiment {experiment_id}")

    mlflow.set_experiment(experiment_name)

    # Загружаем данные
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(config)
    feature_names = X_train.columns.tolist()

    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Обучаем модели
    results = {}
    models_to_train = [args.model] if args.model else config["models"].keys()

    for model_type in models_to_train:
        if model_type not in config["models"]:
            print(f"Warning: {model_type} not found in config")
            continue

        model_params = config["models"][model_type]

        # Берем первые значения из списков параметров для базового обучения
        base_params = {}
        for param, value in model_params.items():
            if isinstance(value, list):
                base_params[param] = value[0]
            else:
                base_params[param] = value

        model, score = train_model(model_type, base_params, X_train, y_train, X_test, y_test, feature_names)
        results[model_type] = score

    # Выводим сравнение результатов (по RMSE)
    print("\n" + "=" * 50)
    print("COMPARISON OF MODELS:")
    print("=" * 50)
    for model_type, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"{model_type:20}: RMSE = {score:.4f}")


if __name__ == "__main__":
    main()
