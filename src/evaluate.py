import json
from os import environ
from pathlib import Path
from warnings import filterwarnings, simplefilter

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from category_encoders import TargetEncoder
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # type: ignore
from wrapper_classes.model_data_types import ModelDataTypes

environ["CUDA_VISIBLE_DEVICES"] = "-1"
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
simplefilter(action="ignore", category=FutureWarning)
filterwarnings(action="ignore", message=".*np.object.*")

# =============================================================================
# Load config
# =============================================================================
base_dir = Path(__file__).parent.parent
with open(base_dir / "params.yaml") as f:
    config = yaml.safe_load(f)

train_dataset = Path(config["data"]["split"]) / "train.csv"
val_dataset = Path(config["data"]["split"]) / "val.csv"
model_path = base_dir / "models" / "TrainDelayPrediction.keras"
artifacts_dir = Path("reports")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MLflow
# =============================================================================
experiment_name = config["experiments"]["experiment_name"]
tracking_uri = config["experiments"]["tracking_uri"]
if tracking_uri.startswith("sqlite:///"):
    db_path = base_dir / tracking_uri.replace("sqlite:///", "")
    db_path.parent.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name=experiment_name)
model_name = config["mlflow"]["model_name"]


# =============================================================================
# Column types and target
# =============================================================================
dtype = ModelDataTypes.get_dtype_dict()
target_column = config["data"]["target_column"]
categorical_cols = config["encoding"]["categorical_cols"]

# =============================================================================
# Load datasets
# =============================================================================
train_df = pd.read_csv(train_dataset, dtype=dtype).fillna(0)  # type: ignore
val_df = pd.read_csv(val_dataset, dtype=dtype).fillna(0)  # type: ignore

x_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
x_val = val_df.drop(columns=[target_column])
y_val = val_df[target_column]

for col in categorical_cols:
    x_train[col] = x_train[col].astype(str)
    x_val[col] = x_val[col].astype(str)

# =============================================================================
# Encoding + Scaling
# =============================================================================
target_encoder = TargetEncoder(
    cols=categorical_cols, smoothing=config["preprocess"]["target_encoder_smoothing"]
)
x_train_te = target_encoder.fit_transform(x_train, y_train)
x_val_te = target_encoder.transform(x_val).fillna(0)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_te)
x_val_scaled = scaler.transform(x_val_te)

# =============================================================================
# Load model
# =============================================================================
model = load_model(model_path)
y_pred = model.predict(x_val_scaled).reshape(-1)

# =============================================================================
# Metrics
# =============================================================================
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

y_val_clip = np.clip(y_val, 0, None)
y_pred_clip = np.clip(y_pred, 0, None)

rmsle = np.sqrt(mean_squared_log_error(y_val_clip, y_pred_clip))
medae = median_absolute_error(y_val, y_pred)
max_err = max_error(y_val, y_pred)
residuals = y_val - y_pred

eval_df = x_val.copy()
eval_df["residuals"] = residuals
eval_df[target_column] = y_val

metrics = {
    "mae": float(mae),
    "mse": float(mse),
    "rmse": float(rmse),
    "r2": float(r2),
    "rmsle": float(rmsle),
    "median_absolute_error": float(medae),
    "max_error": float(max_err),
}
# =============================================================================
# Plots
# =============================================================================
path_1 = artifacts_dir / "true_vs_predicted.png"
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.5, color="tab:orange", label="Predicted")
sns.lineplot(
    x=[y_val.min(), y_val.max()],
    y=[y_val.min(), y_val.max()],
    color="tab:blue",
    linestyle="--",
    label="Ideal",
)
plt.xlabel("True delay (min)")
plt.ylabel("Predicted delay (min)")
plt.title("True vs Predicted delay")
plt.legend()
plt.tight_layout()
plt.savefig(path_1)
plt.close()

path_2 = artifacts_dir / "residuals_vs_predicted.png"
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color="tab:red", label="Residuals")
plt.axhline(0, color="tab:blue", linestyle="--", label="Zero error")
plt.xlabel("Predicted delay (min)")
plt.ylabel("Residual (min)")
plt.title("Residuals vs Predicted delay")
plt.legend()
plt.tight_layout()
plt.savefig(path_2)
plt.close()
"""
path_3 = artifacts_dir / "residuals_categorical"
categorical_features = config["encoding"]["categorical_cols"]
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=eval_df[feature], y=eval_df["residuals"])
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel(feature)
    plt.ylabel("Residuals")
    plt.title(f"Residuals by {feature}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path_3 / f"residuals_by_{feature}.png")
    plt.close()

path_4 = artifacts_dir / "residuals_numerical"
numerical_features = config["scaling"]["numerical_cols"]
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=eval_df[feature], y=eval_df["residuals"], alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel(feature)
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs {feature}")
    plt.tight_layout()
    plt.savefig(path_4 / f"residuals_vs_{feature}.png")
    plt.close()
"""
# =============================================================================
# MLflow logging
# =============================================================================
with mlflow.start_run():
    mlflow.log_metrics(metrics=metrics)
    mlflow.log_artifact(str(path_1))
    mlflow.log_artifact(str(path_2))

# =============================================================================
# Json logging
# =============================================================================
eval_json_path = config["mlflow"]["json_path"]
if not config["mlflow"]["json_path"]:
    eval_json_path = artifacts_dir / "eval.json"
with open(eval_json_path, "w") as f:
    json.dump(metrics, f, indent=4)
# =============================================================================
# Console output
# =============================================================================
print("Evaluation on val set")
print(f"MAE   : {mae:.4f}")
print(f"MSE   : {mse:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"R2    : {r2:.4f}")
print(f"RMSLE : {rmsle:.4f}")
print(f"MedAE : {medae:.4f}")
print(f"MaxErr: {max_err:.4f}")

print("MLflow tracking URI:", mlflow.get_tracking_uri(), flush=True)
with mlflow.start_run() as run:
    print(
        "run_id:",
        run.info.run_id,
        "artifact_uri:",
        mlflow.get_artifact_uri(),
        flush=True,
    )
