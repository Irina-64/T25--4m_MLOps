from os import environ
from pathlib import Path
from warnings import filterwarnings, simplefilter

import joblib
import mlflow
import mlflow.keras
import mlflow.models.signature
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from split_data import load_config
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.losses import huber  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import RMSprop
from wrapper_classes.model_data_types import ModelDataTypes

environ["CUDA_VISIBLE_DEVICES"] = "-1"
environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
simplefilter(action="ignore", category=FutureWarning)
filterwarnings(action="ignore", message=".*np.object.*")

# === Пути к данным ===
base_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd()
config_dir = base_dir / "params.yaml"
train_dataset = base_dir / "data/split/train.csv"
test_dataset = base_dir / "data/split/test.csv"
validation_dataset = base_dir / "data/split/val.csv"

config = load_config(config_dir)

# === Определяем типы колонок ===
dtype = ModelDataTypes.get_dtype_dict()

# === Чтение данных ===
train_df = pd.read_csv(train_dataset, dtype=dtype)  # type: ignore
validation_df = pd.read_csv(validation_dataset, dtype=dtype)  # type: ignore
test_df = pd.read_csv(test_dataset, dtype=dtype)  # type: ignore

target_column = "delay"

# === Обработка пропусков ===
train_df.fillna(0, inplace=True)
validation_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)


print("Если не начинает работать обучение, проверьте, запущен ли mlflow server.")
# === MLflow ===
tracking_uri = config["experiments"]["tracking_uri"]
if tracking_uri.startswith("sqlite:///"):
    db_path = base_dir / tracking_uri.replace("sqlite:///", "")
    db_path.parent.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)


# === Разделяем X и y ===
def prepare_data(df, target_col):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


x_train, y_train = prepare_data(train_df, target_column)
x_val, y_val = prepare_data(validation_df, target_column)
x_test, y_test = prepare_data(test_df, target_column)

# === Определяем типы колонок ===
categorical_cols = ["carrier", "connection", "arrival", "name", "id_main"]
numerical_cols = [col for col in x_train.columns if col not in categorical_cols]

# Приведение категориальных колонок к строкам
for col in categorical_cols:
    x_train[col] = x_train[col].astype(str)
    x_val[col] = x_val[col].astype(str)
    x_test[col] = x_test[col].astype(str)

# === Target encoding для категориальных колонок ===
target_encoder = TargetEncoder(cols=categorical_cols, smoothing=0.3)
x_train_te = target_encoder.fit_transform(x_train, y_train)
x_val_te = target_encoder.transform(x_val)
x_test_te = target_encoder.transform(x_test)

# Заполняем NaN (новые категории)
x_val_te.fillna(0, inplace=True)
x_test_te.fillna(0, inplace=True)

# === Масштабирование всех колонок ===
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_te)
x_val_scaled = scaler.transform(x_val_te)
x_test_scaled = scaler.transform(x_test_te)


# === Создание модели ===
def create_model(input_dims: int) -> Sequential:
    model_layers = Sequential(
        [
            Input(shape=(input_dims,)),
            Dense(256),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(128),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(64),
            BatchNormalization(),
            Activation("relu"),
            Dense(1, activation="linear"),
        ]
    )
    model_layers.compile(
        optimizer=RMSprop(
            learning_rate=1e-4, rho=0.9, momentum=0.0, epsilon=1e-07, name="RMSprop"
        ),
        loss=huber,
        metrics=["mae", "mse"],
    )
    return model_layers


input_dim = x_train_scaled.shape[1]
model = create_model(input_dim)

# === MLflow Experiment ===
experiment_name = config["experiments"]["experiment_name"]
mlflow.set_experiment(experiment_name)

early_stopping = EarlyStopping(
    monitor="val_mae", patience=10, restore_best_weights=True
)
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)


# === Сохранение препроцессоров ===
def save_preprocessors(model_dir: Path, target_encoder, scaler, categorical_cols):
    preprocessors_dir = model_dir / "preprocessors"
    preprocessors_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(target_encoder, preprocessors_dir / "target_encoder.joblib")
    joblib.dump(scaler, preprocessors_dir / "standard_scaler.joblib")
    joblib.dump(categorical_cols, preprocessors_dir / "categorical_columns.joblib")

    print(f"Preprocessors saved to {preprocessors_dir}")

    return preprocessors_dir


model_name = config["mlflow"]["model_name"]

with mlflow.start_run() as run:
    # HYPER Параметры
    mlflow.log_params(
        {
            "input_dim": input_dim,
            "optimizer": "RMSprop",
            "learning_rate": 1e-4,
            "rho": 0.9,
            "momentum": 0.0,
            "epsilon": 1e-07,
            "batch_size": 1024,
            "train_size": len(x_train_scaled),
            "val_size": len(x_val_scaled),
            "test_size": len(x_test_scaled),
            "dropout": 0.3,
            "target_encoder_smoothing": 0.3,
        }
    )

    # Обучение
    history = model.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_val_scaled, y_val),
        epochs=300,
        batch_size=1024,
        verbose=1,
        callbacks=[early_stopping, lr_scheduler],
    )

    mlflow.log_metrics(
        {
            "train_mse": history.history["mse"][-1],
            "val_mse": history.history["val_mse"][-1],
            "train_mae": history.history["mae"][-1],
            "val_mae": history.history["val_mae"][-1],
            "train_huber": history.history["loss"][-1],
            "val_huber": history.history["val_loss"][-1],
        }
    )

    # Логируем модель в MLflow
    signature = mlflow.models.signature.infer_signature(
        x_train_scaled[:100], model.predict(x_val_scaled[:100])
    )
    mlflow.keras.log_model(  # type: ignore
        model, name="model", signature=signature, registered_model_name=model_name
    )

    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "TrainDelayPrediction.keras")

    preprocessors_dir = save_preprocessors(
        model_dir=model_dir,
        target_encoder=target_encoder,
        scaler=scaler,
        categorical_cols=categorical_cols,
    )

    mlflow.log_artifacts(str(preprocessors_dir), artifact_path="preprocessors")
