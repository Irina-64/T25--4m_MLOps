import os
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from .wrapper_classes import InferenceDelay, PredictRequest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

base_dir = Path(__file__).parent.parent
model_dir = base_dir / "models"
model_path = model_dir / "TrainDelayPredictionm.keras"
preprocessors_dir = model_dir / "preprocessors"

target_encoder_path = Path(preprocessors_dir / "target_encoder.joblib")
scaler_path = Path(preprocessors_dir / "standard_scaler.joblib")
categorical_cols_path = Path(preprocessors_dir / "categorical_columns.joblib")

app = FastAPI(title="Train Delay Prediction API")


class PredictApi:
    @staticmethod
    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "API для предсказания задержки поездов. Документация: /docs"}

    @staticmethod
    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, int]:
        try:
            df = pd.DataFrame([request.model_dump()])
            delay_raw = InferenceDelay.predict_delay(
                df, categorical_cols_path, target_encoder_path, scaler_path, model_path
            )

            # Автоматически приводим к Python int, независимо от того, что вернула функция
            if isinstance(delay_raw, pd.Series):
                delay: int = int(delay_raw.iloc[0])
            elif hasattr(delay_raw, "item"):  # np scalar
                delay = delay_raw.item()
            else:
                delay = int(delay_raw)

            return {"delay_minutes": delay}
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            ) from e


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.210.85", port=8000)
