from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model  # type: ignore

base_dir = Path(__file__).parent.parent.parent
model_dir = base_dir / "models"
mo_path = model_dir / "TrainDelayPredictionm.keras"
preprocessors_dir = model_dir / "preprocessors"


target_path = Path(preprocessors_dir / "target_encoder.joblib")
scal_path = Path(preprocessors_dir / "standard_scaler.joblib")
catego_path = Path(preprocessors_dir / "categorical_columns.joblib")


class InferenceDelay:
    @staticmethod
    def predict_delay(
        df: pd.DataFrame,
        categorical_cols_path: Path = Path(catego_path),
        target_encoder_path: Path = Path(target_path),
        scaler_path: Path = Path(scal_path),
        model_path: Path = Path(mo_path),
    ) -> pd.Series:

        model = load_model(model_path)

        target_encoder = load(target_encoder_path)
        scaler = load(scaler_path)
        categorical_cols = load(categorical_cols_path)

        x = df.copy()

        if "hour" not in x.columns:
            raise ValueError("Обязательное поле 'hour' отсутствует")

        # 2. hour → sin / cos
        x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24)
        x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24)

        x.drop(columns=["hour"], inplace=True)

        x.fillna(0, inplace=True)

        for col in categorical_cols:
            x[col] = x[col].astype(str)

        x_te = target_encoder.transform(x)
        x_te.fillna(0, inplace=True)

        x_scaled = scaler.transform(x_te)

        preds = model.predict(x_scaled, verbose=0).flatten()

        preds_int = np.round(preds).astype("int32")

        return pd.Series(preds_int, name="delay")


#
