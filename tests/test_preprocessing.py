# tests/test_preprocess.py

import pandas as pd

from src.wrapper_classes import InferenceDelay


def test_predict_returns_series():
    df = pd.DataFrame(
        [
            {
                "carrier": "PKP",
                "connection": "A-B",
                "name": "A",
                "id_main": "ID1",
                "arrival": "12:00",
                "year": 2024,
                "month": 5,
                "day": 8,
                "dayofweek": 2,
                "season": 1,
                "is_weekend": False,
                "is_holiday": False,
                "hour": 12,
            }
        ]
    )

    result = InferenceDelay.predict_delay(df)

    assert isinstance(result, pd.Series)
    assert result.name == "delay"
    assert isinstance(float(result.iloc[0]), float)
