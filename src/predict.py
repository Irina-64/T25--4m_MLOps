import os
from pathlib import Path

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# === Пути ===
base_dir = Path(__file__).parent.parent
model_dir = base_dir / "models"


def input_from_terminal() -> pd.DataFrame:
    data = {
        "carrier": input("carrier: "),
        "connection": input("connection: "),
        "name": input("name: "),
        "id_main": input("id_main: "),
        "arrival": input("arrival (HH:MM): "),
        "year": int(input("year: ")),
        "month": int(input("month: ")),
        "day": int(input("day: ")),
        "dayofweek": int(input("dayofweek (0=Mon ... 6=Sun): ")),
        "season": int(input("season: ")),
        "is_weekend": input("is_weekend (true/false): ").lower() == "true",
        "is_holiday": input("is_holiday (true/false): ").lower() == "true",
        "hour": int(input("hour (0–23): ")),
    }

    return pd.DataFrame([data])


data = pd.DataFrame(
    [
        {
            "carrier": "PKP Intercity",
            "connection": "Wrocław Główny - Bydgoszcz Główna",
            "name": "Bydgoszcz Główna",
            "id_main": "ID001",
            "arrival": "21:48",
            "year": 2024,
            "month": 5,
            "day": 8,
            "dayofweek": 6,
            "season": 1,
            "is_weekend": True,
            "is_holiday": False,
            "hour": 2,
        }
    ]
)

# prediction = predict_delay(data)
# print(int(prediction.iloc[0]))
