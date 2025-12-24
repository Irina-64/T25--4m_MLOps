import dataclasses

import numpy as np


@dataclasses.dataclass
class ModelDataTypes:
    carrier: str
    connection: str
    arrival: str
    name: str
    id_main: str

    delay: float
    year: float
    month: float
    day: float
    dayofweek: float
    season: float
    hour_sin: float
    hour_cos: float

    is_weekend: bool
    is_holiday: bool

    @classmethod
    def get_dtype_dict(cls):
        """
        Возвращает словарь {column_name: numpy/pandas dtype},
        совместимый с Keras/TensorFlow.
        """
        dtype_dict = {
            "carrier": object,
            "connection": object,
            "arrival": object,
            "name": object,
            "id_main": object,
            "delay": np.float32,
            "year": np.float32,
            "month": np.float32,
            "day": np.float32,
            "dayofweek": np.float32,
            "season": np.float32,
            "hour_sin": np.float32,
            "hour_cos": np.float32,
            "is_weekend": bool,
            "is_holiday": bool,
        }
        return dtype_dict
