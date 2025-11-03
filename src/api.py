from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATHS = [
    "/workspace/models/lgb_model.joblib",  #путь внутри основного контейнера
    "./models/lgb_model.joblib"            #путь при локальном запуске
]

#находим существующий путь к модели
MODEL_PATH = next((path for path in MODEL_PATHS if os.path.exists(path)), None)
if MODEL_PATH is None:
    raise FileNotFoundError("Модель не найдена ни по одному из путей: " + ", ".join(MODEL_PATHS))

#пути к CSV с фичами (для получения всех признаков)
FEATURES_PATHS = [
    "/workspace/data/processed/processed.csv",
    "./data/processed/processed.csv"
]
FEATURES_PATH = next((path for path in FEATURES_PATHS if os.path.exists(path)), None)
if FEATURES_PATH is None:
    raise FileNotFoundError("Фичи не найдены ни по одному из путей: " + ", ".join(FEATURES_PATHS))

app = FastAPI()

#модель
model = joblib.load(MODEL_PATH)

#имена признаков, на которых обучалась модель
if hasattr(model, "feature_name_") and model.feature_name_ is not None:
    MODEL_FEATURES = list(model.feature_name_)
else:
    #берём все колонки из CSV кроме user_id и churn
    features_df = pd.read_csv(FEATURES_PATH)
    MODEL_FEATURES = [col for col in features_df.columns if col not in ["user_id", "churn"]]

@app.post("/predict")
def predict(payload: dict):
    """
    Пример запроса:
    {
        "all_sum": -12193.8,
        "all_mean": -80.22,
        "all_std": 10773.88,
        "all_count": 152,
        "has_gap_30d": 0,
        ...
    }
    """
    #преобразуем вход в DataFrame
    df = pd.DataFrame([payload])

    #добавляем недостающие признаки
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0  # безопасно

    #удаляем лишние признаки
    extra_cols = [c for c in df.columns if c not in MODEL_FEATURES]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    #приводим типы к числовым и заполняем NaN
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    #проверка количества признаков
    if df.shape[1] != model.n_features_in_:
        raise ValueError(f"Количество признаков не совпадает: модель обучена на {model.n_features_in_}, пришло {df.shape[1]}")

    #предсказание
    preds = model.predict_proba(df)[:, 1]
    prob = float(preds[0])

    return {"churn_probability": prob}