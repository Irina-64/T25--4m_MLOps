from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import os
import time

from prometheus_client import Counter, Histogram, make_asgi_app

MODEL_PATHS = [
    "/workspace/models/lgb_model.joblib",
    "./models/lgb_model.joblib"
]

MODEL_PATH = next((path for path in MODEL_PATHS if os.path.exists(path)), None)
if MODEL_PATH is None:
    raise FileNotFoundError("Модель не найдена ни по одному из путей: " + ", ".join(MODEL_PATHS))

FEATURES_PATHS = [
    "/workspace/data/processed/processed.csv",
    "./data/processed/processed.csv"
]
FEATURES_PATH = next((path for path in FEATURES_PATHS if os.path.exists(path)), None)
if FEATURES_PATH is None:
    raise FileNotFoundError("Фичи не найдены ни по одному из путей: " + ", ".join(FEATURES_PATHS))

REQUEST_COUNT = Counter(
    "request_count",
    "Количество запросов к API",
    ["endpoint", "method", "http_status"],
) 

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Задержка обработки запросов в секундах",
    ["endpoint"],
) 

PREDICTION_DISTRIBUTION = Histogram(
    "prediction_distribution",
    "Распределение вероятностей оттока",
    buckets=[i / 10.0 for i in range(11)],
) 

app = FastAPI()

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

model = joblib.load(MODEL_PATH)

if hasattr(model, "feature_name_") and model.feature_name_ is not None:
    MODEL_FEATURES = list(model.feature_name_)
else:
    features_df = pd.read_csv(FEATURES_PATH)
    MODEL_FEATURES = [col for col in features_df.columns if col not in ["user_id", "churn"]]

@app.post("/predict")
def predict(payload: dict):
    start_time = time.time()
    endpoint = "/predict"
    method = "POST"

    try:
        df = pd.DataFrame([payload])

        for col in MODEL_FEATURES:
            if col not in df.columns:
                df[col] = 0

        extra_cols = [c for c in df.columns if c not in MODEL_FEATURES]
        if extra_cols:
            df = df.drop(columns=extra_cols)

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if df.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Количество признаков не совпадает: модель обучена на {model.n_features_in_}, пришло {df.shape[1]}"
            )

        preds = model.predict_proba(df)[:, 1]
        prob = float(preds[0])

        duration = time.time() - start_time
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, http_status="200").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        PREDICTION_DISTRIBUTION.observe(prob)

        return {"churn_probability": prob}
    except Exception:
        duration = time.time() - start_time
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, http_status="500").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        raise