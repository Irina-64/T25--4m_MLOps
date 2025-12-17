import json
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Summary,
    generate_latest,
)

REQUEST_COUNT = Counter("request_count", "Total number of requests", ["endpoint"])

REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

PREDICTION_DISTRIBUTION = Summary(
    "prediction_distribution", "Model prediction probabilities"
)


app = FastAPI(title="UFC Winner Prediction API")

model_path = "models/model.joblib"
features_path = "models/feature_names.json"

model = joblib.load(model_path)

with open(features_path, "r") as f:
    feature_names = json.load(f)


@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    try:
        data = await request.json()
        df = pd.DataFrame([data])

        df = pd.get_dummies(df, drop_first=True)

        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_names]
        df = df.astype(float)

        preds = model.predict_proba(df)[:, 1]
        prob = float(preds[0])

        PREDICTION_DISTRIBUTION.observe(prob)

        return {"win_probability": round(prob, 4)}

    except Exception as e:
        return {"error": str(e)}

    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
