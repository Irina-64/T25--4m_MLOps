from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/model.joblib")

app = FastAPI(title="Flight Delay Predictor")

model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    proba = model.predict_proba(df)[:, 1][0]
    return {"delay_probability": float(proba)}

