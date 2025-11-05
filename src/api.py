import json

import joblib
import pandas as pd
from fastapi import FastAPI, Request

app = FastAPI(title="UFC Winner Prediction API")

model_path = "models/model.joblib"
features_path = "models/feature_names.json"

model = joblib.load(model_path)

with open(features_path, "r") as f:
    feature_names = json.load(f)


@app.post("/predict")
async def predict(request: Request):
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
        return {"win_probability": round(float(preds[0]), 4)}

    except Exception as e:
        return {"error": str(e)}
