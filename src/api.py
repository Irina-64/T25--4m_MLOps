import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

class PersonalityFeatures(BaseModel):
    Time_broken_spent_Alone: float
    Stage_fear: int
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: int
    Friends_circle_size: float
    Post_frequency: float

class BatchPredictionRequest(BaseModel):
    samples: List[PersonalityFeatures]

app = FastAPI(title="Personality Classifier API", version="1.0.0")

try:
    model = joblib.load("models/classifier_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
except Exception as e:
    model = None
    scaler = None

@app.get("/")
async def root():
    return {
        "message": "Personality Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - предсказание для одного примера",
            "/predict_batch": "POST - предсказание для нескольких примеров",
            "/health": "GET - проверка здоровья API",
            "/model_info": "GET - информация о модели"
        }
    }

@app.get("/health")
async def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/model_info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return {
        "model_type": model.__class__.__name__,
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown",
        "features": ["Time_broken_spent_Alone","Stage_fear", "Social_event_attendance","Going_outside","Drained_after_socializing","Friends_circle_size","Post_frequency"]
    }

@app.post("/predict")
async def predict(features: PersonalityFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        data = pd.DataFrame([features.model_dump()])
        data_scaled = scaler.transform(data)

        probability = model.predict_proba(data_scaled)[0][1]
        prediction = int(model.predict(data_scaled)[0])
        
        return {
            "probability_extrovert": float(probability),
            "prediction": prediction,
            "personality": "Extrovert" if prediction == 1 else "Introvert",
            "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        data = pd.DataFrame([sample.model_dump() for sample in request.samples])
        data_scaled = scaler.transform(data)

        probabilities = model.predict_proba(data_scaled)[:, 1]
        predictions = model.predict(data_scaled)
        
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            results.append({
                "sample_id": i,
                "probability_extrovert": float(prob),
                "prediction": int(pred),
                "personality": "Extrovert" if pred == 1 else "Introvert"
            })
        
        return {
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {str(e)}")