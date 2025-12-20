import joblib
import uvicorn
import numpy as np
import pandas as pd
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List
from prometheus_client import Counter, Histogram, Summary, Gauge, generate_latest, REGISTRY, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio


REQUEST_COUNT = Counter(
    'personality_api_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'personality_api_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

PREDICTION_PROBABILITY = Summary(
    'personality_api_prediction_probability',
    'Prediction probability distribution',
    ['personality_type']
)

PREDICTION_DISTRIBUTION = Histogram(
    'personality_api_prediction_distribution',
    'Detailed prediction distribution',
    ['personality_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MODEL_LOADED = Gauge('personality_api_model_loaded', 'Model loaded status')
UPTIME = Gauge('personality_api_uptime_seconds', 'API uptime in seconds')
REQUEST_IN_PROGRESS = Gauge('personality_api_requests_in_progress', 'Requests in progress')

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        method = request.method
        endpoint = request.url.path
        REQUEST_IN_PROGRESS.inc()
        
        try:
            start_time = time.time()
            response = await call_next(request)
            request_time = time.time() - start_time

            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=response.status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(request_time)
            
            return response
        except Exception as e:
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
            raise e
        finally:
            REQUEST_IN_PROGRESS.dec()

START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if model is not None and scaler is not None:
        MODEL_LOADED.set(1)
    else:
        MODEL_LOADED.set(0)
    yield


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

app = FastAPI(
    title="Personality Classifier API", 
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(MetricsMiddleware)

try:
    model = joblib.load("models/classifier_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    MODEL_LOADED.set(1)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None
    scaler = None
    MODEL_LOADED.set(0)


@app.get("/")
async def root():
    UPTIME.set(time.time() - START_TIME)
    return {
        "message": "Personality Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - предсказание для одного примера",
            "/predict_batch": "POST - предсказание для нескольких примеров",
            "/health": "GET - проверка здоровья API",
            "/model_info": "GET - информация о модели",
            "/metrics": "GET - Prometheus метрики"
        }
    }

@app.get("/health")
async def health_check():
    UPTIME.set(time.time() - START_TIME)
    
    if model is None or scaler is None:
        MODEL_LOADED.set(0)
        raise HTTPException(status_code=503, detail="Модель не загружена")

    MODEL_LOADED.set(1)
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "timestamp": pd.Timestamp.now().isoformat(),
        "service": "personality-classifier-api",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/metrics")
async def metrics():
    UPTIME.set(time.time() - START_TIME)
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/model_info")
async def model_info():
    UPTIME.set(time.time() - START_TIME)
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return {
        "model_type": model.__class__.__name__,
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown",
        "features": ["Time_broken_spent_Alone","Stage_fear", "Social_event_attendance","Going_outside","Drained_after_socializing","Friends_circle_size","Post_frequency"]
    }

@app.post("/predict")
async def predict(features: PersonalityFeatures):
    UPTIME.set(time.time() - START_TIME)
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        data = pd.DataFrame([features.model_dump()])
        data_scaled = scaler.transform(data)

        probability = model.predict_proba(data_scaled)[0][1]
        prediction = int(model.predict(data_scaled)[0])

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            **{k.lower(): v for k, v in features.dict().items()},
            'prediction': prediction,
            'probability': probability
        }

        log_dir = '/opt/airflow/data/production_logs'
        os.makedirs(log_dir, exist_ok=True)

        log_df = pd.DataFrame([log_entry])
        log_path = os.path.join(log_dir, 'predictions.csv')
        log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

        personality_type = "extrovert" if prediction == 1 else "introvert"
        PREDICTION_PROBABILITY.labels(personality_type=personality_type).observe(probability)
        PREDICTION_DISTRIBUTION.labels(personality_type=personality_type).observe(probability)
        
        return {
            "probability_extrovert": float(probability),
            "prediction": prediction,
            "personality": "Extrovert" if prediction == 1 else "Introvert",
            "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания:{str(e)}")

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    UPTIME.set(time.time() - START_TIME)
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        data = pd.DataFrame([sample.model_dump() for sample in request.samples])
        data_scaled = scaler.transform(data)

        probabilities = model.predict_proba(data_scaled)[:, 1]
        predictions = model.predict(data_scaled)
        
        results = []
        log_entries = []
        
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            personality_type = "extrovert" if pred == 1 else "introvert"

            PREDICTION_PROBABILITY.labels(personality_type=personality_type).observe(prob)
            PREDICTION_DISTRIBUTION.labels(personality_type=personality_type).observe(prob)
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                **{k.lower(): v for k, v in request.samples[i].model_dump().items()},
                'prediction': int(pred),
                'probability': float(prob)
            }
            log_entries.append(log_entry)
            
            results.append({
                "sample_id": i,
                "probability_extrovert": float(prob),
                "prediction": int(pred),
                "personality": "Extrovert" if pred == 1 else "Introvert"
            })
        
        if log_entries:
            log_dir = '/opt/airflow/data/production_logs'
            os.makedirs(log_dir, exist_ok=True)
            log_df = pd.DataFrame(log_entries)
            log_path = os.path.join(log_dir, 'predictions.csv')
            log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
        
        return {
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)