from fastapi import FastAPI, HTTPException, Request
import joblib
import pandas as pd
import os
import time
from prometheus_client import Counter, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

# Инициализация метрик Prometheus
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP Requests', 
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'HTTP request latency',
    ['method', 'endpoint']
)

PREDICTION_DISTRIBUTION = Histogram(
    'prediction_probability', 
    'Prediction probability distribution',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MODEL_LOAD_COUNT = Counter(
    'model_load_total',
    'Total model load attempts',
    ['status']
)

# Middleware для сбора метрик HTTP
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(latency)
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code
            ).inc()
        
        return response

# Инициализация FastAPI приложения
app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

# Добавление middleware
app.add_middleware(PrometheusMiddleware)

# Загрузка модели
def _load_model():
    model_dir = "models"
    
    try:
        possible_paths = [
            os.path.join(model_dir, "model.joblib"),
            os.path.join(model_dir, "telco_churn_model.joblib"),
            os.path.join(model_dir, "logisticregression_model.joblib"),
            os.path.join(model_dir, "randomforest_model.joblib"),
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                print(f"📦 Загружаем модель: {model_path}")
                MODEL_LOAD_COUNT.labels(status='success').inc()
                return joblib.load(model_path)
        
        if not os.path.isdir(model_dir):
            MODEL_LOAD_COUNT.labels(status='error').inc()
            raise FileNotFoundError(f"Model directory '{model_dir}' not found")
        
        candidates = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.joblib')]
        if not candidates:
            MODEL_LOAD_COUNT.labels(status='error').inc()
            raise FileNotFoundError(f"No .joblib models found in '{model_dir}'")
        
        latest = max(candidates, key=os.path.getmtime)
        print(f"📦 Загружаем последнюю модель: {latest}")
        MODEL_LOAD_COUNT.labels(status='success').inc()
        return joblib.load(latest)
        
    except Exception as e:
        MODEL_LOAD_COUNT.labels(status='error').inc()
        raise

try:
    model = _load_model()
    _MODEL_PATH = getattr(model, '_loaded_from', None)
except Exception as e:
    model = None
    _load_error = str(e)

# Эндпоинт для метрик Prometheus
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Эндпоинт для проверки здоровья
@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

# Основной эндпоинт для предсказаний
@app.post("/predict")
def predict(payload: dict):
    """Simple prediction endpoint.
    
    Expects a JSON object that maps feature names to values.
    Returns: {"delay_prob": 0.123}
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {_load_error}")

    # Build DataFrame from payload
    try:
        df = pd.DataFrame([payload])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    # Try predict_proba
    try:
        preds = model.predict_proba(df)[:, 1]
        prediction_prob = float(preds[0])
        
        # Записываем метрику предсказания
        PREDICTION_DISTRIBUTION.observe(prediction_prob)
        
        return {"delay_prob": prediction_prob}
        
    except Exception as e:
        # Attempt to align features if model exposes feature names
        if hasattr(model, 'feature_names_in_'):
            cols = list(model.feature_names_in_)
            for c in cols:
                if c not in df.columns:
                    df[c] = 0
            df = df[cols]
            try:
                preds = model.predict_proba(df)[:, 1]
                prediction_prob = float(preds[0])
                PREDICTION_DISTRIBUTION.observe(prediction_prob)
                return {"delay_prob": prediction_prob}
            except Exception as e2:
                raise HTTPException(status_code=400, detail=f"Prediction failed after aligning features: {e2}")
        else:
            raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# Информационный эндпоинт
@app.get("/")
def root():
    return {
        "message": "Telco Churn Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "POST /predict - Make predictions",
            "GET /metrics - Prometheus metrics",
            "GET /health - Health check",
            "GET /docs - API documentation"
        ],
        "model_loaded": model is not None
    }