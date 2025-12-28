from fastapi import FastAPI, HTTPException, Response
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
import os
import logging
import sys
import time
import asyncio

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PROMETHEUS METRICS ====================
# Для лабораторной 11
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from prometheus_client import make_asgi_app  # ← ДОБАВЛЕНО

    # Метрики
    REQUEST_COUNT = Counter(
        'currency_api_requests_total',
        'Total number of HTTP requests to the API',
        ['method', 'endpoint', 'http_status']
    )

    REQUEST_LATENCY = Histogram(
        'currency_api_request_duration_seconds',
        'Histogram of request processing latency in seconds',
        ['method', 'endpoint'],
        buckets=[0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    )

    PREDICTION_DISTRIBUTION = Histogram(
        'currency_api_prediction_probability',
        'Distribution of predicted probability values (delay_prob)',
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    ERROR_COUNT = Counter(
        'currency_api_errors_total',
        'Total number of errors encountered',
        ['error_type']
    )

    PREDICTION_COUNT = Counter(
        'currency_api_predictions_total',
        'Total number of predictions made',
        ['prediction_class']
    )

    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus клиент доступен")

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus клиент не установлен. Метрики недоступны.")

    # Заглушки для метрик
    class DummyMetric:
        def labels(self, **kwargs):
            return self
        def inc(self, amount=1):
            pass
        def observe(self, value):
            pass

    REQUEST_COUNT = REQUEST_LATENCY = PREDICTION_DISTRIBUTION = DummyMetric()
    ERROR_COUNT = PREDICTION_COUNT = DummyMetric()

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Currency Direction Prediction API", 
    description="API для предсказания направления изменения курса валют",
    version="1.0"
)

# ДОБАВЛЯЕМ METRICS APP ЕСЛИ PROMETHEUS ДОСТУПЕН
if PROMETHEUS_AVAILABLE:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

# ==================== МОДЕЛИ ====================
model = None
scaler = None
feature_names = None

def load_models():
    """Загрузка моделей и артефактов"""
    global model, scaler, feature_names
    try:
        logger.info("Загрузка моделей и артефактов...")

        # Проверяем существование файлов
        required_files = [
            "models/best_model.joblib",
            "models/scaler.joblib", 
            "models/feature_names.joblib"
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")

        # Загружаем модели
        model = joblib.load("models/best_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        feature_names = joblib.load("models/feature_names.joblib")

        logger.info(f"Модели загружены. Признаков: {len(feature_names)}")
        return True

    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей: {e}")
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="model_load_error").inc()
        return False

# Загружаем модели при старте
load_models()

# ==================== MIDDLEWARE ДЛЯ МЕТРИК ====================
@app.middleware("http")
async def metrics_middleware(request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.time()
    method = request.method
    endpoint = request.url.path

    try:
        response = await call_next(request)
        latency = time.time() - start_time

        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                http_status=response.status_code
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(latency)

        return response

    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="middleware_error").inc()
        raise

# ==================== ЭНДПОИНТЫ ====================
class PredictionInput(BaseModel):
    USD_RUB: float
    EUR_RUB: float
    GBP_RUB: float
    day_of_week: int
    is_weekend: int
    USD_RUB_lag_1: float
    USD_RUB_lag_2: float
    USD_RUB_lag_3: float
    USD_RUB_lag_5: float
    USD_RUB_lag_7: float
    EUR_RUB_lag_1: float
    EUR_RUB_lag_2: float
    EUR_RUB_lag_3: float
    EUR_RUB_lag_5: float
    EUR_RUB_lag_7: float
    GBP_RUB_lag_1: float
    GBP_RUB_lag_2: float
    GBP_RUB_lag_3: float
    GBP_RUB_lag_5: float
    GBP_RUB_lag_7: float
    USD_RUB_MA_3: float
    USD_RUB_MA_5: float
    USD_RUB_MA_7: float
    EUR_RUB_MA_3: float
    EUR_RUB_MA_5: float
    EUR_RUB_MA_7: float
    GBP_RUB_MA_3: float
    GBP_RUB_MA_5: float
    GBP_RUB_MA_7: float
    USD_RUB_change_1: float
    USD_RUB_change_3: float

@app.get("/")
async def read_root():
    """Главная страница API"""
    return {
        "message": "Currency Direction Prediction API",
        "version": "1.0",
        "status": "running",
        "model_loaded": model is not None,
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "metrics_endpoint": "/metrics",
        "endpoints": {
            "GET /": "Эта страница",
            "GET /health": "Проверка здоровья",
            "POST /predict": "Предсказание курса",
            "GET /metrics": "Метрики Prometheus"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": time.time(),
        "service": "Currency Prediction API"
    }


# @app.get("/metrics")
# async def get_metrics():
#     """Эндпоинт для сбора метрик Prometheus"""
#     if not PROMETHEUS_AVAILABLE:
#         raise HTTPException(
#             status_code=501,
#             detail="Prometheus клиент не установлен. Установите: pip install prometheus-client"
#         )



#     from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


#     return Response(
#         content=generate_latest(),
#         media_type=CONTENT_TYPE_LATEST
#     )

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Предсказание направления изменения курса USD/RUB
    Returns: вероятность роста курса (класс 1)
    """
    if model is None or scaler is None or feature_names is None:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="service_unavailable").inc()
        raise HTTPException(
            status_code=503,
            detail="Сервис временно недоступен. Модели не загружены."
        )

    start_time = time.time()

    try:
        # Конвертируем входные данные в DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])

        # Убеждаемся что порядок признаков правильный
        df = df[feature_names]

        # Масштабируем признаки
        scaled_data = scaler.transform(df)

        # Предсказание
        prediction_proba = model.predict_proba(scaled_data)

        # Вероятность класса 1 (рост курса)
        delay_prob = float(prediction_proba[0, 1])

        # Логируем метрики предсказания
        if PROMETHEUS_AVAILABLE:
            PREDICTION_DISTRIBUTION.observe(delay_prob)

            # Определяем класс предсказания
            prediction_class = "rise" if delay_prob > 0.5 else "fall"
            PREDICTION_COUNT.labels(prediction_class=prediction_class).inc()

        processing_time = time.time() - start_time

        return {
            "delay_prob": delay_prob,
            "prediction": "rise" if delay_prob > 0.5 else "fall",
            "confidence": delay_prob if delay_prob > 0.5 else 1 - delay_prob,
            "features_used": len(feature_names),
            "model_loaded": True,
            "processing_time_seconds": round(processing_time, 4),
            "prometheus_metrics": PROMETHEUS_AVAILABLE
        }

    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Ошибка при предсказании: {e}")

        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type=error_type).inc()

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}"
        )

# Тестовый эндпоинт для проверки
@app.get("/test")
async def test_endpoint():
    """Тестовый эндпоинт для проверки работы сервера"""
    return {
        "message": "Сервер работает!",
        "timestamp": time.time(),
        "model_status": "loaded" if model is not None else "not loaded"
    }

# ==================== ПРАВИЛЬНЫЙ ЗАПУСК СЕРВЕРА ====================
def run_server():
    """Функция запуска сервера"""
    import uvicorn

    print("=" * 60)
    print("ЗАПУСК CURRENCY PREDICTION API")
    print("=" * 60)
    print("Доступные эндпоинты:")
    print("  • http://localhost:8080/ - Главная страница")
    print("  • http://localhost:8080/health - Проверка здоровья")
    print("  • http://localhost:8080/predict - Предсказание курса")
    print("  • http://localhost:8080/metrics - Метрики Prometheus")
    print("  • http://localhost:8080/test - Тестовый эндпоинт")
    print("=" * 60)
    print("Нажмите Ctrl+C для остановки сервера")
    print("=" * 60)

    # Проверяем загрузку моделей
    if model is None:
        print("ВНИМАНИЕ: Модели не загружены!")
        print("Проверьте наличие файлов в папке models/:")
        print("  - best_model.joblib")
        print("  - scaler.joblib")
        print("  - feature_names.joblib")

    # Запускаем сервер
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()