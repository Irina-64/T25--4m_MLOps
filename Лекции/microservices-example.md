# Минимальный пример микросервисного ML API с автотестом и Docker

## Структура проекта

```
ml-microservices-example/
├── services/
│   ├── model_service/
│   │   ├── main.py              # API сервис для инференса
│   │   ├── model.pkl            # обученная модель
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── preprocessing_service/
│   │   ├── main.py              # сервис предобработки
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── monitoring_service/
│       ├── main.py              # сервис мониторинга
│       ├── requirements.txt
│       └── Dockerfile
├── tests/
│   ├── test_model_service.py
│   ├── test_preprocessing_service.py
│   └── conftest.py
├── docker-compose.yml           # оркестрация всех сервисов локально
├── requirements-dev.txt         # зависимости для разработки
└── README.md
```

---

## Сервис 1: Model Service (Inference)

### `services/model_service/main.py`

```python
"""
Model Service - микросервис для инференса ML-модели.
Отвечает за загрузку модели и предсказания.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import logging
import time
from datetime import datetime
from prometheus_client import Counter, Histogram, make_wsgi_app
from functools import wraps

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(
    title="Model Service",
    description="Микросервис для предсказания задержки рейса",
    version="1.0"
)

# Загрузка модели при старте
model = None

@app.on_event("startup")
async def load_model():
    """Загрузка модели при старте приложения"""
    global model
    try:
        model = joblib.load("model.pkl")
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise

# Определение Pydantic моделей для валидации

class FlightData(BaseModel):
    """Входные данные о рейсе"""
    carrier: str = Field(..., min_length=2, max_length=2)
    dep_hour: int = Field(..., ge=0, le=23)
    distance: float = Field(..., gt=0, le=5000)
    weather_delay: float = Field(..., ge=0)
    
    class Config:
        example = {
            "carrier": "AA",
            "dep_hour": 9,
            "distance": 550.0,
            "weather_delay": 0.0
        }

class PredictionResponse(BaseModel):
    """Формат ответа"""
    prediction: int
    probability: float
    timestamp: str
    service_version: str = "1.0"

# Метрики Prometheus
predictions_total = Counter(
    'predictions_total',
    'Total predictions made',
    ['service']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency',
    ['service']
)

# Эндпоинты

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "service": "model_service",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(flight_data: FlightData) -> PredictionResponse:
    """
    Предсказание задержки рейса
    
    Args:
        flight_data: Данные о рейсе
        
    Returns:
        PredictionResponse: Предсказание и вероятность
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Преобразуем входные данные в формат для модели
        features = [[
            flight_data.dep_hour,
            flight_data.distance,
            flight_data.weather_delay
        ]]
        
        # Предсказание
        prediction = model.predict(features)[0]
        probability = float(model.predict_proba(features)[0, 1])
        
        # Логируем метрики
        predictions_total.labels(service="model_service").inc()
        latency = time.time() - start_time
        prediction_latency.labels(service="model_service").observe(latency)
        
        logger.info(f"Prediction made: {prediction}, latency: {latency:.3f}s")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=round(probability, 4),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Эндпоинт для сбора метрик Prometheus"""
    return make_wsgi_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### `services/model_service/requirements.txt`

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
scikit-learn==1.3.2
joblib==1.3.2
prometheus-client==0.18.0
```

### `services/model_service/Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модель
COPY main.py .
COPY model.pkl .

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

# Запуск
EXPOSE 8001
CMD ["python", "main.py"]
```

---

## Сервис 2: Preprocessing Service

### `services/preprocessing_service/main.py`

```python
"""
Preprocessing Service - микросервис для предобработки данных.
Преобразует сырые данные в формат, пригодный для модели.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Preprocessing Service",
    description="Микросервис предобработки данных",
    version="1.0"
)

class RawFlightData(BaseModel):
    """Сырые данные"""
    carrier: str
    departure_time: str  # формат HH:MM
    distance_miles: float
    weather_code: int
    
class ProcessedFlightData(BaseModel):
    """Предобработанные данные"""
    carrier: str
    dep_hour: int
    distance: float
    weather_delay: float

# Маппинги для преобразования

WEATHER_CODE_TO_DELAY = {
    1: 0.0,      # ясно
    2: 5.0,      # облачно
    3: 15.0,     # дождь
    4: 30.0      # гроза
}

@app.get("/health")
async def health_check():
    """Проверка здоровья"""
    return {"status": "healthy", "service": "preprocessing_service"}

@app.post("/preprocess", response_model=ProcessedFlightData)
async def preprocess(data: RawFlightData) -> ProcessedFlightData:
    """
    Предобработка данных
    """
    try:
        # Извлечение часа из времени
        dep_hour = int(data.departure_time.split(":")[0])
        
        # Маппирование кода погоды в задержку
        weather_delay = WEATHER_CODE_TO_DELAY.get(data.weather_code, 0.0)
        
        processed = ProcessedFlightData(
            carrier=data.carrier,
            dep_hour=dep_hour,
            distance=data.distance_miles,
            weather_delay=weather_delay
        )
        
        logger.info(f"Preprocessing completed for carrier {data.carrier}")
        return processed
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_preprocess")
async def batch_preprocess(data_list: List[RawFlightData]) -> List[ProcessedFlightData]:
    """
    Пакетная предобработка
    """
    results = []
    for data in data_list:
        results.append(await preprocess(data))
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### `services/preprocessing_service/requirements.txt`

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
```

### `services/preprocessing_service/Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8002
CMD ["python", "main.py"]
```

---

## Сервис 3: Monitoring Service

### `services/monitoring_service/main.py`

```python
"""
Monitoring Service - сервис для сбора логов и метрик.
Агрегирует информацию со всех других сервисов.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Monitoring Service",
    description="Сервис мониторинга и логирования",
    version="1.0"
)

class ServiceLog(BaseModel):
    """Структура логов"""
    service: str
    level: str  # INFO, WARNING, ERROR
    message: str
    timestamp: str

class ServiceMetric(BaseModel):
    """Структура метрик"""
    service: str
    metric_name: str
    value: float
    timestamp: str

# Хранилище логов в памяти (в реальной системе это база данных)
logs_storage: List[ServiceLog] = []
metrics_storage: List[ServiceMetric] = []

@app.get("/health")
async def health_check():
    """Проверка здоровья"""
    return {"status": "healthy", "service": "monitoring_service"}

@app.post("/logs")
async def log_event(log: ServiceLog):
    """
    Получение логов от других сервисов
    """
    logs_storage.append(log)
    logger.info(f"[{log.service}] {log.level}: {log.message}")
    return {"status": "logged"}

@app.post("/metrics")
async def record_metric(metric: ServiceMetric):
    """
    Получение метрик от других сервисов
    """
    metrics_storage.append(metric)
    logger.info(f"[{metric.service}] {metric.metric_name}: {metric.value}")
    return {"status": "recorded"}

@app.get("/reports/logs")
async def get_logs(service: str = None, limit: int = 100):
    """
    Получить логи
    """
    filtered = logs_storage
    if service:
        filtered = [l for l in filtered if l.service == service]
    return filtered[-limit:]

@app.get("/reports/metrics")
async def get_metrics(service: str = None):
    """
    Получить метрики
    """
    filtered = metrics_storage
    if service:
        filtered = [m for m in filtered if m.service == service]
    
    # Агрегирование по сервисам
    summary = {}
    for metric in filtered:
        if metric.service not in summary:
            summary[metric.service] = []
        summary[metric.service].append({
            "metric": metric.metric_name,
            "value": metric.value
        })
    
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
```

### `services/monitoring_service/requirements.txt`

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
```

---

## Тесты

### `tests/conftest.py`

```python
"""
Конфигурация для pytest
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Добавляем пути к сервисам
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../services/model_service'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../services/preprocessing_service'))

@pytest.fixture
def model_client():
    """Клиент для Model Service"""
    from services.model_service.main import app
    return TestClient(app)

@pytest.fixture
def preprocessing_client():
    """Клиент для Preprocessing Service"""
    from services.preprocessing_service.main import app
    return TestClient(app)
```

### `tests/test_model_service.py`

```python
"""
Тесты для Model Service
"""

import pytest

class TestModelServiceHealth:
    """Тесты health check"""
    
    def test_health_check(self, model_client):
        """Проверка /health эндпоинта"""
        response = model_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

class TestModelServicePrediction:
    """Тесты предсказаний"""
    
    def test_predict_valid_input(self, model_client):
        """Тест валидного предсказания"""
        payload = {
            "carrier": "AA",
            "dep_hour": 9,
            "distance": 550.0,
            "weather_delay": 0.0
        }
        response = model_client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "timestamp" in data
        assert 0 <= data["probability"] <= 1
    
    def test_predict_invalid_carrier(self, model_client):
        """Тест невалидного carrier"""
        payload = {
            "carrier": "A",  # слишком короткий
            "dep_hour": 9,
            "distance": 550.0,
            "weather_delay": 0.0
        }
        response = model_client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_predict_invalid_hour(self, model_client):
        """Тест невалидного часа"""
        payload = {
            "carrier": "AA",
            "dep_hour": 25,  # более 23
            "distance": 550.0,
            "weather_delay": 0.0
        }
        response = model_client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_predict_probability_range(self, model_client):
        """Тест что вероятность в диапазоне [0, 1]"""
        payload = {
            "carrier": "AA",
            "dep_hour": 9,
            "distance": 550.0,
            "weather_delay": 0.0
        }
        response = model_client.post("/predict", json=payload)
        assert response.status_code == 200
        prob = response.json()["probability"]
        assert 0.0 <= prob <= 1.0

class TestModelServiceIntegration:
    """Интеграционные тесты"""
    
    def test_multiple_predictions(self, model_client):
        """Тест множественных предсказаний"""
        payloads = [
            {"carrier": "AA", "dep_hour": 6, "distance": 300.0, "weather_delay": 0.0},
            {"carrier": "UA", "dep_hour": 12, "distance": 1000.0, "weather_delay": 10.0},
            {"carrier": "DL", "dep_hour": 18, "distance": 500.0, "weather_delay": 5.0},
        ]
        
        for payload in payloads:
            response = model_client.post("/predict", json=payload)
            assert response.status_code == 200
```

### `tests/test_preprocessing_service.py`

```python
"""
Тесты для Preprocessing Service
"""

import pytest

class TestPreprocessingServiceHealth:
    """Тесты health check"""
    
    def test_health_check(self, preprocessing_client):
        """Проверка /health эндпоинта"""
        response = preprocessing_client.get("/health")
        assert response.status_code == 200

class TestPreprocessingService:
    """Тесты предобработки"""
    
    def test_preprocess_valid_data(self, preprocessing_client):
        """Тест валидной предобработки"""
        payload = {
            "carrier": "AA",
            "departure_time": "09:30",
            "distance_miles": 550.0,
            "weather_code": 1
        }
        response = preprocessing_client.post("/preprocess", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["carrier"] == "AA"
        assert data["dep_hour"] == 9
        assert data["distance"] == 550.0
        assert data["weather_delay"] == 0.0
    
    def test_preprocess_weather_mapping(self, preprocessing_client):
        """Тест маппирования погоды"""
        test_cases = [
            (1, 0.0),      # ясно
            (2, 5.0),      # облачно
            (3, 15.0),     # дождь
            (4, 30.0)      # гроза
        ]
        
        for weather_code, expected_delay in test_cases:
            payload = {
                "carrier": "AA",
                "departure_time": "09:00",
                "distance_miles": 500.0,
                "weather_code": weather_code
            }
            response = preprocessing_client.post("/preprocess", json=payload)
            assert response.status_code == 200
            assert response.json()["weather_delay"] == expected_delay
    
    def test_batch_preprocess(self, preprocessing_client):
        """Тест пакетной предобработки"""
        payloads = [
            {
                "carrier": "AA",
                "departure_time": "09:30",
                "distance_miles": 550.0,
                "weather_code": 1
            },
            {
                "carrier": "UA",
                "departure_time": "14:00",
                "distance_miles": 1000.0,
                "weather_code": 3
            }
        ]
        response = preprocessing_client.post("/batch_preprocess", json=payloads)
        
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2
```

---

## Docker Compose для локальной разработки

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  # Model Service
  model-service:
    build:
      context: ./services/model_service
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - LOG_LEVEL=INFO
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

  # Preprocessing Service
  preprocessing-service:
    build:
      context: ./services/preprocessing_service
      dockerfile: Dockerfile
    ports:
      - "8002:8002"
    environment:
      - LOG_LEVEL=INFO
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

  # Monitoring Service
  monitoring-service:
    build:
      context: ./services/monitoring_service
      dockerfile: Dockerfile
    ports:
      - "8003:8003"
    environment:
      - LOG_LEVEL=INFO
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

networks:
  ml-network:
    driver: bridge
```

---

## Файл для разработки

### `requirements-dev.txt`

```
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.0
black==23.11.0
flake8==6.1.0
```

---

## README

### `README.md`

```markdown
# Микросервисный ML API - Пример для лекции 8

Минимальный пример архитектуры на базе микросервисов для ML.

## Структура

- **Model Service** — инференс модели (порт 8001)
- **Preprocessing Service** — предобработка данных (порт 8002)
- **Monitoring Service** — логирование и метрики (порт 8003)

## Требования

- Docker
- Docker Compose
- Python 3.10+ (для локальной разработки)

## Запуск

### Вариант 1: Docker Compose (рекомендуется)

```bash
docker-compose up -d
```

### Вариант 2: Локально

```bash
# Установка зависимостей
pip install -r requirements-dev.txt

# Запуск каждого сервиса в отдельном терминале
python services/model_service/main.py
python services/preprocessing_service/main.py
python services/monitoring_service/main.py
```

## Тестирование

```bash
# Запуск всех тестов
pytest tests/

# С покрытием
pytest tests/ --cov=services

# Конкретный тест
pytest tests/test_model_service.py::TestModelServicePrediction::test_predict_valid_input
```

## Примеры запросов

### Health Check

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

### Предсказание

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "carrier": "AA",
    "dep_hour": 9,
    "distance": 550.0,
    "weather_delay": 0.0
  }'
```

### Предобработка

```bash
curl -X POST http://localhost:8002/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "carrier": "AA",
    "departure_time": "09:30",
    "distance_miles": 550.0,
    "weather_code": 1
  }'
```

## Документация

- Model Service: http://localhost:8001/docs
- Preprocessing Service: http://localhost:8002/docs
- Monitoring Service: http://localhost:8003/docs

## Расширение

- Добавить JWT аутентификацию
- Интегрировать Prometheus для метрик
- Настроить логирование в ELK stack
- Развернуть в Kubernetes
```

---

## Быстрый старт

1. **Клонируйте/создайте структуру проекта**

2. **Соберите Docker образы:**
```bash
docker-compose build
```

3. **Запустите все сервисы:**
```bash
docker-compose up
```

4. **Протестируйте в другом терминале:**
```bash
# Проверка здоровья
curl http://localhost:8001/health

# Предсказание
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"carrier":"AA","dep_hour":9,"distance":550.0,"weather_delay":0.0}'
```

5. **Запустите тесты:**
```bash
pytest tests/ -v
```

---

## Ключевые концепции, продемонстрированные в коде

✅ **Микросервисы** — каждый сервис отвечает за одну функцию  
✅ **Независимые Dockerfile** — каждый сервис упакован отдельно  
✅ **Docker Compose** — простая локальная оркестрация  
✅ **Health checks** — каждый сервис проверяется  
✅ **Pydantic валидация** — автоматическая валидация входных данных  
✅ **Единая тестовая база** — pytest с фикстурами  
✅ **Метрики Prometheus** — подготовка к мониторингу  
✅ **API документация** — автоматические OpenAPI/Swagger docs
```

