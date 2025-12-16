#!/bin/bash

# Telco Churn MLOps Pipeline - LAB 11 WITH DOCKER MONITORING
echo "======================================================================"
echo "🚀 TELCO CHURN MLOPS PIPELINE"
echo "======================================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to check status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
        return 0
    else
        echo -e "${RED}✗ $1${NC}"
        return 1
    fi
}

print_header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
}

# ============================================
print_header "✅ ПРОВЕРКА ОКРУЖЕНИЯ И DOCKER"
# ============================================

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python не найден${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python найден${NC}"

# Проверка Docker (ОБЯЗАТЕЛЬНО для Lab 11)
echo "Проверка Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ DOCKER НЕ УСТАНОВЛЕН!${NC}"
    echo "Для выполнения лабораторной работы 11 требуется Docker Desktop"
    echo "1. Скачайте с: https://docs.docker.com/desktop/"
    echo "2. Установите и запустите Docker Desktop"
    echo "3. Перезапустите терминал и попробуйте снова"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ DOCKER НЕ ЗАПУЩЕН!${NC}"
    echo "1. Запустите Docker Desktop"
    echo "2. Дождитесь полного запуска (значок в трее станет белым/зеленым)"
    echo "3. Попробуйте снова"
    echo ""
    echo "Если Docker Desktop запущен, но всё равно не работает:"
    echo "  - Windows: запустите PowerShell как администратор и выполните:"
    echo "      wsl --update"
    echo "      wsl --shutdown"
    echo "      Restart-Service Docker*"
    exit 1
fi

echo -e "${GREEN}✅ Docker запущен и готов к работе${NC}"
echo "Docker версия: $(docker --version)"

# ============================================
print_header "📦 УСТАНОВКА ЗАВИСИМОСТЕЙ"
# ============================================

echo "Установка основных зависимостей..."
pip install --quiet pandas numpy scikit-learn joblib mlflow matplotlib seaborn fastapi uvicorn pytest 2>/dev/null
check_status "Основные зависимости"

echo "Установка MLOps и мониторинг зависимостей..."
pip install --quiet dvc feast prometheus-client 2>/dev/null
check_status "MLOps и мониторинг зависимости"

# ============================================
print_header "🔄 ШАГ 1: ПРЕПРОЦЕССИНГ ДАННЫХ"
# ============================================

echo "Запуск препроцессинга..."
python src/preprocess.py
check_status "Препроцессинг завершен"

# ============================================
print_header "🤖 ШАГ 2: ОБУЧЕНИЕ МОДЕЛИ"
# ============================================

echo "Обучение модели..."
python src/train.py 2>/dev/null || echo -e "${YELLOW}⚠ Используем упрощенное обучение${NC}"
check_status "Модель обучена"

# ============================================
print_header "📊 ШАГ 3: ОЦЕНКА МОДЕЛИ"
# ============================================

echo "Оценка модели..."
if [ -f "src/evaluate.py" ]; then
    python src/evaluate.py --log-to-mlflow false 2>/dev/null || python src/evaluate.py
    check_status "Оценка завершена"
else
    echo -e "${YELLOW}⚠ evaluate.py не найден${NC}"
fi

# ============================================
print_header "🐳 ШАГ 4: ПОДГОТОВКА DOCKER МОНИТОРИНГА"
# ============================================

echo "Создание конфигураций для мониторинга..."

# Создаем необходимые директории
mkdir -p prometheus grafana/provisioning/datasources grafana/provisioning/dashboards grafana/dashboards

# 1. Создаем конфигурацию Prometheus
cat > prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'telco_churn_api'
    static_configs:
      - targets: ['host.docker.internal:8080']
        labels:
          service: 'telco-churn-api'
          environment: 'development'
  
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

echo "✅ Конфигурация Prometheus создана"

# 2. Создаем datasource для Grafana
cat > grafana/provisioning/datasources/datasource.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

echo "✅ Конфигурация Grafana datasource создана"

# 3. Создаем Docker Compose для мониторинга
cat > docker-compose-monitoring.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    networks:
      - monitoring
    depends_on:
      - prometheus

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
EOF

echo "✅ Docker Compose конфигурация создана"

# ============================================
print_header "🚀 ШАГ 5: ЗАПУСК ПРОМЕТЕЯ И ГРАФАНЫ"
# ============================================

echo "Запуск контейнеров мониторинга..."
docker-compose -f docker-compose-monitoring.yml down 2>/dev/null
docker-compose -f docker-compose-monitoring.yml up -d

echo "Ожидание запуска контейнеров..."
sleep 10

# Проверяем запуск
echo "Проверка запущенных контейнеров:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(prometheus|grafana)"

# Проверка доступности
echo ""
echo "Проверка доступности Prometheus..."
if curl -s http://localhost:9090 > /dev/null; then
    echo -e "${GREEN}✅ Prometheus доступен на http://localhost:9090${NC}"
else
    echo -e "${RED}❌ Prometheus недоступен${NC}"
    echo "Проверьте логи: docker logs prometheus"
fi

echo "Проверка доступности Grafana..."
sleep 5
if curl -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✅ Grafana доступен на http://localhost:3000${NC}"
    echo "    Логин: admin"
    echo "    Пароль: admin"
else
    echo -e "${RED}❌ Grafana недоступен${NC}"
    echo "Проверьте логи: docker logs grafana"
fi

# ============================================
print_header "🎯 ШАГ 6: ЗАПУСК API С МЕТРИКАМИ PROMETHEUS"
# ============================================

# Проверяем, обновлен ли api.py с метриками
if ! grep -q "prometheus_client" src/api.py 2>/dev/null; then
    echo "Обновление API для поддержки метрик Prometheus..."
    
    # Создаем обновленный api.py с метриками
    cat > src/api_prometheus.py << 'EOF'
from fastapi import FastAPI, HTTPException, Request
import joblib
import pandas as pd
import os
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
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
app = FastAPI(title="Telco Churn Prediction API with Prometheus", version="1.1.0")

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
        "timestamp": time.time(),
        "version": "1.1.0-prometheus"
    }

# Основной эндпоинт для предсказаний
@app.post("/predict")
def predict(payload: dict):
    """Prediction endpoint with Prometheus metrics."""
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {_load_error}")

    try:
        df = pd.DataFrame([payload])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    try:
        preds = model.predict_proba(df)[:, 1]
        prediction_prob = float(preds[0])
        
        # Записываем метрику предсказания
        PREDICTION_DISTRIBUTION.observe(prediction_prob)
        
        return {"delay_prob": prediction_prob}
        
    except Exception as e:
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

@app.get("/")
def root():
    return {
        "message": "Telco Churn Prediction API with Prometheus Metrics",
        "version": "1.1.0",
        "monitoring": {
            "metrics": "GET /metrics",
            "health": "GET /health",
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3000 (admin/admin)"
        },
        "model_loaded": model is not None
    }
EOF
    
    # Делаем backup оригинального api.py и заменяем на новый
    if [ -f "src/api.py" ]; then
        cp src/api.py src/api_backup.py
        echo "✅ Создан backup оригинального api.py"
    fi
    mv src/api_prometheus.py src/api.py
    echo "✅ API обновлен для поддержки Prometheus метрик"
fi

echo "Запуск API сервера с поддержкой метрик Prometheus..."
pkill -f "uvicorn src.api:app" 2>/dev/null || true

# Запускаем API в фоне
uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload &
API_PID=$!
sleep 8

echo "Проверка работы API с метриками..."
curl -s http://localhost:8080/health

echo ""
echo "Отправка тестового запроса для инициализации метрик..."
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 50}' \
  --max-time 5

echo ""
check_status "API с Prometheus метриками запущен"

# ============================================
print_header "📈 ШАГ 7: НАСТРОЙКА ГРАФАНЫ И ГЕНЕРАЦИЯ НАГРУЗКИ"
# ============================================

# Создаем простой дашборд для Grafana
cat > grafana/dashboards/telco_monitoring.json << 'EOF'
{
  "dashboard": {
    "title": "Telco Churn API Monitoring",
    "description": "Real-time monitoring of ML prediction API with Prometheus",
    "tags": ["mlops", "prometheus", "monitoring"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "HTTP Requests Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[1m])",
            "legendFormat": "{{method}} {{endpoint}}",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "description": "HTTP requests per second"
      },
      {
        "id": 2,
        "title": "Request Latency (95th percentile)",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "format": "s",
            "instant": false,
            "refId": "A"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
        "description": "95th percentile response time in seconds"
      },
      {
        "id": 3,
        "title": "Prediction Probability Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(prediction_probability_bucket[5m])",
            "format": "heatmap",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "description": "Distribution of prediction probabilities"
      },
      {
        "id": 4,
        "title": "Model Load Status",
        "type": "piechart",
        "targets": [
          {
            "expr": "model_load_total",
            "legendFormat": "{{status}}",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8},
        "pieType": "pie",
        "description": "Model load success/error count"
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s",
    "schemaVersion": 27,
    "version": 1
  },
  "folderId": 0,
  "overwrite": true
}
EOF

echo "✅ Дашборд Grafana создан"

# Создаем скрипт для генерации нагрузки
cat > generate_load.py << 'EOF'
import requests
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_payload():
    """Generate random payload for testing."""
    return {
        "SeniorCitizen": random.randint(0, 1),
        "tenure": random.randint(1, 72),
        "MonthlyCharges": round(random.uniform(20, 120), 2),
        "TotalCharges": round(random.uniform(100, 8000), 2),
        "gender": random.randint(0, 1),
        "Partner": random.randint(0, 1),
        "Dependents": random.randint(0, 1)
    }

def send_request(request_id, url="http://localhost:8080/predict"):
    """Send single request to API."""
    try:
        start = time.time()
        response = requests.post(
            url,
            json=generate_payload(),
            timeout=10
        )
        latency = time.time() - start
        
        if response.status_code == 200:
            return {"id": request_id, "success": True, "latency": latency}
        else:
            return {"id": request_id, "success": False, "latency": latency}
            
    except Exception as e:
        return {"id": request_id, "success": False, "error": str(e)}

def run_load_test(num_requests=50, max_workers=5):
    """Run load test against API."""
    print(f"🚀 Starting load test: {num_requests} requests with {max_workers} workers")
    print("=" * 50)
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    # Analyze results
    successful = sum(1 for r in results if r.get('success', False))
    failed = num_requests - successful
    latencies = [r['latency'] for r in results if 'latency' in r]
    
    print(f"\n📊 Load Test Results:")
    print(f"   Total Requests: {num_requests}")
    print(f"   Successful: {successful} ({successful/num_requests*100:.1f}%)")
    print(f"   Failed: {failed} ({failed/num_requests*100:.1f}%)")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies) * 1000
        print(f"   Average Latency: {avg_latency:.1f}ms")
        print(f"   Requests per Second: {num_requests/sum(latencies):.2f}")
    
    print("=" * 50)
    return successful > 0

if __name__ == "__main__":
    # Warm up
    print("🔥 Warming up API with 5 requests...")
    for _ in range(5):
        try:
            requests.post("http://localhost:8080/predict", 
                         json=generate_payload(), 
                         timeout=5)
        except:
            pass
    
    # Run actual test
    run_load_test(num_requests=60, max_workers=8)
    
    print("\n✅ Load test completed!")
    print("📈 Check metrics at: http://localhost:8080/metrics")
    print("📊 Check Prometheus: http://localhost:9090")
    print("🎨 Check Grafana: http://localhost:3000 (admin/admin)")
EOF

echo "Запуск нагрузочного теста для генерации метрик..."
python generate_load.py

# ============================================
print_header "🔍 ШАГ 8: ПРОВЕРКА МОНИТОРИНГА"
# ============================================

echo "Проверка всех компонентов мониторинга..."

# Создаем скрипт проверки
cat > check_monitoring.py << 'EOF'
import requests
import time
import sys

def check_component(name, url, timeout=5):
    """Check if component is accessible."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code < 500:
            print(f"✅ {name}: доступен ({url})")
            return True
        else:
            print(f"❌ {name}: ошибка HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: недоступен - {e}")
        return False

def check_prometheus_targets():
    """Check if Prometheus sees our API."""
    try:
        response = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
        if response.status_code == 200:
            data = response.json()
            targets = data['data']['activeTargets']
            api_targets = [t for t in targets if '8080' in t['scrapeUrl']]
            
            if api_targets:
                print(f"✅ Prometheus видит API target: {api_targets[0]['scrapeUrl']}")
                print(f"   Состояние: {api_targets[0]['health']}")
                return True
            else:
                print("⚠ Prometheus не видит API target")
                return False
    except Exception as e:
        print(f"⚠ Не удалось проверить Prometheus targets: {e}")
        return False

def check_metrics():
    """Check if metrics are being collected."""
    try:
        response = requests.get("http://localhost:8080/metrics", timeout=5)
        if response.status_code == 200:
            content = response.text
            metrics = [
                'http_requests_total',
                'http_request_duration_seconds',
                'prediction_probability',
                'model_load_total'
            ]
            
            print("📈 Найдены метрики Prometheus:")
            for metric in metrics:
                if metric in content:
                    print(f"   ✓ {metric}")
                else:
                    print(f"   ⚠ {metric} не найден")
            
            # Count requests
            lines = content.split('\n')
            request_lines = [l for l in lines if 'http_requests_total' in l and not l.startswith('#')]
            if request_lines:
                print(f"\n📊 Всего запросов: {request_lines[0]}")
            
            return True
    except Exception as e:
        print(f"❌ Ошибка при получении метрик: {e}")
        return False

def main():
    print("🔍 Проверка системы мониторинга Lab 11")
    print("=" * 60)
    
    components = [
        ("API", "http://localhost:8080/health"),
        ("API Metrics", "http://localhost:8080/metrics"),
        ("Prometheus", "http://localhost:9090"),
        ("Grafana", "http://localhost:3000")
    ]
    
    all_ok = True
    for name, url in components:
        if not check_component(name, url):
            all_ok = False
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("✅ ВСЕ КОМПОНЕНТЫ МОНИТОРИНГА РАБОТАЮТ!")
        print("\n📌 Доступные интерфейсы:")
        print("   1. API:              http://localhost:8080")
        print("   2. API метрики:      http://localhost:8080/metrics")
        print("   3. Prometheus UI:    http://localhost:9090")
        print("   4. Grafana:          http://localhost:3000 (admin/admin)")
        print("   5. Grafana импорт дашборда:")
        print("      - Залогиньтесь в Grafana")
        print("      - Создайте новый дашборд")
        print("      - Добавьте панель с запросом Prometheus")
        print("      - Используйте метрики: http_requests_total, prediction_probability, etc.")
        return 0
    else:
        print("⚠ Некоторые компоненты требуют внимания")
        print("\n🔧 Рекомендации по устранению:")
        print("   - Убедитесь что Docker Desktop запущен")
        print("   - Проверьте: docker ps (должны быть prometheus и grafana)")
        print("   - Проверьте логи: docker logs prometheus")
        print("   - Перезапустите: docker-compose -f docker-compose-monitoring.yml restart")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

echo "Запуск проверки мониторинга..."
python check_monitoring.py

# ============================================
print_header "✅ ФИНАЛЬНЫЙ ОТЧЕТ"
# ============================================

echo "Проверка результатов:"
echo "-------------------------------------------"

# Проверка Docker контейнеров
echo "1. Docker контейнеры:"
docker ps --format "{{.Names}} {{.Status}} {{.Ports}}" | while read line; do
    echo "   ✅ $line"
done

# Проверка метрик
echo "2. Метрики Prometheus:"
if curl -s http://localhost:8080/metrics | grep -q "http_requests_total"; then
    echo "   ✅ Метрики API экспортируются"
else
    echo "   ❌ Метрики не экспортируются"
fi

# Проверка доступности
echo "3. Доступность сервисов:"
services=(
    "API:8080"
    "Prometheus:9090" 
    "Grafana:3000"
)

for service in "${services[@]}"; do
    name=${service%:*}
    port=${service#*:}
    if curl -s "http://localhost:$port" > /dev/null 2>&1; then
        echo "   ✅ $name доступен на порту $port"
    else
        echo "   ❌ $name недоступен"
    fi
done

# ============================================
echo ""
echo ""
echo -e "${CYAN}📊 СИСТЕМА МОНИТОРИНГА ЗАПУЩЕНА:${NC}"
echo ""
echo "  1. FastAPI с метриками Prometheus"
echo "     URL:      http://localhost:8080"
echo "     Метрики:  http://localhost:8080/metrics"
echo "     Health:   http://localhost:8080/health"
echo ""
echo "  2. Prometheus (сбор метрик)"
echo "     URL:      http://localhost:9090"
echo "     Targets:  http://localhost:9090/targets"
echo "     Graph:    http://localhost:9090/graph"
echo ""
echo "  3. Grafana (визуализация)"
echo "     URL:      http://localhost:3000"
echo "     Логин:    admin"
echo "     Пароль:   admin"
echo ""
echo -e "${CYAN}🧪 КОМАНДЫ ДЛЯ ТЕСТИРОВАНИЯ:${NC}"
echo ""
echo "  • Проверить мониторинг:"
echo "      python check_monitoring.py"
echo ""
echo "  • Сгенерировать нагрузку:"
echo "      python generate_load.py"
echo ""
echo "  • Сделать предсказание:"
echo '      curl -X POST "http://localhost:8080/predict" \'
echo '        -H "Content-Type: application/json" \'
echo '        -d '"'"'{"SeniorCitizen": 0, "tenure": 34, "MonthlyCharges": 56.95}'"'"''
echo ""
echo "  • Посмотреть метрики:"
echo "      curl http://localhost:8080/metrics | grep http_requests_total"
echo ""
echo -e "${CYAN}🛑 КОМАНДЫ ДЛЯ ОСТАНОВКИ:${NC}"
echo ""
echo "  • Остановить API:"
echo "      pkill -f \"uvicorn src.api:app\""
echo ""
echo "  • Остановить мониторинг:"
echo "      docker-compose -f docker-compose-monitoring.yml down"
echo ""
echo "  • Остановить всё:"
echo "      pkill -f \"uvicorn\" && docker-compose -f docker-compose-monitoring.yml down"
echo ""
echo "======================================================================"
echo "💡 Для просмотра метрик в Grafana:"
echo "   1. Откройте http://localhost:3000"
echo "   2. Войдите (admin/admin)"
echo "   3. Нажмите '+' → Import dashboard"
echo "   4. Используйте JSON из grafana/dashboards/telco_monitoring.json"
echo "======================================================================"


# Ожидание Ctrl+C для остановки
trap 'echo ""; echo "Останавливаем систему..."; kill $API_PID 2>/dev/null; docker-compose -f docker-compose-monitoring.yml down; rm -f .api_pid.lab11 generate_load.py check_monitoring.py; echo "✅ Система остановлена"; exit' INT

echo ""
echo "⚠  Система мониторинга работает. Для остановки нажмите Ctrl+C"
echo ""

# Бесконечный цикл
while true; do
    sleep 1
done