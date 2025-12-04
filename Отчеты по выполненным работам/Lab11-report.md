# Лабораторная работа 11  
## Мониторинг сервиса: Prometheus + Grafana + метрики приложения

### Цель работы
Инструментировать FastAPI-сервис метриками Prometheus, настроить сбор данных Prometheus и визуализацию ключевых метрик в Grafana. В результате должен быть создан dashboard, отображающий производительность и поведение модели в режиме реального времени.

---

## 1. Инструментирование FastAPI-сервиса

В файле `src/api.py` были добавлены метрики через библиотеку `prometheus_client`:

### Добавленные метрики:
- **REQUEST_COUNT** — количество запросов на каждый endpoint  
- **REQUEST_LATENCY** — гистограмма времени обработки запросов  
- **PREDICTION_DISTRIBUTION** — гистограмма длины сгенерированного текста модели

```python
REQUEST_COUNT = Counter(
    "detox_request_total",
    "Number of requests received",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "detox_request_latency_seconds",
    "Request latency distribution",
    ["endpoint"]
)

PREDICTION_DDIST = Histogram(
    "detox_prediction_length_bucket",
    "Length of generated detoxed text",
    buckets=[5,10,20,30,40,60,80,120]
)
```

Метрики автоматически доступны по адресу:
```bash
/metrics
```
2. Конфигурация Prometheus

Создан файл prometheus.yml, в котором настроен таргет:
```yaml
scrape_configs:
  - job_name: "api"
    scrape_interval: 5s
    static_configs:
      - targets: ["api:8080"]
```

Prometheus собирает метрики контейнера API каждые 5 секунд.

3. Docker Compose для мониторинга

Создан файл docker-compose.monitoring.yaml, содержащий сервисы:
    api — наше FastAPI приложение с метриками
    prometheus — собирает метрики
    grafana — визуализация данных
    Grafana автоматически поднимается на:
```bash
http://localhost:3000
```

4. Настройка Grafana и построение Dashboard
В Grafana была добавлена data source: Prometheus → http://prometheus:9090
Создан dashboard с тремя основными графиками:

    1. Requests per second
    PromQL:
```scss
sum(rate(detox_request_total[1m])) by (endpoint)
```
Отображает количество запросов к /predict и /health.

    2. Latency quantiles (p50, p90, p95)
    PromQL:
```scss
histogram_quantile(0.95,
  sum(rate(detox_request_latency_seconds_bucket[5m])) by (le, endpoint)
)
```
Позволяет отслеживать время реакции модели.

    3. Prediction length (p50)
    PromQL:
```scss
histogram_quantile(
  0.5,
  sum(rate(detox_prediction_length_bucket[5m])) by (le)
)
```
    Показывает распределение длины сгенерированного текста.

5. Нагрузочное тестирование
Для генерации нагрузки были отправлены серии запросов:
```powershell
for ($i = 0; $i -lt 100; $i++) {
  Invoke-RestMethod `
    -Uri "http://localhost:8080/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"text":"I hate this movie so much"}' | Out-Null
}
```
6. Полученные результаты
После нагрузки:
    Requests per second отображает резкий рост ~1.4 req/s
    p95 latency достигает ~450–500 ms (ожидаемо при T5-small)
    Prediction length показывает стабильную длину текста ~30 токенов
    Dashboard визуализирует все ключевые показатели, соответствующие требованиям лабораторной.

Провков Иван — Инструментация API: добавление метрик detox_request_total, detox_request_latency_seconds, detox_prediction_length, монтирование /metrics.
Власюк Данил — Конфигурация Prometheus: prometheus.yml, alert.rules.yml, проверка таргетов и правил.
Беспалый Максим — Docker Compose для мониторинга: сервисы API/Prometheus/Grafana, тома, healthcheck.
Скрыпник Михаил — Grafana: провиженинг datasource/dashboards, дашборд с rps, p50/p90/p95, длиной предсказаний.
Яковенко Максим — Тестирование и демонстрация: генерация нагрузки (быстрая/медленная), проверка /metrics, Prometheus targets, алертов и дашборда.

