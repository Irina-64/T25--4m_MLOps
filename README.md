# Used Cars Price Prediction

MLOps‑проект для предсказания цен на подержанные автомобили на основе данных с **auto.ru**. Репозиторий содержит полный production‑ориентированный ML‑pipeline: подготовку и версионирование данных, обучение и оценку модели, API для инференса, оркестрацию, мониторинг и автоматическую реакцию на деградацию качества.

Проект ориентирован на воспроизводимость, автоматизацию и масштабируемый деплой.

---

## Общая идея

Модель предсказывает цену автомобиля по его характеристикам (марка, год, пробег, параметры двигателя и др.).

Pipeline построен по следующей логике:

**Данные → Признаки → Обучение → Оценка → API → Мониторинг → Drift → Retrain**

---

## Используемые технологии

* **Python 3.12**
* **DVC** — версионирование данных и ML‑пайплайнов
* **MLflow** — эксперименты и Model Registry
* **FastAPI** — REST API для инференса
* **Docker / Docker Compose** — контейнеризация
* **Apache Airflow** — оркестрация пайплайнов
* **Feast** — Feature Store
* **Kubernetes (Minikube)** — деплой и масштабирование
* **Prometheus + Grafana** — мониторинг
* **Pytest** — тестирование

---

## Качество модели

Метрики на тестовой выборке:

| Метрика | Значение |
| ------- | -------- |
| RMSE    | 0.533    |
| MAE     | 0.487    |
| R²      | 0.469    |

---

## Данные и признаки

* Исходные данные хранятся и версионируются через **DVC**
* Предобработка выполняется скриптом `src/preprocess.py`
* Сформированные признаки сохраняются в `data/features/`
* Используется **Feature Store (Feast)** для оффлайн‑доступа к признакам при обучении

---

## Обучение и оценка

Основные скрипты:

* `src/train.py` — базовое обучение модели
* `src/train_best_model.py` — обучение лучшей конфигурации
* `src/hyperopt_tune.py` — подбор гиперпараметров
* `src/evaluate.py` — расчёт метрик и отчётов

Все эксперименты логируются в **MLflow**, модели могут регистрироваться в Model Registry.

---

## API для инференса

* Реализован REST API на **FastAPI** (`src/api.py`)
* Endpoint `/predict` принимает параметры автомобиля и возвращает предсказанную цену
* API инструментировано метриками Prometheus (`/metrics`)

---

## Оркестрация и автоматизация

Используется **Apache Airflow** для управления пайплайнами:

* основной pipeline (preprocess → train → evaluate)
* проверка дрейфа данных и качества
* автоматический retrain при превышении порогов

DAG'и находятся в `airflow/dags/`.

---

## Мониторинг и дрейф

* **Prometheus** собирает метрики API
* **Grafana** используется для визуализации (latency, RPS, распределения предсказаний)
* Скрипт `src/drift_check.py` отслеживает feature / performance drift
* Отчёты сохраняются в `data/drift/`

---

## Деплой

### Docker

```bash
docker build -t car_price_predict .
docker run -p 8000:8000 car_price_predict
```

### Airflow

```bash
cd airflow
docker-compose up
```

### Kubernetes (Minikube)

```bash
minikube start
kubectl apply -f k8s/
```

Поддерживается горизонтальное масштабирование (HPA).

---

## Тестирование

* Используется **pytest**
* Основные тесты API находятся в `tests/test_api.py`

---

## Структура проекта

```bash
.
├── airflow            # Airflow DAG'и и конфигурация
├── data
│   ├── features       # Сформированные признаки
│   └── drift          # Отчёты дрейфа
├── feature_repo       # Feast Feature Store
├── k8s                # Kubernetes манифесты
├── src
│   ├── api.py
│   ├── preprocess.py
│   ├── train.py
│   ├── train_best_model.py
│   ├── evaluate.py
│   ├── drift_check.py
│   └── tools
├── tests
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── prometheus.yml
├── requirements.txt
└── README.md
```

---

## Воспроизведение

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
dvc pull
dvc repro
```

---

## Статус проекта

Проект представляет собой полноценную MLOps‑систему с автоматизированным обучением, деплоем, мониторингом и контролем деградации модели.
