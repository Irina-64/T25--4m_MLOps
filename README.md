## 🔎 Обзор проекта

- src/: скрипты на Python для загрузки данных, предобработки, тренировки модели, оценки и регистрации модели.
- feature_repo/: репозиторий для Feast feature store (offline CSV и SQLite online store).
- dags/: Airflow DAG, orchestrating steps (download, preprocess, train, evaluate, register).
- Dockerfile / docker-compose-airflow.yml / Dockerfile.airflow: конфигурации для локального запуска Airflow и приложения.
- k8s/: манифесты для деплоймента в Kubernetes (Deployment, Service, HPA)
- tests/: pytest тесты (API и DAG structure).
- MLflow: модель и метрики логируются в MLflow при обучении (по умолчанию на http://localhost:5000).

---

## 🚀 Быстрый старт (Windows PowerShell)

### Предварительные шаги

1) Клонируйте репозиторий и перейдите в папку проекта:

	git clone https://github.com/MaxJalo/ML-Ops.git

2) Установите зависимости (лучше в venv/conda env):

	python -m venv .venv; .\\.venv\\Scripts\\Activate; python -m pip install --upgrade pip; pip install -r requirements.txt

### Настройте MLflow (локальный сервер) — опционально

MLflow UI можно запустить локально:

	mlflow ui --port 5000

По умолчанию MLflow будет писать в директорию `mlruns/`.

---

## 🧭 Как запустить конвейер локально

### 1. Запустить Airflow локально (docker-compose)

- Убедитесь, что Docker запущен.
- В каталоге проекта есть `docker-compose-airflow.yml`.

PowerShell:

	docker compose -f docker-compose-airflow.yml up -d

Это запустит веб-сервер и scheduler. Откройте Airflow UI (обычно http://localhost:8080).

> ⚠️ Если у вас возникли проблемы: проверьте Docker Desktop (включены ли ресурсы), убедитесь, что порты свободны.

### 2. Локально запустить API

PowerShell: (в корне проекта), для запуска API без GPU:

	Set-Item -Path Env:SKIP_MODEL_LOAD -Value 1; uvicorn src.api:app --reload --port 8000

- `SKIP_MODEL_LOAD=1` — полезно для тестов и CI: приложение запустится без реальной загрузки `models/model.pt`.
- Чтобы загрузить модель при запуске (если она есть): выключите SKIP_MODEL_LOAD.

### 3. Подготовить / скачать данные и обучить модель

Скрипты:

	python src/download_data.py --raw-path data/raw/churn_predict.csv
	python src/preprocess.py --raw-path data/raw/churn_predict.csv --processed-path data/processed/processed.csv
	python src/train.py --processed-path data/processed/processed.csv

Запуск с MLflow: mlflow сервер должен быть запущен, чтобы видеть результаты работы `mlflow ui`.

---

## 🧪 Тесты и CI

Параметры для запуска тестов (pytest):

	Set-Item -Path Env:SKIP_MODEL_LOAD -Value 1; pytest -q

CI: В репозитории добавлен GitHub Actions workflow `.github/workflows/ci.yml`.

---

## 🔗 Feature Store (Feast)

Репозиторий `feature_repo/` содержит пример конфигурации Feast, который использует CSV как offline store и SQLite как online store.

### Пример команд в PowerShell:

	Set-Location feature_repo
	feast apply
	feast materialize 2025-01-01 2025-12-31

Для подготовки данных (в `src/feast_prepare_features.py`):

	python src/feast_prepare_features.py --raw-path ../data/raw/churn_predict.csv --out-path feature_repo/data/features.csv

Затем:

	cd feature_repo; feast apply; feast materialize 2025-01-01 2025-12-31

### Использование Feast в тренировке

В `src/train.py` есть опция `--use-feast`, которая использует `FeatureStore.get_historical_features()` для формирования датасета обучения.

Пример запуска обучения через Feast:

	python src/train.py --use-feast --start-date 2025-01-01 --end-date 2025-12-31

---

## 🧠 MLflow

- MLflow logs: `mlruns/` (по-умолчанию локально). Если нужен сервер, запустите `mlflow ui`.
- В `src/train.py` добавлено логирование параметров, метрик (например ROC_AUC) и сохранение артефактов (dataset, model) в MLflow.
- Пример:

	python src/train.py --processed-path data/processed/processed.csv

После запуска мб видно эксперимент в `http://localhost:5000`.

---

## 🔌 API

API использует FastAPI и реализует два endpoint'а:

- GET / — приветственная страница
- POST /predict — inference для одного образца
- POST /predict_batch — inference для списка образцов

Пример запроса (PowerShell) — один пример:

	curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"age": 36, "balance": 1000, "churn": 0}'

---

## 🌐 Kubernetes

Приложение имеет пример манифестов в `k8s/`:

- `k8s/deployment.yaml` — Deployment и pod шаблон
- `k8s/service.yaml` — NodePort сервис
- `k8s/hpa.yaml` — Horizontal Pod Autoscaler

Для локального тестирования с Minikube:

	minikube start
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/hpa.yaml

Проверьте сервис:

	kubectl get svc -n default

---

## 🧰 Полезные команды и скрипты

- Run API (dev):
	Set-Item -Path Env:SKIP_MODEL_LOAD -Value 1; uvicorn src.api:app --reload --port 8000

- Run full training (no feast):
	python src/preprocess.py --raw-path data/raw/churn_predict.csv --processed-path data/processed/processed.csv
	python src/train.py --processed-path data/processed/processed.csv

- Run training with Feast (after feast apply & materialize):
	python src/train.py --use-feast --start-date 2025-01-01 --end-date 2025-12-31

- Start local MLflow: mlflow ui --port 5000

- Run tests: Set-Item -Path Env:SKIP_MODEL_LOAD -Value 1; pytest -q

---

## 🧭 Troubleshooting

- PyTorch issues: Убедитесь в корректной версии PyTorch, особенно если вы используете GPU/CUDA (подберите соответствующие wheel'ы с сайта PyTorch).
- Docker: Убедитесь, что Docker запущен и что порты/ресурсы не конфликтуют.
- Airflow: Если webserver/scheduler не появляются, проверьте логи docker-compose: `docker compose -f docker-compose-airflow.yml logs --follow`.
- Feature store (Feast): Если `feast materialize` не возвращает ожидаемых строк, убедитесь, что `feature_repo/data/features.csv` содержит корректные timestamps, entity_id и т.д.

---

## 👩‍💻 Разработка и CI

- Тесты: pytest (tests/test_predict.py, tests/test_root.py, tests/test_dag_structure.py).
- GitHub Actions: `.github/workflows/ci.yml` запускает тесты и pylint checks.

---

## 📌 Что дальше

- Добавить инструкции для запуска end-to-end тестов (feast apply, materialize и train) в CI (опционально).
- Добавить подробные примеры POST requests для `/predict` и `/predict_batch`.

---

Если хотите — могу дополнительно:
- Добавить `scripts/` с командами для fast-start (PowerShell friendly).
- Добавить интеграционные тесты для Feast (feature repo apply/materialize) в CI.


