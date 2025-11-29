# Airflow ML Pipeline Tutorial

This tutorial describes how to run an ML pipeline in Apache Airflow using Docker Compose in the project.

Overview
- The DAG `dags/flight_pipeline.py` contains tasks: `download_data`, `preprocess`, `train`, `evaluate`, `register`.
- Each step uses Python scripts in `src/` to perform the logic.
- `docker-compose-airflow.yml` defines a minimal Airflow stack with Postgres and Redis and mounts the repository into `/opt/airflow` inside the container.

Files created
- `dags/flight_pipeline.py`: Airflow DAG.
- `src/download_data.py`: Create a mock dataset if none exists.
- `src/preprocess.py`: Preprocess dataset; now supports `--raw-path` and `--processed-path`.
- `src/train.py`: Training script accepts `--processed-path` and `--model-path`; logs to MLflow and saves `model.pt`.
- `src/evaluate.py`: Evaluates trained model and writes `data/result.json`.
- `src/register.py`: Copies model into the `models/` folder if metric exceeds threshold.
- `docker-compose-airflow.yml`: Docker Compose to run Airflow locally.

Usage

1. Ensure Docker and Docker Compose are installed.
2. Start Airflow stack:

```bash
# Start the stack
docker compose -f docker-compose-airflow.yml up --build -d
# Wait a minute for the webserver to start, then open http://localhost:8080
```

3. Create Airflow Variables (via UI > Admin > Variables) or they will be set to defaults:
- `DATA_RAW`: `/opt/airflow/data/raw/churn_predict.csv`
- `DATA_PROCESSED`: `/opt/airflow/data/processed/processed.csv`
- `MODEL_PATH`: `/opt/airflow/model.pt`
- `MODELS_FOLDER`: `/opt/airflow/models`
- `MODEL_REGISTRATION_THRESHOLD`: `0.6`

4. Trigger DAG `flight_pipeline` from the Airflow UI (or via CLI).

What each script does

- `download_data.py`:
  - If the raw data file (`DATA_RAW`) doesn't exist, it generates a mock dataset of users with transactions.

- `preprocess.py`:
  - Reads the raw csv file, computes day index and stores the processed dataset.

- `train.py`:
  - Reads processed data, trains an LSTM model, logs metrics to MLflow, and saves model checkpoint `model.pt`.

- `evaluate.py`:
  - Loads the model checkpoint and processed data, computes `roc_auc` and writes a JSON result.

- `register.py`:
  - Given `--metric` and `--threshold`, copies model into `models/` if metric >= threshold.

Notes and Tips
- The system uses `SKIP_MODEL_LOAD` variable in tests and conftest to not require a model file when running tests.
- The `dags/flight_pipeline.py` implements a simple conditional register step: evaluate writes `data/result.json`; the register task reads it and copies the model if metric >= threshold.
- In production you may want to move heavy computations (training) to dedicated workers or use `KubernetesExecutor`.

Testing
- You can run scripts locally without Airflow to confirm functionality:

```bash
python src/download_data.py --raw-path data/raw/churn_predict.csv
python src/preprocess.py --raw-path data/raw/churn_predict.csv --processed-path data/processed/processed.csv
python src/train.py --processed-path data/processed/processed.csv --model-path model.pt
python src/evaluate.py --processed-path data/processed/processed.csv --model-path model.pt --out-json data/result.json
python src/register.py --model-path model.pt --metric 0.75 --threshold 0.6 --out-folder models
```

FAQ
- Q: I want to use a real dataset or model registry.
  - A: Change the `download_data` task to download from your remote storage, and change `register` to publish models to your registry.

- Q: How to change the threshold or file paths?
  - A: Use Airflow Variables in the UI or update defaults in `dags/flight_pipeline.py`.

If you need, I can also:
- Add `Dockerfile` for your image that includes required dependencies so it can be used in Airflow worker images.
- Add a small test to check if the DAG runs end-to-end locally with a minimal dataset.

Enjoy running your pipeline!
