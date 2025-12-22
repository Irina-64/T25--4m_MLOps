# airflow\Dockerfile
FROM apache/airflow:2.8.1

USER airflow

RUN pip install --user --no-cache-dir \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
# Устанавливаем зависимости через pip в домашнюю директорию airflow
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --user -r /tmp/requirements.txt
