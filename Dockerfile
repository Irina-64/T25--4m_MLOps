FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Создаем виртуальное окружение и устанавливаем Python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Создаем и активируем виртуальное окружение
ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Настройки CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Устанавливаем зависимости для src
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    pandas==2.1.1 \
    numpy==1.26.0 \
    scikit-learn==1.3.1 \
    mlflow==2.7.1 \
    fastapi==0.103.2 \
    uvicorn[standard]==0.23.2 \
    python-multipart==0.0.6

# Если есть requirements.txt, устанавливаем и их
COPY requirements.txt .
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Копируем код и модель
COPY src/ ./src
COPY mlruns/2/f251cdb0d19648d59601ca4ad5bafc57/artifacts/model.pt ./model.pt

# Запускаем inference с переменными окружения для входного и выходного путей
CMD ["python", "src/inference.py", "--input-path", "$INPUT_PATH", "--output-path", "$OUTPUT_PATH"]
