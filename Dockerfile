FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Кладём requirements внутрь образа
COPY requirements.txt .

# Обновим pip
RUN pip install --no-cache-dir --upgrade pip

# 1. Ставим PyTorch CPU-версию из официального репозитория PyTorch
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu torchvision==0.19.0+cpu torchaudio==2.4.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Ставим остальные зависимости проекта
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
