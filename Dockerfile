# базовый образ
FROM python:3.11-slim

# рабочая директория
WORKDIR /app

# копируем зависимости
COPY requirements.txt .

# устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# копируем весь проект
COPY . .

# команда по умолчанию
# CMD для FastAPI
CMD ["uvicorn", "src.Api:app", "--host", "0.0.0.0", "--port", "8000"]

