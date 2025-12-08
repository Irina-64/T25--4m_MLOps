FROM python:3.10-slim

WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p models data/processed reports

# Открываем порт
EXPOSE 8080

# Запускаем приложение
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
