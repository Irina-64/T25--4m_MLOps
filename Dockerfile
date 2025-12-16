FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]