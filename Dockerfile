FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirememts.txt

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]