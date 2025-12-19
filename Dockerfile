FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pandas==2.1.3 \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    pydantic==2.5.0
RUN pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    starlette==0.27.0

COPY src/ ./src/
COPY models/ ./models/
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8080
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
