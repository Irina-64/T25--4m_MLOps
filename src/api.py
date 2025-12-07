import os
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIR = Path("models/detox_model")
FALLBACK_MODEL = "t5-small"


class TextIn(BaseModel):
    text: str
    max_length: Optional[int] = 96
    num_beams: Optional[int] = 2


class TextOut(BaseModel):
    detox_text: str


app = FastAPI(title="Detox T5 API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend (React build)
frontend_build_path = Path(__file__).parent.parent / "frontend-build"
frontend_react_path = Path(__file__).parent.parent / "frontend-react"

# Try React build first, then fallback to old frontend
if frontend_build_path.exists():
    # Mount assets directory for React build
    assets_path = frontend_build_path / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")
else:
    # Fallback to old frontend
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        static_files = StaticFiles(directory=str(frontend_path))
        app.mount("/static", static_files, name="static")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REQUEST_COUNT = Counter(
    "detox_request_total",
    "Total HTTP requests for Detox API",
    ["endpoint", "method", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "detox_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "method"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)
PREDICTION_LENGTH = Histogram(
    "detox_prediction_length",
    "Distribution of detoxified text length (characters).",
    buckets=(16, 32, 48, 64, 80, 96, 128, 160, 200, 256),
)

# Disable tokenizer multithreading noise and force CPU usage.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
device = torch.device("cpu")


def load_model():
    path = MODEL_DIR if MODEL_DIR.exists() else FALLBACK_MODEL
    try:
        tok = T5Tokenizer.from_pretrained(path)
        mdl = T5ForConditionalGeneration.from_pretrained(path)
        mdl.eval()
        return tok, mdl, str(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {path}: {exc}")


tokenizer, model, model_source = load_model()
print(f"[API] Model loaded from: {model_source}")


@app.get("/health")
def health():
    started = time.perf_counter()
    status_code = 200
    response = {"status": "ok", "model_source": model_source}
    try:
        return response
    finally:
        elapsed = time.perf_counter() - started
        REQUEST_LATENCY.labels("/health", "GET").observe(elapsed)
        REQUEST_COUNT.labels("/health", "GET", status_code).inc()


@app.post("/predict", response_model=TextOut)
def predict(payload: TextIn):
    started = time.perf_counter()
    status_code = 200

    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    max_len = payload.max_length or 96
    num_beams = payload.num_beams or 2

    try:
        inputs = tokenizer(payload.text, return_tensors="pt", truncation=True, max_length=max_len)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_len,
                num_beams=num_beams,
                length_penalty=1.0,
                early_stopping=True,
            )
        det = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        PREDICTION_LENGTH.observe(len(det))
        return {"detox_text": det}
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    except Exception as exc:
        status_code = 400
        raise HTTPException(status_code=400, detail=f"Unable to process request: {exc}")
    finally:
        elapsed = time.perf_counter() - started
        REQUEST_LATENCY.labels("/predict", "POST").observe(elapsed)
        REQUEST_COUNT.labels("/predict", "POST", status_code).inc()


@app.get("/")
async def root():
    """Serve the frontend index page"""
    # Try React build first
    frontend_build_path = Path(__file__).parent.parent / "frontend-build" / "index.html"
    if frontend_build_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(frontend_build_path))
    
    # Fallback to old frontend
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(frontend_path))
    
    return {"message": "Detox API is running. Frontend not found. Please build React app first."}
