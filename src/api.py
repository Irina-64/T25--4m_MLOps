from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os

# важно: импортировать torch/transformers только когда реально загружаем модель,
# чтобы быстрее стартовал процесс
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

MODEL_DIR = Path("models/detox_model")
FALLBACK_MODEL = "t5-small"  # запасной вариант, если твоей модели нет

class TextIn(BaseModel):
    text: str
    max_length: int | None = 96
    num_beams: int | None = 2

class TextOut(BaseModel):
    detox_text: str

app = FastAPI(title="Detox T5 API", version="1.0.0")

# подавим ворнинги токенайзера и сделаем запуск предсказуемым на CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
device = torch.device("cpu")

def load_model():
    path = MODEL_DIR if MODEL_DIR.exists() else FALLBACK_MODEL
    try:
        tok = T5Tokenizer.from_pretrained(path)
        mdl = T5ForConditionalGeneration.from_pretrained(path)
        mdl.eval()
        return tok, mdl, str(path)
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить модель из {path}: {e}")

tokenizer, model, model_source = load_model()
print(f"[API] Модель загружена из: {model_source}")

@app.get("/health")
def health():
    return {"status": "ok", "model_source": model_source}

@app.post("/predict", response_model=TextOut)
def predict(payload: TextIn):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
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
        return {"detox_text": det}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка инференса: {e}")
