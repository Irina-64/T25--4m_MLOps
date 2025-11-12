# src/evaluate.py
from pathlib import Path
import json
import pandas as pd
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate as hf_evaluate
from tqdm.auto import tqdm

TEST_CSV = Path("data/processed/test.csv")
MODEL_DIR = Path("models/detox_model")  # <-- ты сохраняешь сюда в train.py
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_JSON = REPORTS_DIR / "eval.json"
SAMPLES_JSONL = REPORTS_DIR / "samples.jsonl"

MAX_LEN = 128
NUM_SAMPLES_TO_SAVE = 20  # сколько примеров положить в отчёт

def main():
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Не найден тестовый датасет: {TEST_CSV}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка модели: {MODEL_DIR}. Сначала запусти training.")

    df = pd.read_csv(TEST_CSV)
    # ожидаем колонки из твоего train.py
    if not {"input_text", "target_text"} <= set(df.columns):
        raise ValueError(f"Ожидаю колонки 'input_text' и 'target_text' в {TEST_CSV}. Найдено: {list(df.columns)}")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    bleu = hf_evaluate.load("bleu")

    preds = []
    refs = []
    samples = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        inp = str(row["input_text"])
        ref = str(row["target_text"])

        inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=MAX_LEN,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        preds.append(pred)
        refs.append([ref])
        if len(samples) < NUM_SAMPLES_TO_SAVE:
            samples.append({"input": inp, "reference": ref, "prediction": pred})

    result = bleu.compute(predictions=preds, references=refs)  # {'bleu': ...}

    report = {
        "metric": "BLEU",
        "bleu": float(result["bleu"]),
        "n_samples": int(len(df)),
        "max_length": MAX_LEN,
        "beam_search": True,
        "num_beams": 4,
    }
    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(SAMPLES_JSONL, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[OK] Сохранил отчёт: {EVAL_JSON}")
    print(f"[OK] Сэмплы: {SAMPLES_JSONL}")

if __name__ == "__main__":
    import torch  # импорт здесь, чтобы не ругался при статическом анализе
    main()
