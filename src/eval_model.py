import pandas as pd
import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from evaluate import load
import torch
import mlflow
from detoxify import Detoxify  # Для оценки токсичности (pip install detoxify, если нужно; но в вашем env может не быть, так что опционально)

# Создать директорию reports, если нет
os.makedirs("reports", exist_ok=True)

# Загрузка данных
test_df = pd.read_csv("data/processed/test.csv")
test_dataset = Dataset.from_pandas(test_df)

# Токенизатор и модель (загружаем из локальной директории)
model_path = "models/model.safetensors"  # Или путь к model.safetensors, если это отдельный файл: T5ForConditionalGeneration.from_pretrained("path/to/dir")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Функция для генерации предсказаний
def generate_predictions(examples):
    inputs = tokenizer(examples["input_text"], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs

# Применить генерацию (батчами для эффективности)
predictions = []
batch_size = 16
for i in range(0, len(test_dataset), batch_size):
    batch = test_dataset[i:i+batch_size]
    batch_preds = generate_predictions(batch)
    predictions.extend(batch_preds)

# Истинные значения
references = test_df["target_text"].tolist()

# Метрики
bleu = load("bleu")
rouge = load("rouge")

bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
rouge_result = rouge.compute(predictions=predictions, references=references)

metrics = {
    "bleu": bleu_result["bleu"],
    "rouge1": rouge_result["rouge1"],
    "rouge2": rouge_result["rouge2"],
    "rougeL": rouge_result["rougeL"],
}

# Опционально: Toxicity score (если detoxify установлен)
if 'Detoxify' in globals():
    detox = Detoxify('original')
    toxic_scores_input = detox.predict(test_df["toxic_text"].tolist())["toxicity"]
    toxic_scores_output = detox.predict(predictions)["toxicity"]
    metrics["avg_toxicity_input"] = sum(toxic_scores_input) / len(toxic_scores_input)
    metrics["avg_toxicity_output"] = sum(toxic_scores_output) / len(toxic_scores_output)

# Сохранить в JSON
with open("reports/eval.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Сохранить в HTML (простая таблица)
html_content = "<html><body><h1>Evaluation Report</h1><table border='1'>"
for key, value in metrics.items():
    html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
html_content += "</table></body></html>"
with open("reports/eval.html", "w") as f:
    f.write(html_content)

# Логировать в MLflow (опционально, если в run)
with mlflow.start_run():
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("reports/eval.json")
    mlflow.log_artifact("reports/eval.html")

print("Evaluation completed. Reports saved in reports/")