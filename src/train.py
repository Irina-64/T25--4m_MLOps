import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import mlflow
import mlflow.transformers
import torch

# Загрузка данных
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Токенизатор и модель
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def preprocess_function(examples):
    # Tokenize inputs and labels with padding and truncation
    model_inputs = tokenizer(examples["input_text"], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(examples["target_text"], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing with batched processing
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Аргументы тренировки
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
)

# Метрика BLEU через nltk (без сетевых зависимостей)
_smooth_fn = SmoothingFunction().method3

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # NLTK ожидает токенизированные строки; делаем простую split по пробелам
    preds_tok = [p.split() for p in decoded_preds]
    refs_tok = [[l.split()] for l in decoded_labels]
    bleu_score = corpus_bleu(refs_tok, preds_tok, smoothing_function=_smooth_fn)
    return {"bleu": bleu_score}

# Трейнер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


mlflow.set_experiment("text_detox")
with mlflow.start_run():
    mlflow.log_params({"model": "t5-small", "epochs": 1, "batch_size": 8})
    trainer.train()
    eval_results = trainer.evaluate()
    mlflow.log_metrics({"bleu": eval_results.get("eval_bleu", 0.0)})
    mlflow.transformers.log_model(model, "model")

model.save_pretrained("models/detox_model")
tokenizer.save_pretrained("models/detox_model")