import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import mlflow
import mlflow.transformers
from evaluate import load
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
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
)

# Метрика BLEU
bleu = load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": result["bleu"]}

# Трейнер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# MLflow
mlflow.set_experiment("text_detox")
with mlflow.start_run():
    mlflow.log_params({"model": "t5-small", "epochs": 1, "batch_size": 256})
    trainer.train()
    eval_results = trainer.evaluate()
    mlflow.log_metrics({"bleu": eval_results["eval_bleu"]})
    mlflow.transformers.log_model(model, "model")

# Сохранение модели локально (опционально)
model.save_pretrained("models/detox_model")
tokenizer.save_pretrained("models/detox_model")