from datasets import load_dataset
import pandas as pd
import os

# Загрузка датасета
ds = load_dataset("thesofakellers/jigsaw-toxic-comment-classification-challenge", split="train")

# Сэмпл 1000 строк
sample_df = ds.to_pandas().sample(n=1000, random_state=42)

# Сохранение
os.makedirs("data/raw", exist_ok=True)
sample_df.to_csv("data/raw/toxic_comments_sample.csv", index=False)
print("Скачан сэмпл: data/raw/toxic_comments_sample.csv")