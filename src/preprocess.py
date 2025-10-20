import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create data/processed directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Чтение data/raw/paradetox.csv
df = pd.read_csv("data/raw/paradetox.csv")

# Очистка: удалить NaN, строки короче 5 символов
df = df.dropna()
df = df[(df["toxic_text"].str.len() >= 5) & (df["detoxified_text"].str.len() >= 5)]

# Добавление префикса для seq2seq
df["input_text"] = "detoxify: " + df["toxic_text"]
df["target_text"] = df["detoxified_text"]

# Разделение train/test (80/20)
train_df, test_df = train_test_split(df[["input_text", "target_text"]], test_size=0.2, random_state=42)

# Сохранить
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)