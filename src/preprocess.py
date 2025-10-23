import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create data/processed directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Чтение data/raw/paradetox.csv
df = pd.read_csv("data/raw/data.csv")


# Разделение train/test (80/20)
train_df, test_df = train_test_split(df[["input_text", "target_text"]], test_size=0.2, random_state=42)

# Сохранить
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)