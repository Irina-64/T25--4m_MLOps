from datasets import load_dataset
import pandas as pd
import os

# Create data/raw directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

dataset = load_dataset("s-nlp/paradetox")["train"]
df = pd.DataFrame({
    "toxic_text": dataset["en_toxic_comment"],
    "detoxified_text": dataset["en_neutral_comment"]
})
df.to_csv("data/raw/paradetox.csv", index=False)