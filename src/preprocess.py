import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 335
RAW_DATA = Path("data/raw/churn_predict.csv")
PROCESSED_DATA = Path("data/processed/processed.csv")

def preprocess():
    df = pd.read_csv(RAW_DATA)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - pd.to_datetime("2024-01-01")).dt.days
    df = df[df["day"] >= 0]  # только 2024

    # Ограничиваем диапазон
    df = df[df["day"] < SEQ_LEN]
    df.to_csv(PROCESSED_DATA, index=False)
if __name__ == "__main__":
    preprocess()
