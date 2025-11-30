import pandas as pd
import os
import argparse

SEQ_LEN = 335

def preprocess(raw_data_path: str = "data/raw/churn_predict.csv", processed_path: str = "data/processed/processed.csv"):
    df = pd.read_csv(raw_data_path)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - pd.to_datetime("2024-01-01")).dt.days
    df = df[df["day"] >= 0]  # только 2024

    # Ограничиваем диапазон
    df = df[df["day"] < SEQ_LEN]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Processed saved to {processed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-path', default='data/raw/churn_predict.csv')
    parser.add_argument('--processed-path', default='data/processed/processed.csv')
    args = parser.parse_args()
    preprocess(args.raw_path, args.processed_path)
if __name__ == "__main__":
    preprocess()
