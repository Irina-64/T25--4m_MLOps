import argparse
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    val_size = config["preprocess"]["val_size"]
    test_size = config["preprocess"]["test_size"]

    df = pd.read_csv(config["data"]["processed"])

    train, temp = train_test_split(
        df, test_size=val_size + test_size, random_state=42, shuffle=True
    )

    val, test = train_test_split(
        temp,
        test_size=test_size / (val_size + test_size),
        random_state=42,
        shuffle=True,
    )

    os.makedirs("data/split", exist_ok=True)
    os.makedirs("data/drift", exist_ok=True)

    train.to_csv("data/split/train.csv", index=False)
    test.to_csv("data/split/test.csv", index=False)
    val.to_csv("data/split/val.csv", index=False)

    sample = val.sample(n=5000, random_state=42)
    sample.to_csv("data/drift/reference.csv", index=False)
    sample.to_csv("data/drift/production.csv", index=False)
