import argparse
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_data(df, handle_missing):
    """Очистка данных и приведение типов"""
    df.replace(["nan", "None", ""], np.nan, inplace=True)
    df.dropna(how="all", inplace=True)

    int_cols = ["owners", "year", "price", "mileage", "power", "super_gen_year_from"]
    float_cols = ["displacement", "super_gen_year_to"]

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    if handle_missing:
        df.fillna(0, inplace=True)

    return df


def encode_categorical(df, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = df[col].astype(str)
            df[f"{col}_encoded"] = encoder.fit_transform(df[col])
            encoders[col] = encoder
    return df, encoders


def scale_numerical(df, numerical_cols):
    scaler = StandardScaler()
    existing_cols = [col for col in numerical_cols if col in df.columns]
    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    return df, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    raw_path = config["data"]["raw"]
    processed_path = config["data"]["processed"]

    df = pd.read_csv(raw_path)

    # Очистка
    df = clean_data(df, config["preprocess"]["handle_missing"])

    # Кодирование категориальных
    df, encoders = encode_categorical(df, config["encoding"]["categorical_cols"])

    # Масштабирование числовых
    df, scaler = scale_numerical(df, config["scaling"]["numerical_cols"])

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/encoders.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
